#include "cu_icp.cuh"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <omp.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>

CU_ICP::CU_ICP(const ICPParams& params) : params(params), processing_complete(false) {
    initializeStreams();
}

CU_ICP::~CU_ICP() {
    // GPU 스레드 정리
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        processing_complete = true;
    }
    queue_cv.notify_all();
    if (gpu_thread.joinable()) {
        gpu_thread.join();
    }
    
    cleanupStreams();
}

void CU_ICP::initializeStreams() {
    for (int i = 0; i < HardwareParams::NUM_STREAMS; ++i) {
        StreamContext ctx;
        ctx.is_busy = false;
        CUDA_CHECK(cudaStreamCreateWithFlags(&ctx.stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&ctx.compute_complete, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ctx.transfer_complete, cudaEventDisableTiming));
        
        // 페이지 고정 호스트 메모리 할당
        CUDA_CHECK(cudaMallocHost(&ctx.pinned_host_buffer, 
                                 HardwareParams::CHUNK_SIZE * sizeof(float)));
        
        // 디바이스 메모리 할당
        CUDA_CHECK(cudaMalloc(&ctx.device_buffer, 
                             HardwareParams::CHUNK_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctx.d_correspondences, 
                             HardwareParams::CHUNK_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ctx.d_transformation, 
                             16 * sizeof(float)));
        
        stream_contexts.push_back(ctx);
    }
}

void CU_ICP::cleanupStreams() {
    for (auto& ctx : stream_contexts) {
        CUDA_CHECK(cudaStreamDestroy(ctx.stream));
        CUDA_CHECK(cudaEventDestroy(ctx.compute_complete));
        CUDA_CHECK(cudaEventDestroy(ctx.transfer_complete));
        CUDA_CHECK(cudaFreeHost(ctx.pinned_host_buffer));
        CUDA_CHECK(cudaFree(ctx.device_buffer));
        CUDA_CHECK(cudaFree(ctx.d_correspondences));
        CUDA_CHECK(cudaFree(ctx.d_transformation));
    }
    stream_contexts.clear();
}

void CU_ICP::setInputSource(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    source_cloud = cloud;
    num_points = cloud->points.size();
    
    // 전역 소스 포인트클라우드 메모리 할당
    CUDA_CHECK(cudaMalloc(&d_source_global, 
                         num_points * sizeof(float) * 3));
    
    // 데이터 전송
    CUDA_CHECK(cudaMemcpy(d_source_global, 
                         cloud->points.data(), 
                         num_points * sizeof(float) * 3, 
                         cudaMemcpyHostToDevice));
    
    // 청크 수 계산
    num_chunks = (num_points * sizeof(float) * 3 + HardwareParams::CHUNK_SIZE - 1) 
                 / HardwareParams::CHUNK_SIZE;
}

void CU_ICP::setInputTarget(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    target_cloud = cloud;
    
    // 타겟 포인트클라우드 메모리 할당 및 전송
    CUDA_CHECK(cudaMalloc(&d_target_global, 
                         cloud->points.size() * sizeof(float) * 3));
    CUDA_CHECK(cudaMemcpy(d_target_global, 
                         cloud->points.data(), 
                         cloud->points.size() * sizeof(float) * 3, 
                         cudaMemcpyHostToDevice));
}

__global__ void CU_ICP::kernelNearestNeighborSearch(
    const float* __restrict__ source,
    const float* __restrict__ target,
    float* __restrict__ correspondences,
    const int num_points,
    const float max_distance
) {
    __shared__ float shared_target[256 * 3];  // L1 캐시 활용
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_points) return;
    
    // 소스 포인트 로드
    float sx = source[tid * 3];
    float sy = source[tid * 3 + 1];
    float sz = source[tid * 3 + 2];
    
    float min_dist = max_distance;
    int min_idx = -1;
    
    // 타겟 포인트클라우드와 비교
    for (int i = 0; i < num_points; i += blockDim.x) {
        // 공유 메모리에 타겟 포인트 로드
        if (i + threadIdx.x < num_points) {
            shared_target[threadIdx.x * 3] = target[(i + threadIdx.x) * 3];
            shared_target[threadIdx.x * 3 + 1] = target[(i + threadIdx.x) * 3 + 1];
            shared_target[threadIdx.x * 3 + 2] = target[(i + threadIdx.x) * 3 + 2];
        }
        __syncthreads();
        
        // 최근접 포인트 검색
        for (int j = 0; j < blockDim.x && i + j < num_points; ++j) {
            float tx = shared_target[j * 3];
            float ty = shared_target[j * 3 + 1];
            float tz = shared_target[j * 3 + 2];
            
            float dx = sx - tx;
            float dy = sy - ty;
            float dz = sz - tz;
            float dist = dx * dx + dy * dy + dz * dz;
            
            if (dist < min_dist) {
                min_dist = dist;
                min_idx = i + j;
            }
        }
        __syncthreads();
    }
    
    // 대응점 저장
    if (min_idx >= 0) {
        correspondences[tid * 3] = target[min_idx * 3];
        correspondences[tid * 3 + 1] = target[min_idx * 3 + 1];
        correspondences[tid * 3 + 2] = target[min_idx * 3 + 2];
    }
}

void CU_ICP::processPointCloudAsync() {
    const int num_cpu_threads = omp_get_max_threads();
    std::vector<ChunkData> cpu_chunks(num_cpu_threads);
    
    // CPU에서 OpenMP를 사용한 청크 분할
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        size_t points_per_thread = num_points / num_cpu_threads;
        size_t start_idx = thread_id * points_per_thread;
        size_t end_idx = (thread_id == num_cpu_threads - 1) ? 
                        num_points : start_idx + points_per_thread;

        ChunkData& chunk = cpu_chunks[thread_id];
        chunk.start_idx = start_idx;
        chunk.num_points = end_idx - start_idx;
        chunk.points.resize(chunk.num_points * 3);

        // 포인트 복사 및 전처리
        #pragma omp for schedule(dynamic)
        for (size_t i = start_idx; i < end_idx; ++i) {
            size_t local_idx = i - start_idx;
            chunk.points[local_idx * 3] = source_cloud->points[i].x;
            chunk.points[local_idx * 3 + 1] = source_cloud->points[i].y;
            chunk.points[local_idx * 3 + 2] = source_cloud->points[i].z;
        }

        // 청크 큐에 추가
        #pragma omp critical
        {
            chunk_queue.push(std::move(chunk));
            queue_cv.notify_one();
        }
    }

    // GPU 처리 스레드 시작
    gpu_thread = std::thread([this]() {
        processGPUChunks();
    });
}

void CU_ICP::processGPUChunks() {
    while (true) {
        ChunkData chunk;
        
        // 청크 큐에서 데이터 가져오기
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cv.wait(lock, [this]() {
                return !chunk_queue.empty() || processing_complete;
            });
            
            if (chunk_queue.empty() && processing_complete) {
                break;
            }
            
            chunk = std::move(chunk_queue.front());
            chunk_queue.pop();
        }

        // 스트림 선택
        int stream_idx = chunk.start_idx % HardwareParams::NUM_STREAMS;
        StreamContext& ctx = stream_contexts[stream_idx];

        // 비동기 메모리 전송
        CUDA_CHECK(cudaMemcpyAsync(
            ctx.device_buffer,
            chunk.points.data(),
            chunk.points.size() * sizeof(float),
            cudaMemcpyHostToDevice,
            ctx.stream
        ));
        CUDA_CHECK(cudaEventRecord(ctx.transfer_complete, ctx.stream));

        // 커널 실행
        const int threads_per_block = 256;
        dim3 grid((chunk.num_points + threads_per_block - 1) / threads_per_block);
        dim3 block(threads_per_block);

        kernelNearestNeighborSearch<<<grid, block, 0, ctx.stream>>>(
            ctx.device_buffer,
            d_target_global,
            ctx.d_correspondences,
            chunk.num_points,
            params.max_correspondence_distance
        );

        kernelComputeTransformation<<<grid, block, 0, ctx.stream>>>(
            ctx.device_buffer,
            ctx.d_correspondences,
            ctx.d_transformation,
            chunk.num_points
        );

        CUDA_CHECK(cudaEventRecord(ctx.compute_complete, ctx.stream));

        // 결과 비동기 복사
        CUDA_CHECK(cudaMemcpyAsync(
            ((float*)source_cloud->points.data()) + chunk.start_idx * 3,
            ctx.device_buffer,
            chunk.num_points * sizeof(float) * 3,
            cudaMemcpyDeviceToHost,
            ctx.stream
        ));
    }

    // 모든 스트림 동기화
    for (auto& ctx : stream_contexts) {
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    }
}

void CU_ICP::align(pcl::PointCloud<pcl::PointXYZ>& output) {
    // 결과 저장을 위한 메모리 할당
    CUDA_CHECK(cudaMalloc(&d_result_global, 
                         num_points * sizeof(float) * 3));
    
    for (int iter = 0; iter < params.max_iter; ++iter) {
        processPointCloudAsync();
        
        // GPU 스레드 완료 대기
        if (gpu_thread.joinable()) {
            gpu_thread.join();
        }
        
        // 수렴 검사
        // ... 수렴 검사 로직 구현 ...
    }
    
    // 최종 결과 복사
    output.points.resize(num_points);
    CUDA_CHECK(cudaMemcpy(output.points.data(),
                         d_result_global,
                         num_points * sizeof(float) * 3,
                         cudaMemcpyDeviceToHost));
    
    // 결과 메모리 해제
    CUDA_CHECK(cudaFree(d_result_global));
}

void CU_ICP::checkCudaErrors(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", 
                file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

int main(){
    CU_ICP cu_icp;

    pcl::PointCloud<pcl::PointXYZ>::Ptr source(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr target(new pcl::PointCloud<pcl::PointXYZ>);


    
    cu_icp.setInputSource(source);
    cu_icp.setInputTarget(target);
    cu_icp.align(*source);

    return 0;
}