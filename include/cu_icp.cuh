#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <omp.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>

//This is a library for chuncked based cuda icp only
//1. Chunking the point cloud
//2. Search the nearest neighbors
//3. calculate correspondences error form the nearest neighbors
//4. calculate the transformation using optimization(like Gauss-newton or Levenberg-Marquardt) (we can use GPU to accelerate the optimization)
//5. update the transformation
//6. repeat the process until convergence

class CU_ICP
{
    public:
        // 4080 SUPER 최적화 파라미터
        struct HardwareParams {
            static const int MAX_THREADS_PER_BLOCK = 1024;
            static const int WARP_SIZE = 32;
            static const int NUM_SMS = 80;
            static const int OPTIMAL_OCCUPANCY = 32;
            static const size_t CHUNK_SIZE = 1 << 20;  // 1MB
            static const int NUM_STREAMS = 16;
            static const size_t L2_CACHE_LINE_SIZE = 128;
        };

        struct ICPParams {
            float voxel_size;
            int max_iter;
            float tolerance;
            float max_correspondence_distance;
            
            static ICPParams getDefault() {
                ICPParams params;
                params.voxel_size = 0.1f;
                params.max_iter = 30;
                params.tolerance = 1e-6f;
                params.max_correspondence_distance = 1.0f;
                return params;
            }
        };

        struct ChunkData {
            std::vector<float> points;
            size_t start_idx;
            size_t num_points;
        };

        CU_ICP(const ICPParams& params = ICPParams::getDefault());
        ~CU_ICP();

        void setInputSource(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
        void setInputTarget(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
        void align(pcl::PointCloud<pcl::PointXYZ> &output);

    private:
        struct StreamContext {
            cudaStream_t stream;
            cudaEvent_t compute_complete;
            cudaEvent_t transfer_complete;
            float* pinned_host_buffer;
            float* device_buffer;
            float* d_correspondences;
            float* d_transformation;
            bool is_busy;
        };

        void initializeStreams();
        void cleanupStreams();
        void processPointCloudAsync();
        void processGPUChunks();
        void checkCudaErrors(cudaError_t error, const char* file, int line);

        // CUDA 커널 선언
        __global__ void kernelNearestNeighborSearch(
            const float* __restrict__ source,
            const float* __restrict__ target,
            float* __restrict__ correspondences,
            const int num_points,
            const float max_distance
        );

        __global__ void kernelComputeTransformation(
            const float* __restrict__ source,
            const float* __restrict__ correspondences,
            float* __restrict__ transformation,
            const int num_points
        );

        // 멤버 변수
        ICPParams params;
        std::vector<StreamContext> stream_contexts;
        float* d_target_global;
        float* d_source_global;
        float* d_result_global;
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud;
        pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud;
        int num_points;
        int num_chunks;

        pcl::PointCloud<pcl::PointXYZ>::Ptr source;
        pcl::PointCloud<pcl::PointXYZ>::Ptr target;
        pcl::PointCloud<pcl::PointXYZ>::Ptr output;
        pcl::PointCloud<pcl::PointXYZ>::Ptr correspondences;
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_source;
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_target;

        //parameters
        float voxel_size;
        int max_iter;
        float tolerance;
        float max_correspondence_distance;
        float transformation_epsilon;
        float max_correspondence_distance;

        //cuda
        float *d_source;
        float *d_target;
        float *d_output;
        float *d_correspondences;
        float *d_transformed_source;
        float *d_transformed_target;

        //cuda kernel
        __global__ void cu_icp_kernel(float *d_source, float *d_target, float *d_output, float *d_correspondences, float *d_transformed_source, float *d_transformed_target);
        __global__ void cu_icp_kernel_search_neighbors(float *d_source, float *d_target, float *d_correspondences, float *d_transformed_source, float *d_transformed_target);
        __global__ void cu_icp_kernel_calculate_correspondences(float *d_source, float *d_target, float *d_correspondences, float *d_transformed_source, float *d_transformed_target);
        __global__ void cu_icp_kernel_calculate_transformation(float *d_source, float *d_target, float *d_correspondences, float *d_transformed_source, float *d_transformed_target);
        __global__ void cu_icp_kernel_update_transformation(float *d_source, float *d_target, float *d_correspondences, float *d_transformed_source, float *d_transformed_target);

        // 청크 처리를 위한 멤버 변수 추가
        std::queue<ChunkData> chunk_queue;
        std::mutex queue_mutex;
        std::condition_variable queue_cv;
        bool processing_complete;
        std::thread gpu_thread;
}

// CUDA 에러 체크 매크로
#define CUDA_CHECK(err) checkCudaErrors(err, __FILE__, __LINE__)





