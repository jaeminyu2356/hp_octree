#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include <cusolverDn.h>
#include <vector>

// CUDA kernel for computing nearest neighbors
__global__ void findNearestNeighborsKernel(
    const float* source_points,
    const float* target_points,
    int* correspondences,
    float* distances,
    const int num_source,
    const int num_target
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_source) return;

    float min_dist = INFINITY;
    int best_match = -1;
    
    float src_x = source_points[idx * 4];
    float src_y = source_points[idx * 4 + 1];
    float src_z = source_points[idx * 4 + 2];

    for (int j = 0; j < num_target; j++) {
        float tgt_x = target_points[j * 4];
        float tgt_y = target_points[j * 4 + 1];
        float tgt_z = target_points[j * 4 + 2];

        float dx = src_x - tgt_x;
        float dy = src_y - tgt_y;
        float dz = src_z - tgt_z;
        
        float dist = dx*dx + dy*dy + dz*dz;
        
        if (dist < min_dist) {
            min_dist = dist;
            best_match = j;
        }
    }

    correspondences[idx] = best_match;
    distances[idx] = min_dist;
}

// CUDA kernel for computing point-to-point error
__global__ void computeTransformationKernel(
    const float* source_points,
    const float* target_points,
    const int* correspondences,
    float* centroids,
    float* covariance,
    const int num_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    int corr_idx = correspondences[idx];
    if (corr_idx < 0) return;

    // Load source point
    float sx = source_points[idx * 4];
    float sy = source_points[idx * 4 + 1];
    float sz = source_points[idx * 4 + 2];

    // Load target point
    float tx = target_points[corr_idx * 4];
    float ty = target_points[corr_idx * 4 + 1];
    float tz = target_points[corr_idx * 4 + 2];

    // Atomic add to compute centroids
    atomicAdd(&centroids[0], sx);
    atomicAdd(&centroids[1], sy);
    atomicAdd(&centroids[2], sz);
    atomicAdd(&centroids[3], tx);
    atomicAdd(&centroids[4], ty);
    atomicAdd(&centroids[5], tz);

    // Compute contribution to covariance matrix
    float cov[9];
    cov[0] = sx * tx; cov[1] = sx * ty; cov[2] = sx * tz;
    cov[3] = sy * tx; cov[4] = sy * ty; cov[5] = sy * tz;
    cov[6] = sz * tx; cov[7] = sz * ty; cov[8] = sz * tz;

    for (int i = 0; i < 9; i++) {
        atomicAdd(&covariance[i], cov[i]);
    }
}

// CUDA kernel for parallel reduction
__global__ void reduceSum(float* input, float* output, int N) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    sdata[tid] = 0;
    
    if (i < N) {
        sdata[tid] = input[i];
        if (i + blockDim.x < N) 
            sdata[tid] += input[i + blockDim.x];
    }
    __syncthreads();
    
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// CUDA kernel for transforming points
__global__ void transformPointsKernel(
    float* points,
    const float* transform,
    int num_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    float x = points[idx * 4];
    float y = points[idx * 4 + 1];
    float z = points[idx * 4 + 2];
    float w = 1.0f;

    float new_x = transform[0] * x + transform[4] * y + transform[8] * z + transform[12] * w;
    float new_y = transform[1] * x + transform[5] * y + transform[9] * z + transform[13] * w;
    float new_z = transform[2] * x + transform[6] * y + transform[10] * z + transform[14] * w;

    points[idx * 4] = new_x;
    points[idx * 4 + 1] = new_y;
    points[idx * 4 + 2] = new_z;
}

// CUDA kernel for splitting point cloud into chunks
__global__ void splitPointCloudKernel(
    const float* source_points,
    float* chunk_points,
    const int* chunk_offsets,
    const int chunk_size,
    const int num_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    int chunk_idx = idx / chunk_size;
    int local_idx = idx % chunk_size;
    int chunk_offset = chunk_offsets[chunk_idx];

    // Copy point to its chunk position
    for (int i = 0; i < 4; i++) {
        chunk_points[chunk_offset * 4 + local_idx * 4 + i] = source_points[idx * 4 + i];
    }
}

// CUDA kernel for averaging transforms
__global__ void averageTransformsKernel(
    const float* transforms,
    float* avg_transform,
    const int num_transforms
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 16) return;  // 4x4 matrix elements

    float sum = 0.0f;
    for (int i = 0; i < num_transforms; i++) {
        sum += transforms[i * 16 + idx];
    }
    avg_transform[idx] = sum / num_transforms;
}

class CUICP {
public:
    CUICP() : max_iterations_(50), distance_threshold_(0.05), min_points_per_chunk_(1000) {
        cudaMalloc(&d_temp_storage_, sizeof(float) * 1024);  // For reduction
        cusolverDnCreate(&cusolver_handle_);
    }

    ~CUICP() {
        if (cusolver_handle_) cusolverDnDestroy(cusolver_handle_);
        cudaFree(d_temp_storage_);
    }

    void setMaxIterations(int max_iter) { max_iterations_ = max_iter; }
    void setDistanceThreshold(float threshold) { distance_threshold_ = threshold; }
    void setMinPointsPerChunk(int min_points) { min_points_per_chunk_ = min_points; }

    Eigen::Matrix4f align(pcl::PointCloud<pcl::PointXYZI>::Ptr source,
                         pcl::PointCloud<pcl::PointXYZI>::Ptr target) {
        int num_source = source->size();
        int num_target = target->size();

        // Calculate optimal number of chunks
        int num_chunks = std::max(1, num_source / min_points_per_chunk_);
        int chunk_size = (num_source + num_chunks - 1) / num_chunks;

        // Allocate device memory for all chunks
        float *d_source_points, *d_target_points;
        float *d_chunk_points;  // Buffer for chunk points
        int *d_chunk_offsets;   // Offset for each chunk
        float *d_chunk_transforms;  // Transforms for each chunk

        cudaMalloc(&d_source_points, num_source * 4 * sizeof(float));
        cudaMalloc(&d_target_points, num_target * 4 * sizeof(float));
        cudaMalloc(&d_chunk_points, num_source * 4 * sizeof(float));
        cudaMalloc(&d_chunk_offsets, num_chunks * sizeof(int));
        cudaMalloc(&d_chunk_transforms, num_chunks * 16 * sizeof(float));

        // Copy point clouds to device
        copyPointCloudToDevice(source, d_source_points, num_source);
        copyPointCloudToDevice(target, d_target_points, num_target);

        // Prepare chunk offsets
        std::vector<int> h_chunk_offsets(num_chunks);
        for (int i = 0; i < num_chunks; i++) {
            h_chunk_offsets[i] = i * chunk_size;
        }
        cudaMemcpy(d_chunk_offsets, h_chunk_offsets.data(), num_chunks * sizeof(int), cudaMemcpyHostToDevice);

        // Split source points into chunks
        const int block_size = 256;
        const int num_blocks = (num_source + block_size - 1) / block_size;
        
        splitPointCloudKernel<<<num_blocks, block_size>>>(
            d_source_points,
            d_chunk_points,
            d_chunk_offsets,
            chunk_size,
            num_source
        );

        // Process each chunk
        for (int chunk = 0; chunk < num_chunks; chunk++) {
            int chunk_points = (chunk == num_chunks - 1) ? 
                             num_source - chunk * chunk_size : chunk_size;

            float *d_chunk_source = d_chunk_points + chunk * chunk_size * 4;
            float *d_chunk_transform = d_chunk_transforms + chunk * 16;
            
            processChunk(d_chunk_source, d_target_points, chunk_points, num_target, d_chunk_transform);
        }

        // Average all transforms
        float *d_final_transform;
        cudaMalloc(&d_final_transform, 16 * sizeof(float));

        averageTransformsKernel<<<1, 16>>>(
            d_chunk_transforms,
            d_final_transform,
            num_chunks
        );

        // Copy final transform back to host
        Eigen::Matrix4f final_transform;
        cudaMemcpy(final_transform.data(), d_final_transform, 16 * sizeof(float), cudaMemcpyDeviceToHost);

        // Cleanup
        cudaFree(d_source_points);
        cudaFree(d_target_points);
        cudaFree(d_chunk_points);
        cudaFree(d_chunk_offsets);
        cudaFree(d_chunk_transforms);
        cudaFree(d_final_transform);

        return final_transform;
    }

private:
    int max_iterations_;
    float distance_threshold_;
    int min_points_per_chunk_;
    cusolverDnHandle_t cusolver_handle_;
    float* d_temp_storage_;

    void processChunk(float* d_chunk_source, float* d_target_points,
                     int num_chunk_points, int num_target, float* d_chunk_transform) {
        // Allocate chunk-specific memory
        float *d_distances;
        int *d_correspondences;
        cudaMalloc(&d_distances, num_chunk_points * sizeof(float));
        cudaMalloc(&d_correspondences, num_chunk_points * sizeof(int));

        const int block_size = 256;
        const int num_blocks = (num_chunk_points + block_size - 1) / block_size;

        Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();

        for (int iter = 0; iter < max_iterations_; iter++) {
            findNearestNeighborsKernel<<<num_blocks, block_size>>>(
                d_chunk_source,
                d_target_points,
                d_correspondences,
                d_distances,
                num_chunk_points,
                num_target
            );

            float *d_centroids, *d_covariance;
            cudaMalloc(&d_centroids, 6 * sizeof(float));
            cudaMalloc(&d_covariance, 9 * sizeof(float));
            cudaMemset(d_centroids, 0, 6 * sizeof(float));
            cudaMemset(d_covariance, 0, 9 * sizeof(float));

            computeTransformationKernel<<<num_blocks, block_size>>>(
                d_chunk_source,
                d_target_points,
                d_correspondences,
                d_centroids,
                d_covariance,
                num_chunk_points
            );

            Eigen::Matrix4f iter_transform = computeTransformationMatrix(
                d_centroids, d_covariance, num_chunk_points);

            float *d_iter_transform;
            cudaMalloc(&d_iter_transform, 16 * sizeof(float));
            cudaMemcpy(d_iter_transform, iter_transform.data(), 16 * sizeof(float), cudaMemcpyHostToDevice);

            transformPointsKernel<<<num_blocks, block_size>>>(
                d_chunk_source,
                d_iter_transform,
                num_chunk_points
            );

            float error = computeError(d_distances, num_chunk_points);
            
            transform = iter_transform * transform;

            if (error < distance_threshold_) break;

            cudaFree(d_centroids);
            cudaFree(d_covariance);
            cudaFree(d_iter_transform);
        }

        // Copy final transform for this chunk
        cudaMemcpy(d_chunk_transform, transform.data(), 16 * sizeof(float), cudaMemcpyHostToDevice);

        // Cleanup chunk-specific memory
        cudaFree(d_distances);
        cudaFree(d_correspondences);
    }

    void copyPointCloudToDevice(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
        float* d_points,
        int num_points
    ) {
        std::vector<float> h_points(num_points * 4);
        for (int i = 0; i < num_points; i++) {
            h_points[i * 4] = cloud->points[i].x;
            h_points[i * 4 + 1] = cloud->points[i].y;
            h_points[i * 4 + 2] = cloud->points[i].z;
            h_points[i * 4 + 3] = cloud->points[i].intensity;
        }
        cudaMemcpy(d_points, h_points.data(), num_points * 4 * sizeof(float), cudaMemcpyHostToDevice);
    }

    float computeError(float* d_distances, int num_points) {
        const int block_size = 256;
        const int num_blocks = (num_points + block_size * 2 - 1) / (block_size * 2);
        
        reduceSum<<<num_blocks, block_size, block_size * sizeof(float)>>>(
            d_distances, d_temp_storage_, num_points);
        
        float total_error;
        cudaMemcpy(&total_error, d_temp_storage_, sizeof(float), cudaMemcpyDeviceToHost);
        
        return total_error / num_points;
    }

    Eigen::Matrix4f computeTransformationMatrix(
        float* d_centroids,
        float* d_covariance,
        int num_points
    ) {
        // Copy data back to host
        float h_centroids[6];
        float h_covariance[9];
        cudaMemcpy(h_centroids, d_centroids, 6 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_covariance, d_covariance, 9 * sizeof(float), cudaMemcpyDeviceToHost);

        // Compute centroids
        Eigen::Vector3f source_centroid(
            h_centroids[0] / num_points,
            h_centroids[1] / num_points,
            h_centroids[2] / num_points
        );
        Eigen::Vector3f target_centroid(
            h_centroids[3] / num_points,
            h_centroids[4] / num_points,
            h_centroids[5] / num_points
        );

        // Compute covariance matrix
        Eigen::Matrix3f covariance;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                covariance(i, j) = h_covariance[i * 3 + j] / num_points -
                    source_centroid[i] * target_centroid[j];
            }
        }

        // Compute SVD
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(
            covariance,
            Eigen::ComputeFullU | Eigen::ComputeFullV
        );

        // Compute rotation
        Eigen::Matrix3f rotation = svd.matrixV() * svd.matrixU().transpose();

        // Handle reflection case
        if (rotation.determinant() < 0) {
            Eigen::Matrix3f V = svd.matrixV();
            V.col(2) *= -1;
            rotation = V * svd.matrixU().transpose();
        }

        // Compute translation
        Eigen::Vector3f translation = target_centroid - rotation * source_centroid;

        // Create transformation matrix
        Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
        transform.block<3, 3>(0, 0) = rotation;
        transform.block<3, 1>(0, 3) = translation;

        return transform;
    }
};