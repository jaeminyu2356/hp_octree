#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <chrono>
#include <omp.h>
#include <limits>

class ParallelICP {
    public:
        ParallelICP() {
            // ICP 기본 파라미터 설정 (SerialICP와 동일)
            icp_.setMaxCorrespondenceDistance(2.0);
            icp_.setMaximumIterations(3000);
            icp_.setTransformationEpsilon(1e-3);
            
            num_threads_ = omp_get_max_threads();
            omp_set_num_threads(num_threads_);
        }

        Eigen::Matrix4f align(const pcl::PointCloud<pcl::PointXYZI>::Ptr& source,
                            const pcl::PointCloud<pcl::PointXYZI>::Ptr& target,
                            int64_t& computation_time) {
            
            auto start = std::chrono::high_resolution_clock::now();

            // 스레드당 2개의 청크로 분할
            size_t optimal_chunks = num_threads_ * 2;
            size_t chunk_size = source->size() / optimal_chunks;
            std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> source_chunks;
            source_chunks.reserve(optimal_chunks);

            // OpenMP task를 사용한 청크 생성 병렬화
            #pragma omp parallel
            {
                #pragma omp single
                {
                    for(size_t i = 0; i < source->size(); i += chunk_size) {
                        #pragma omp task shared(source_chunks)
                        {
                            pcl::PointCloud<pcl::PointXYZI>::Ptr chunk(new pcl::PointCloud<pcl::PointXYZI>);
                            size_t end = std::min(i + chunk_size, source->size());
                            chunk->points.reserve(end - i);  // 메모리 pre-allocation
                            
                            for(size_t j = i; j < end; j++) {
                                chunk->points.push_back(source->points[j]);
                            }

                            #pragma omp critical
                            source_chunks.push_back(chunk);
                        }
                    }
                    #pragma omp taskwait
                }
            }

            // 동적 스케줄링을 사용한 ICP 병렬 실행
            std::vector<Eigen::Matrix4f> transforms(source_chunks.size());
            
            #pragma omp parallel for schedule(dynamic)
            for(size_t i = 0; i < source_chunks.size(); i++) {
                pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> local_icp;
                local_icp.setMaxCorrespondenceDistance(icp_.getMaxCorrespondenceDistance());
                local_icp.setMaximumIterations(icp_.getMaximumIterations());
                local_icp.setTransformationEpsilon(icp_.getTransformationEpsilon());
                
                local_icp.setInputSource(source_chunks[i]);
                local_icp.setInputTarget(target);
                
                pcl::PointCloud<pcl::PointXYZI> final;
                local_icp.align(final);
                
                transforms[i] = local_icp.getFinalTransformation();
            }

            // Reduction을 사용한 변환 행렬 평균 계산 병렬화
            Eigen::Matrix4f final_transform = Eigen::Matrix4f::Zero();
            #pragma omp parallel
            {
                Eigen::Matrix4f local_sum = Eigen::Matrix4f::Zero();
                
                #pragma omp for schedule(static) nowait
                for(size_t i = 0; i < transforms.size(); i++) {
                    local_sum += transforms[i];
                }
                
                #pragma omp critical
                final_transform += local_sum;
            }
            final_transform /= static_cast<float>(transforms.size());

            // 메모리 정리
            for(auto& chunk : source_chunks) {
                chunk.reset();
            }

            auto end = std::chrono::high_resolution_clock::now();
            computation_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            return final_transform;
        }

        // ICP 파라미터 설정 함수들 (SerialICP와 동일)
        void setMaxCorrespondenceDistance(double distance) {
            icp_.setMaxCorrespondenceDistance(distance);
        }

        void setMaximumIterations(int iterations) {
            icp_.setMaximumIterations(iterations);
        }

        void setTransformationEpsilon(double epsilon) {
            icp_.setTransformationEpsilon(epsilon);
        }

    private:
        pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp_;
        int num_threads_;

        Eigen::Matrix4f computeInitialAlignment(
            const pcl::PointCloud<pcl::PointXYZI>::Ptr& source,
            const pcl::PointCloud<pcl::PointXYZI>::Ptr& target) {
            
            // 간단한 중심점 기반 초기 정렬
            Eigen::Vector4f source_centroid, target_centroid;
            pcl::compute3DCentroid(*source, source_centroid);
            pcl::compute3DCentroid(*target, target_centroid);
            
            Eigen::Matrix4f initial_transform = Eigen::Matrix4f::Identity();
            initial_transform.block<3,1>(0,3) = target_centroid.head<3>() - source_centroid.head<3>();
            
            return initial_transform;
        }
};
