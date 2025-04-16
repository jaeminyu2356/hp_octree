#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <chrono>
#include <omp.h>
#include <limits>
#include <Eigen/Geometry>

class ParallelICP {
    public:
        ParallelICP() {
            // ICP 기본 파라미터 설정
            icp_.setMaxCorrespondenceDistance(1.0);  // 증가된 correspondence distance
            icp_.setMaximumIterations(1000);
            icp_.setTransformationEpsilon(1e-3);
            
            num_threads_ = omp_get_max_threads();
            omp_set_num_threads(num_threads_);
            min_points_per_chunk_ = 1000;  // 최소 청크 크기 설정
        }

        inline Eigen::Matrix4f align(const pcl::PointCloud<pcl::PointXYZI>::Ptr& source,
                            const pcl::PointCloud<pcl::PointXYZI>::Ptr& target,
                            Eigen::Matrix4f& init_transform,
                            int64_t& computation_time) {
            
            auto start = std::chrono::high_resolution_clock::now();

            Eigen::Vector3f delta = Eigen::Vector3f::Zero();
            Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();

            if (init_transform == Eigen::Matrix4f::Identity()) {
                delta(0) = 1.0;
                init_guess.block<3,1>(0,3) = delta;
            }
            else {
                init_guess.block<3,1>(0,3) = init_transform.block<3,1>(0,3);
            }

            // 청크 크기 계산 수정
            size_t optimal_chunks = std::min(static_cast<size_t>(num_threads_), source->size() / min_points_per_chunk_);
            if (optimal_chunks == 0) optimal_chunks = 1;
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
            std::vector<Eigen::Matrix4f> transforms;
            transforms.reserve(source_chunks.size());
            std::vector<bool> valid_transforms(source_chunks.size(), false);
            
            #pragma omp parallel for schedule(dynamic)
            for(size_t i = 0; i < source_chunks.size(); i++) {
                pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> local_icp;
                local_icp.setMaxCorrespondenceDistance(icp_.getMaxCorrespondenceDistance());
                local_icp.setMaximumIterations(icp_.getMaximumIterations());
                local_icp.setTransformationEpsilon(icp_.getTransformationEpsilon());
                
                local_icp.setInputSource(source_chunks[i]);
                local_icp.setInputTarget(target);
                
                pcl::PointCloud<pcl::PointXYZI> final;
                local_icp.align(final, init_guess);

                if (local_icp.hasConverged()) {
                    #pragma omp critical
                    {
                        transforms.push_back(local_icp.getFinalTransformation());
                        valid_transforms[i] = true;
                    }
                }
            }

            // Check if we have any valid transforms
            if (transforms.empty()) {
                std::cerr << "Warning: No valid transformations found. Using initial guess." << std::endl;
                return init_guess;
            }

            // Reduction을 사용한 변환 행렬 평균 계산 병렬화
            Eigen::Matrix4f final_transform = averageTransforms(transforms);

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
        size_t min_points_per_chunk_;  // 최소 청크 크기

        // Helper function to decompose transformation matrix
        inline void decomposeTransform(const Eigen::Matrix4f& transform, 
                              Eigen::Vector3f& translation, 
                              Eigen::Quaternionf& rotation) {
            translation = transform.block<3,1>(0,3);
            rotation = Eigen::Quaternionf(transform.block<3,3>(0,0));
            rotation.normalize();  // Ensure unit quaternion
        }

        // Helper function to compose transformation matrix
        inline Eigen::Matrix4f composeTransform(const Eigen::Vector3f& translation,
                                       const Eigen::Quaternionf& rotation) {
            Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
            transform.block<3,3>(0,0) = rotation.toRotationMatrix();
            transform.block<3,1>(0,3) = translation;
            return transform;
        }

        inline Eigen::Matrix4f averageTransforms(const std::vector<Eigen::Matrix4f>& transforms) {
            if (transforms.empty()) return Eigen::Matrix4f::Identity();
            if (transforms.size() == 1) return transforms[0];

            // Decompose all transforms
            std::vector<Eigen::Vector3f> translations;
            std::vector<Eigen::Quaternionf> rotations;
            translations.reserve(transforms.size());
            rotations.reserve(transforms.size());

            for (const auto& transform : transforms) {
                Eigen::Vector3f trans;
                Eigen::Quaternionf rot;
                decomposeTransform(transform, trans, rot);
                translations.push_back(trans);
                rotations.push_back(rot);
            }

            // Average translations
            Eigen::Vector3f avg_translation = Eigen::Vector3f::Zero();
            for (const auto& trans : translations) {
                avg_translation += trans;
            }
            avg_translation /= static_cast<float>(translations.size());

            // Average rotations using quaternion spherical interpolation
            Eigen::Quaternionf avg_rotation = rotations[0];
            float w = 1.0f / static_cast<float>(rotations.size());

            for (size_t i = 1; i < rotations.size(); ++i) {
                // Ensure quaternions are in the same hemisphere
                if (avg_rotation.dot(rotations[i]) < 0) {
                    rotations[i].coeffs() = -rotations[i].coeffs();
                }
                avg_rotation = avg_rotation.slerp(w * static_cast<float>(i + 1), rotations[i]);
            }
            avg_rotation.normalize();

            return composeTransform(avg_translation, avg_rotation);
        }

};