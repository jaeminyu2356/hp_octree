#include <omp_icp.hpp>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <omp.h>
#include <iostream>
#include <chrono>
#include <limits>

void findNearestPointsParallel(const PointCloudT::Ptr& source, 
                              const PointCloudT::Ptr& target,
                              std::vector<int>& correspondences) {
    int source_size = source->points.size();
    correspondences.resize(source_size);
    
    #pragma omp parallel for
    for (int i = 0; i < source_size; i++) {
        double min_dist = std::numeric_limits<double>::max();
        int min_idx = -1;
        
        const PointT& src_point = source->points[i];
        
        for (int j = 0; j < target->points.size(); j++) {
            const PointT& tgt_point = target->points[j];
            double dist = (src_point.x - tgt_point.x) * (src_point.x - tgt_point.x) +
                         (src_point.y - tgt_point.y) * (src_point.y - tgt_point.y) +
                         (src_point.z - tgt_point.z) * (src_point.z - tgt_point.z);
            
            if (dist < min_dist) {
                min_dist = dist;
                min_idx = j;
            }
        }
        
        correspondences[i] = min_idx;
    }
}

double computeErrorParallel(const PointCloudT::Ptr& source, 
                           const PointCloudT::Ptr& target,
                           const std::vector<int>& correspondences) {
    double total_error = 0.0;
    int valid_points = 0;
    
    #pragma omp parallel
    {
        double local_error = 0.0;
        int local_valid = 0;
        
        #pragma omp for nowait
        for (int i = 0; i < source->points.size(); i++) {
            if (correspondences[i] >= 0) {
                const PointT& src_point = source->points[i];
                const PointT& tgt_point = target->points[correspondences[i]];
                
                double dist = (src_point.x - tgt_point.x) * (src_point.x - tgt_point.x) +
                             (src_point.y - tgt_point.y) * (src_point.y - tgt_point.y) +
                             (src_point.z - tgt_point.z) * (src_point.z - tgt_point.z);
                
                local_error += dist;
                local_valid++;
            }
        }
        
        #pragma omp critical
        {
            total_error += local_error;
            valid_points += local_valid;
        }
    }
    
    return valid_points > 0 ? total_error / valid_points : 0.0;
}

Eigen::Matrix4f performParallelICP(const PointCloudT::Ptr& source, 
                                  const PointCloudT::Ptr& target,
                                  int max_iterations,
                                  double convergence_threshold) {
    // 초기 변환 행렬
    Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
    PointCloudT::Ptr transformed_source(new PointCloudT);
    pcl::transformPointCloud(*source, *transformed_source, transformation);
    
    double prev_error = std::numeric_limits<double>::max();
    
    for (int iter = 0; iter < max_iterations; iter++) {
        // 1. 대응점 찾기 (병렬화)
        std::vector<int> correspondences;
        findNearestPointsParallel(transformed_source, target, correspondences);
        
        // 2. 오차 계산 (병렬화)
        double current_error = computeErrorParallel(transformed_source, target, correspondences);
        
        // 수렴 확인
        if (std::abs(prev_error - current_error) < convergence_threshold) {
            std::cout << "ICP converged at iteration " << iter << std::endl;
            break;
        }
        
        prev_error = current_error;
        
        // 3. 변환 행렬 계산
        Eigen::Matrix4f step_transform = Eigen::Matrix4f::Identity();
        
        // 대응점을 이용한 SVD 계산
        Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
        Eigen::Vector3f mean_src = Eigen::Vector3f::Zero();
        Eigen::Vector3f mean_tgt = Eigen::Vector3f::Zero();
        int valid_points = 0;
        
        // 평균 계산
        for (int i = 0; i < transformed_source->points.size(); i++) {
            if (correspondences[i] >= 0) {
                mean_src += Eigen::Vector3f(transformed_source->points[i].x,
                                          transformed_source->points[i].y,
                                          transformed_source->points[i].z);
                mean_tgt += Eigen::Vector3f(target->points[correspondences[i]].x,
                                          target->points[correspondences[i]].y,
                                          target->points[correspondences[i]].z);
                valid_points++;
            }
        }
        
        if (valid_points == 0) {
            std::cout << "No valid correspondences found!" << std::endl;
            break;
        }
        
        mean_src /= valid_points;
        mean_tgt /= valid_points;
        
        // 공분산 행렬 계산
        for (int i = 0; i < transformed_source->points.size(); i++) {
            if (correspondences[i] >= 0) {
                Eigen::Vector3f src_point(transformed_source->points[i].x,
                                        transformed_source->points[i].y,
                                        transformed_source->points[i].z);
                Eigen::Vector3f tgt_point(target->points[correspondences[i]].x,
                                        target->points[correspondences[i]].y,
                                        target->points[correspondences[i]].z);
                
                src_point -= mean_src;
                tgt_point -= mean_tgt;
                
                covariance += src_point * tgt_point.transpose();
            }
        }
        
        // SVD 분해
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3f rotation = svd.matrixV() * svd.matrixU().transpose();
        
        // 반사 행렬 확인
        if (rotation.determinant() < 0) {
            Eigen::Matrix3f V = svd.matrixV();
            V.col(2) *= -1;
            rotation = V * svd.matrixU().transpose();
        }
        
        Eigen::Vector3f translation = mean_tgt - rotation * mean_src;
        
        // 변환 행렬 업데이트
        step_transform.block<3, 3>(0, 0) = rotation;
        step_transform.block<3, 1>(0, 3) = translation;
        
        // 전체 변환 행렬 업데이트
        transformation = step_transform * transformation;
        
        // 소스 포인트 클라우드 변환
        pcl::transformPointCloud(*source, *transformed_source, transformation);
        
        std::cout << "Iteration " << iter << ", Error: " << current_error << std::endl;
    }
    
    return transformation;
}

int runICPComparison(const std::string& source_file, 
                    const std::string& target_file, 
                    int num_threads) {
    // OpenMP 스레드 수 설정
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    omp_set_num_threads(num_threads);
    std::cout << "Using " << num_threads << " OpenMP threads" << std::endl;
    
    // 포인트 클라우드 로드
    PointCloudT::Ptr source_cloud(new PointCloudT);
    PointCloudT::Ptr target_cloud(new PointCloudT);
    
    if (pcl::io::loadPCDFile<PointT>(source_file, *source_cloud) == -1) {
        std::cerr << "Failed to load source cloud: " << source_file << std::endl;
        return -1;
    }
    
    if (pcl::io::loadPCDFile<PointT>(target_file, *target_cloud) == -1) {
        std::cerr << "Failed to load target cloud: " << target_file << std::endl;
        return -1;
    }
    
    std::cout << "Loaded source cloud with " << source_cloud->size() << " points" << std::endl;
    std::cout << "Loaded target cloud with " << target_cloud->size() << " points" << std::endl;
    
    // 비교를 위한 PCL의 ICP 실행 (시리얼)
    std::cout << "\n===== PCL ICP (Serial) =====" << std::endl;
    pcl::IterativeClosestPoint<PointT, PointT> icp;
    icp.setInputSource(source_cloud);
    icp.setInputTarget(target_cloud);
    PointCloudT pcl_aligned;
    
    auto pcl_start_time = std::chrono::high_resolution_clock::now();
    icp.align(pcl_aligned);
    auto pcl_end_time = std::chrono::high_resolution_clock::now();
    auto pcl_duration = std::chrono::duration_cast<std::chrono::milliseconds>(pcl_end_time - pcl_start_time);
    
    std::cout << "PCL ICP completed in " << pcl_duration.count() << " ms" << std::endl;
    std::cout << "PCL ICP has converged: " << (icp.hasConverged() ? "true" : "false") << std::endl;
    std::cout << "PCL ICP fitness score: " << icp.getFitnessScore() << std::endl;
    std::cout << "PCL ICP transformation matrix:" << std::endl;
    std::cout << icp.getFinalTransformation() << std::endl;
    
    // 병렬 ICP 실행
    std::cout << "\n===== Parallel ICP (OpenMP) =====" << std::endl;
    auto omp_start_time = std::chrono::high_resolution_clock::now();
    Eigen::Matrix4f transformation = performParallelICP(source_cloud, target_cloud);
    auto omp_end_time = std::chrono::high_resolution_clock::now();
    auto omp_duration = std::chrono::duration_cast<std::chrono::milliseconds>(omp_end_time - omp_start_time);
    
    std::cout << "Parallel ICP completed in " << omp_duration.count() << " ms" << std::endl;
    std::cout << "Final transformation matrix:" << std::endl;
    std::cout << transformation << std::endl;
    
    // 결과 변환 적용
    PointCloudT::Ptr transformed_cloud(new PointCloudT);
    pcl::transformPointCloud(*source_cloud, *transformed_cloud, transformation);
    
    // 결과 저장
    pcl::io::savePCDFile("transformed_source.pcd", *transformed_cloud);
    std::cout << "Transformed point cloud saved to 'transformed_source.pcd'" << std::endl;
    
    // 성능 비교 출력
    std::cout << "\n===== Performance Comparison =====" << std::endl;
    std::cout << "PCL ICP (Serial): " << pcl_duration.count() << " ms" << std::endl;
    std::cout << "Parallel ICP (OpenMP with " << num_threads << " threads): " << omp_duration.count() << " ms" << std::endl;
    
    if (omp_duration.count() < pcl_duration.count()) {
        double speedup = static_cast<double>(pcl_duration.count()) / omp_duration.count();
        std::cout << "Speedup: " << speedup << "x faster with OpenMP" << std::endl;
    } else {
        double slowdown = static_cast<double>(omp_duration.count()) / pcl_duration.count();
        std::cout << "Slowdown: " << slowdown << "x slower with OpenMP" << std::endl;
    }
    
    return 0;
}
