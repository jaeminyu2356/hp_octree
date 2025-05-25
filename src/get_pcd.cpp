#include "get_pcd.hpp"
#include "serial_icp.hpp"
#include "parallel_icp.hpp"
#include "cu_icp.cuh"
#include <stdio.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <thread>
#include <chrono>
#include <numeric>
#include <cmath>

struct TransformationError {
    float rotation_error;
    float translation_error;
};

inline TransformationError compareTransformations(const Eigen::Matrix4f& T1, const Eigen::Matrix4f& T2) {
    // 회전 오차 계산 (Frobenius norm을 사용하여 회전 행렬 차이를 계산)
    Eigen::Matrix3f R1 = T1.block<3,3>(0,0);
    Eigen::Matrix3f R2 = T2.block<3,3>(0,0);
    Eigen::Matrix3f R_diff = R1 * R2.transpose() - Eigen::Matrix3f::Identity();
    float rotation_diff = R_diff.norm() * 180.0 / M_PI;  // 각도로 변환 (도)

    // 이동 오차 계산 (Euclidean distance)
    Eigen::Vector3f t1 = T1.block<3,1>(0,3);
    Eigen::Vector3f t2 = T2.block<3,1>(0,3);
    float translation_diff = (t1 - t2).norm(); 
    
    return {rotation_diff, translation_diff};
}

int main() {
    PCDReader reader("/home/dataset/sequences/00");
    
    // // Point cloud 읽기 예시
    // for (int i = 0; i < 10; i++) {
    //     pcl::PointCloud<pcl::PointXYZI>::Ptr cloud = reader.getFrame(i);
    //     printf("Frame %d cloud size: %d\n", i, cloud->size());
    //     printf("Frame %d cloud point: %f\n", i, cloud->points[0].x);
    //     printf("Frame %d cloud point: %f\n", i, cloud->points[0].y);
    //     printf("Frame %d cloud point: %f\n", i, cloud->points[0].z);
    //     printf("Frame %d cloud point: %f\n", i, cloud->points[0].intensity);
    // }

    // Pose transformation 예시
    // frame 0과 frame 5 사이의 transformation matrix 가져오기
    //Eigen::Matrix4f transform = reader.getTransformation(0, 5);

    
    std::vector<TransformationError> error_history;
    float total_rotation_error = 0.0f;
    float total_translation_error = 0.0f;

    //이전 프레임 변환 행렬 저장 공유 메모리
    Eigen::Matrix4f serial_transform = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f parallel_transform = Eigen::Matrix4f::Identity();

    for (int i = 0; i < 4540; i++) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud1 = reader.getFrame(i);
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud2 = reader.getFrame(i+1);
        Eigen::Matrix4f gt_transform = reader.getTransformation(i, i+1);

        /////////////////////////////////////////// ICP 매칭 ///////////////////////////////////////////
        SerialICP serial_icp;
        int64_t computation_time;

        // ParallelICP parallel_icp;
        // int64_t parallel_computation_time;
        
        // Serial ICP 매칭 수행
        serial_transform = serial_icp.align(cloud1, cloud2, serial_transform, computation_time);
        std::cout << "[Frame " << i << "] Serial ICP 매칭 소요 시간: " << computation_time << "ms" << std::endl;
        
        // Parallel ICP 매칭 수행
        // Eigen::Matrix4f parallel_transform = parallel_icp.align(cloud1, cloud2, parallel_transform, parallel_computation_time);
        // std::cout << "[Frame " << i << "] Parallel ICP 매칭 소요 시간: " << parallel_computation_time << "ms" << std::endl;

        // 최종 변환 결과와 ground truth 비교
        TransformationError errors = compareTransformations(gt_transform, serial_transform);
        error_history.push_back(errors);
        
        // 현재까지의 평균 에러 계산
        total_rotation_error += errors.rotation_error;
        total_translation_error += errors.translation_error;
        float avg_rotation_error = total_rotation_error / (i + 1);
        float avg_translation_error = total_translation_error / (i + 1);

        std::cout << "[Frame " << i << "] Current Errors - Rotation: " << errors.rotation_error 
                  << ", Translation: " << errors.translation_error << std::endl;
        std::cout << "[Frame " << i << "] Average Errors - Rotation: " << avg_rotation_error 
                  << ", Translation: " << avg_translation_error << std::endl;
        std::cout << "-------------------------------------------" << std::endl;

        //================================ 시간 비교 =================================
        // // 결과 비교
        // bool match = compareTransformations(serial_transform, parallel_transform);
        // results_match.push_back(match);
        // std::cout << "[Frame " << i << "] Results match: " << (match ? "Yes" : "No") << std::endl;
        
        // // 변환 행렬 출력
        // std::cout << "\nSerial Transform:\n" << serial_transform << std::endl;
        std::cout << "\nParallel Transform:\n" << parallel_transform << std::endl;

        // 가속화 계산
        // float speedup = static_cast<float>(computation_time) / parallel_computation_time;
        // std::cout << "[Frame " << i << "] Speedup: " << speedup << "x" << std::endl;
        // mean_time.push_back(speedup);
        
        // std::cout << "-------------------------------------------" << std::endl;
    }
    
    // 최종 통계 출력
    // float mean_speedup = std::accumulate(mean_time.begin(), mean_time.end(), 0.0f) / mean_time.size();
    // int match_count = std::count(results_match.begin(), results_match.end(), true);
    
    // std::cout << "\n=== Final Statistics ===" << std::endl;
    // std::cout << "Average speedup: " << mean_speedup << "x" << std::endl;
    // std::cout << "Matching results: " << match_count << "/" << results_match.size() 
    //           << " (" << (float)match_count/results_match.size()*100 << "%)" << std::endl;

    /////////////////////////////////////////// 시각화 ///////////////////////////////////////////

    // Ground Truth 변환 행렬
    // Eigen::Matrix4f gt_transform = reader.getTransformation(0, 7);

    // // 변환된 포인트 클라우드 생성
    // pcl::PointCloud<pcl::PointXYZI>::Ptr cloud1_icp(new pcl::PointCloud<pcl::PointXYZI>);
    // pcl::PointCloud<pcl::PointXYZI>::Ptr cloud1_gt(new pcl::PointCloud<pcl::PointXYZI>);
    
    // // ICP와 GT 변환 적용
    // pcl::transformPointCloud(*cloud1, *cloud1_icp, icp_transform);
    // pcl::transformPointCloud(*cloud1, *cloud1_gt, gt_transform);

    // pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("ICP vs Ground Truth"));
    // viewer->setBackgroundColor(0, 0, 0);

    // // 원본 cloud2를 흰색으로 표시
    // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> white(cloud2, 255, 255, 255);
    // viewer->addPointCloud<pcl::PointXYZI>(cloud2, white, "cloud2");
    // // 원본 cloud1을 파란색으로 표시
    // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> blue(cloud1, 0, 0, 255);
    // viewer->addPointCloud<pcl::PointXYZI>(cloud1, blue, "cloud1");

    // // ICP로 변환된 cloud1을 빨간색으로 표시
    // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> red(cloud1_icp, 255, 0, 0);
    // viewer->addPointCloud<pcl::PointXYZI>(cloud1_icp, red, "cloud1_icp");

    // // GT로 변환된 cloud1을 초록색으로 표시
    // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> green(cloud1_gt, 0, 255, 0);
    // viewer->addPointCloud<pcl::PointXYZI>(cloud1_gt, green, "cloud1_gt");

    // // 포인트 크기 설정
    // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud2");
    // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud1_icp");
    // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud1_gt");

    // // 카메라 위치 설정
    // viewer->initCameraParameters();
    // viewer->setCameraPosition(0, 0, 50, 0, 0, 0, 0, 1, 0);

    // // 변환 행렬 출력
    // std::cout << "\nICP Transformation matrix:" << std::endl;
    // std::cout << icp_transform << std::endl;
    // std::cout << "\nGround truth transformation matrix:" << std::endl;
    // std::cout << gt_transform << std::endl;

    // // 뷰어 실행
    // while (!viewer->wasStopped()) {
    //     viewer->spinOnce(100);
    //     std::this_thread::sleep_for(std::chrono::milliseconds(100000));
    // }

    /////////////////////////////////////////// 시각화 끝 ///////////////////////////////////////////

    return 0;
}