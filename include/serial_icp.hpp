#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <chrono>

class SerialICP {
    public:
        SerialICP() {
            // ICP 기본 파라미터 설정
            icp_.setMaxCorrespondenceDistance(2.0);
            icp_.setMaximumIterations(3000);
            icp_.setTransformationEpsilon(1e-3);
        }

        // ICP 매칭 수행 함수
        Eigen::Matrix4f align(const pcl::PointCloud<pcl::PointXYZI>::Ptr& source,
                            const pcl::PointCloud<pcl::PointXYZI>::Ptr& target,
                            int64_t& computation_time) {
            
            auto start = std::chrono::high_resolution_clock::now();

            // ICP 입력 설정
            icp_.setInputSource(source);
            icp_.setInputTarget(target);
            
            // ICP 정렬 수행
            pcl::PointCloud<pcl::PointXYZI> final;
            icp_.align(final);

            auto end = std::chrono::high_resolution_clock::now();
            computation_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

            // 변환 행렬 반환
            return icp_.getFinalTransformation();
        }

        // ICP 파라미터 설정 함수들
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
};