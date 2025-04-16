#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/common/centroid.h>
#include <chrono>

class SerialICP {
    public:
        SerialICP() {
            // ICP 기본 파라미터 설정
            icp_.setMaxCorrespondenceDistance(1.0);
            icp_.setMaximumIterations(1000);
            icp_.setTransformationEpsilon(1e-3);
        }

        // ICP 매칭 수행 함수
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
                init_guess = init_transform;
            }


            // ICP 입력 설정
            icp_.setInputSource(source);
            icp_.setInputTarget(target);
            
            // ICP 정렬 수행
            pcl::PointCloud<pcl::PointXYZI> final;
            icp_.align(final, init_guess);

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
