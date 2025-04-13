#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <fstream>
#include <string>
#include <vector>
#include <Eigen/Dense>

class PCDReader {
    public:
        PCDReader(const std::string& base_path) : base_path_(base_path) {}

        pcl::PointCloud<pcl::PointXYZI>::Ptr getFrame(int frame_number) {
            std::string bin_file = generateFileName(frame_number);
            return readBinFile(bin_file);
        }

        // 두 프레임 간의 transformation 행렬을 반환하는 함수
        Eigen::Matrix4f getTransformation(int frame1, int frame2) {
            std::vector<Eigen::Matrix4f> poses = readPoseFile();
            if (frame1 >= poses.size() || frame2 >= poses.size()) {
                std::cerr << "프레임 번호가 유효하지 않습니다." << std::endl;
                return Eigen::Matrix4f::Identity();
            }
            // frame2의 pose를 frame1의 pose에 대해 상대적으로 변환
            return poses[frame1].inverse() * poses[frame2];
        }

    private:
        std::string base_path_;

        std::string generateFileName(int frame_number) {
            std::string frame_num = std::to_string(frame_number);
            // 6자리 숫자로 맞추기 (000000, 000001, ...)
            frame_num = std::string(6 - frame_num.length(), '0') + frame_num;
            return base_path_ + "/velodyne/" + frame_num + ".bin";
        }

        pcl::PointCloud<pcl::PointXYZI>::Ptr readBinFile(const std::string& filename) {
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
            
            std::ifstream file(filename, std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "파일을 열 수 없습니다: " << filename << std::endl;
                return cloud;
            }

            // bin 파일의 각 포인트는 x,y,z,intensity 값을 가짐 (각각 float)
            std::vector<float> data(4);
            while (file.read(reinterpret_cast<char*>(data.data()), sizeof(float) * 4)) {
                pcl::PointXYZI point;
                point.x = data[0];
                point.y = data[1];
                point.z = data[2];
                point.intensity = data[3];
                cloud->push_back(point);
            }

            file.close();
            return cloud;
        }

        std::vector<Eigen::Matrix4f> readPoseFile() {
            std::vector<Eigen::Matrix4f> poses;
            std::string pose_file = base_path_ + "/poses.txt";
            std::ifstream file(pose_file);
            
            if (!file.is_open()) {
                std::cerr << "pose 파일을 열 수 없습니다: " << pose_file << std::endl;
                return poses;
            }

            std::string line;
            while (std::getline(file, line)) {
                std::stringstream ss(line);
                Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
                
                // poses.txt의 각 줄은 12개의 숫자로 구성 (3x4 변환 행렬)
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 4; j++) {
                        float value;
                        ss >> value;
                        pose(i, j) = value;
                    }
                }
                poses.push_back(pose);
            }

            file.close();
            return poses;
        }
};