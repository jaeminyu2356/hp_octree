#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <fstream>
#include <string>
#include <vector>

class PCDReader {
    public:
        PCDReader(const std::string& base_path) : base_path_(base_path) {}

        pcl::PointCloud<pcl::PointXYZI>::Ptr getFrame(int frame_number) {
            std::string bin_file = generateFileName(frame_number);
            return readBinFile(bin_file);
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
};