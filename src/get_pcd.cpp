#include "get_pcd.hpp"
#include <stdio.h>

int main() {
    PCDReader reader("/home/dataset/sequences/00");
    

    // 원하는 프레임 번호를 지정하여 읽기
    // data type
    // pcl::PointXYZI

    for (int i = 0; i < 10; i++) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud = reader.getFrame(i);
        printf("Frame %d cloud size: %d\n", i, cloud->size());
        printf("Frame %d cloud point: %f\n", i, cloud->points[0].x);
        printf("Frame %d cloud point: %f\n", i, cloud->points[0].y);
        printf("Frame %d cloud point: %f\n", i, cloud->points[0].z);
        printf("Frame %d cloud point: %f\n", i, cloud->points[0].intensity);
    }

    return 0;
}