#ifndef OMP_ICP_HPP
#define OMP_ICP_HPP

#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <vector>
#include <Eigen/Core>
#include <string>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

/**
 * @brief Find nearest points in target cloud for each point in source cloud (parallelized)
 * 
 * @param source Source point cloud
 * @param target Target point cloud
 * @param correspondences Output vector of correspondence indices
 */
void findNearestPointsParallel(const PointCloudT::Ptr& source, 
                              const PointCloudT::Ptr& target,
                              std::vector<int>& correspondences);

/**
 * @brief Compute error between source and target clouds using correspondences (parallelized)
 * 
 * @param source Source point cloud
 * @param target Target point cloud
 * @param correspondences Vector of correspondence indices
 * @return double Mean squared error
 */
double computeErrorParallel(const PointCloudT::Ptr& source, 
                           const PointCloudT::Ptr& target,
                           const std::vector<int>& correspondences);

/**
 * @brief Perform ICP registration using OpenMP parallelization
 * 
 * @param source Source point cloud
 * @param target Target point cloud
 * @param max_iterations Maximum number of iterations
 * @param convergence_threshold Convergence threshold for early stopping
 * @return Eigen::Matrix4f Transformation matrix from source to target
 */
Eigen::Matrix4f performParallelICP(const PointCloudT::Ptr& source, 
                                  const PointCloudT::Ptr& target,
                                  int max_iterations = 50,
                                  double convergence_threshold = 1e-6);

/**
 * @brief Run ICP comparison between PCL's implementation and our parallel implementation
 * 
 * @param source_file Path to source PCD file
 * @param target_file Path to target PCD file
 * @param num_threads Number of OpenMP threads to use (default: max available)
 * @return int Status code (0 for success)
 */
int runICPComparison(const std::string& source_file, 
                    const std::string& target_file, 
                    int num_threads = -1);

#endif // OMP_ICP_HPP