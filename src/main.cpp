#include <omp_icp.hpp>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " source.pcd target.pcd [num_threads]" << std::endl;
        return -1;
    }
    
    int num_threads = (argc > 3) ? std::atoi(argv[3]) : -1;
    
    return runICPComparison(argv[1], argv[2], num_threads);
} 