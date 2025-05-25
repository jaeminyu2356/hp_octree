cmake_minimum_required(VERSION 3.10)
project(hp_octree CUDA CXX)

# Set C++14 standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(PCL REQUIRED COMPONENTS common io visualization)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Add CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
    -arch=sm_60 \
    --expt-relaxed-constexpr \
    --extended-lambda \
    --std=c++14 \
    -Xcompiler -fPIC \
    --default-stream per-thread \
    --relocatable-device-code=true \
    -Xcudafe --display_error_number")

# Set source files
set(CUDA_SOURCES
    src/cu_icp_kernels.cu
)

# Create CUDA library
add_library(cu_icp_kernels STATIC ${CUDA_SOURCES})

# Set CUDA library properties
set_target_properties(cu_icp_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_ARCHITECTURES 60
)

# Add include directories for CUDA library
target_include_directories(cu_icp_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Link CUDA libraries
target_link_libraries(cu_icp_kernels PUBLIC
    CUDA::cudart
    CUDA::cusolver
)

# Create main library
add_library(cu_icp STATIC include/cu_icp.cuh)

# Set language for cu_icp
set_source_files_properties(include/cu_icp.cuh PROPERTIES LANGUAGE CUDA)

# Set properties for cu_icp
set_target_properties(cu_icp PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_ARCHITECTURES 60
)

# Link libraries for main library
target_link_libraries(cu_icp PUBLIC
    cu_icp_kernels
    ${PCL_LIBRARIES}
)

# Add include directories for main library
target_include_directories(cu_icp PUBLIC
    ${PCL_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Create main executable
add_executable(get_pcd src/get_pcd.cpp)

# Set properties for get_pcd
set_target_properties(get_pcd PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_ARCHITECTURES 60
)

# Link libraries for main executable
target_link_libraries(get_pcd PRIVATE
    cu_icp
    ${PCL_LIBRARIES}
    OpenMP::OpenMP_CXX
    CUDA::cudart
)

# Add PCL definitions
add_definitions(${PCL_DEFINITIONS})
