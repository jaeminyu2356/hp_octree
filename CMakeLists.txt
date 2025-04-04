cmake_minimum_required(VERSION 3.0.2)
project(hp_octree)

# C++14 사용
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# PCL 라이브러리 찾기
find_package(PCL REQUIRED)

# 헤더 파일 경로 추가
include_directories(
    ${PCL_INCLUDE_DIRS}
    include
)

# 링크 디렉토리 추가
link_directories(${PCL_LIBRARY_DIRS})

# 매크로 추가
add_definitions(${PCL_DEFINITIONS})

# 실행 파일 생성
add_executable(get_pcd 
    src/get_pcd.cpp
)

# PCL 라이브러리 링크
target_link_libraries(get_pcd
    ${PCL_LIBRARIES}
)
