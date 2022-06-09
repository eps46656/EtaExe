CMAKE_MINIMUM_REQUIRED(VERSION 3.14)
PROJECT(EtaExe)
# SET(CMAKE_VERBOSE_MAKEFILE true)

################################################################################

FIND_PACKAGE(CUDA REQUIRED)

INCLUDE_DIRECTORIES(/home/eps/desktop/Eta)

################################################################################

SET(CMAKE_BUILD_TYPE RELEASE)

SET(CMAKE_CXX_FLAGS "")
STRING(APPEND CMAKE_CXX_FLAGS "-std=c++14 ")
STRING(APPEND CMAKE_CXX_FLAGS "-Wall ")
MESSAGE(${CMAKE_CXX_FLAGS})

SET(CMAKE_CXX_FLAGS_DEBUG "")
STRING(APPEND CMAKE_CXX_FLAGS_DEBUG "-g ")

SET(CMAKE_CXX_FLAGS_RELEASE "")
STRING(APPEND CMAKE_CXX_FLAGS_RELEASE "-O3 ")

################################################################################

SET(CUDA_NVCC_FLAGS "")
STRING(APPEND CUDA_NVCC_FLAGS "-arch=sm_61 ")
STRING(APPEND CUDA_NVCC_FLAGS "-Xptxas=-v ")
STRING(APPEND CUDA_NVCC_FLAGS "-maxrregcount=64 ")
STRING(APPEND CUDA_NVCC_FLAGS "-Werror all-warnings ")
STRING(APPEND CUDA_NVCC_FLAGS "--optimize=3 ")
STRING(APPEND CUDA_NVCC_FLAGS "--use_fast_math ")
MESSAGE(${CUDA_NVCC_FLAGS})
# LIST(APPEND CUDA_NVCC_FLAGS -G)
# LIST(APPEND CUDA_NVCC_FLAGS -rdc=true)

################################################################################

LIST(APPEND SRC src/main19.cu)

################################################################################

CUDA_ADD_EXECUTABLE(exe ${SRC})
