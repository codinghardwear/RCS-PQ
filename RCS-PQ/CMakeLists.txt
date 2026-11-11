cmake_minimum_required(VERSION 3.16)
project(RCS_PQ_Standalone LANGUAGES CXX)

option(RCSPQ_NATIVE "Compile for the build machine's native CPU (recommended for benchmarking)" ON)
option(RCSPQ_BUILD_TESTS "Build the small standalone smoke test" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)

add_library(rcspq_core src/rcspq.cpp)
target_include_directories(rcspq_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(rcspq_core PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(rcspq_core PRIVATE -O3 -Wall -Wextra -Wpedantic)
if(RCSPQ_NATIVE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(rcspq_core PRIVATE -march=native)
endif()

add_executable(rcspq src/main.cpp)
target_link_libraries(rcspq PRIVATE rcspq_core)
target_compile_options(rcspq PRIVATE -O3 -Wall -Wextra -Wpedantic)

if(RCSPQ_BUILD_TESTS)
    enable_testing()
    add_executable(rcspq_smoke_test tests/smoke_test.cpp)
    target_link_libraries(rcspq_smoke_test PRIVATE rcspq_core)
    add_test(NAME rcspq_smoke_test COMMAND rcspq_smoke_test)
endif()
