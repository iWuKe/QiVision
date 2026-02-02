cmake_minimum_required(VERSION 3.16)

add_executable(unit_test
    test_scale_verify.cpp
)

add_executable(test_scale_simple
    test_scale_simple.cpp
)

target_link_libraries(unit_test PRIVATE QiVision)
target_link_libraries(test_scale_simple PRIVATE QiVision)

target_include_directories(unit_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_scale_simple PRIVATE ${CMAKE_SOURCE_DIR}/include)
