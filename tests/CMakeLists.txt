cmake_minimum_required(VERSION 3.16)

add_executable(unit_test
    test_scale_verify.cpp
)

add_executable(test_scale_simple
    test_scale_simple.cpp
)

add_executable(test_ncc_serialization
    test_ncc_serialization.cpp
)

add_executable(test_ncc_match
    test_ncc_match.cpp
)

add_executable(test_intensity_stats
    test_intensity_stats.cpp
)

add_executable(test_component_model
    test_component_model.cpp
)

add_executable(test_caliper
    test_caliper.cpp
)

add_executable(test_caliper_regression
    test_caliper_regression.cpp
)

add_executable(test_caliper_param_scan
    test_caliper_param_scan.cpp
)

target_link_libraries(unit_test PRIVATE QiVision)
target_link_libraries(test_scale_simple PRIVATE QiVision)
target_link_libraries(test_ncc_serialization PRIVATE QiVision)
target_link_libraries(test_ncc_match PRIVATE QiVision)
target_link_libraries(test_intensity_stats PRIVATE QiVision)
target_link_libraries(test_component_model PRIVATE QiVision)
target_link_libraries(test_caliper PRIVATE QiVision)
target_link_libraries(test_caliper_regression PRIVATE QiVision)
target_link_libraries(test_caliper_param_scan PRIVATE QiVision)

target_include_directories(unit_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_scale_simple PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_ncc_serialization PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_ncc_match PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_intensity_stats PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_component_model PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_caliper PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_caliper_regression PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(test_caliper_param_scan PRIVATE ${CMAKE_SOURCE_DIR}/include)
