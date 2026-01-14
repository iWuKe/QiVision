# =============================================================================
# QiVision Tests
# =============================================================================

# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# =============================================================================
# Unit Tests
# =============================================================================
set(UNIT_TEST_SOURCES
    unit/test_main.cpp
    unit/core/test_types.cpp
    unit/core/test_matrix.cpp
    unit/core/test_contour.cpp
    unit/core/test_contour_array.cpp
    unit/platform/test_simd.cpp
    unit/platform/test_random.cpp
    unit/platform/test_thread.cpp
    unit/platform/test_timer.cpp
    unit/platform/test_fileio.cpp
    unit/internal/test_gaussian.cpp
    unit/internal/test_interpolate.cpp
    unit/internal/test_gradient.cpp
    unit/internal/test_convolution.cpp
    unit/internal/test_edge1d.cpp
    unit/internal/test_hessian.cpp
    unit/internal/test_steger.cpp
    unit/internal/test_nms.cpp
    unit/internal/test_edgelinking.cpp
    unit/internal/test_canny.cpp
    unit/internal/test_pyramid.cpp
    unit/internal/test_profiler.cpp
    unit/internal/test_histogram.cpp
    unit/internal/test_threshold.cpp
    unit/internal/test_internal_matrix.cpp
    unit/internal/test_solver.cpp
    unit/internal/test_fitting.cpp
    unit/internal/test_subpixel.cpp
    unit/internal/test_geometry2d.cpp
    unit/internal/test_distance.cpp
    unit/internal/test_intersection.cpp
    unit/internal/test_geom_construct.cpp
    unit/internal/test_geom_relation.cpp
    unit/internal/test_contour_process.cpp
    unit/internal/test_contour_analysis.cpp
    unit/internal/test_contour_select.cpp
    unit/internal/test_contour_convert.cpp
    unit/internal/test_contour_segment.cpp
    unit/internal/test_rle_ops.cpp
    unit/internal/test_struct_element.cpp
    unit/internal/test_morph_binary.cpp
    unit/internal/test_morph_gray.cpp
    unit/internal/test_connected_component.cpp
    unit/internal/test_distance_transform.cpp
    unit/internal/test_region_features.cpp
    unit/internal/test_affine_transform.cpp
    unit/internal/test_homography.cpp
    unit/internal/test_eigen.cpp
    unit/internal/test_hough.cpp
    # Measure (Feature Layer)
    unit/measure/test_caliper.cpp
    unit/measure/test_caliper_array.cpp
)

add_executable(unit_test ${UNIT_TEST_SOURCES})

target_link_libraries(unit_test
    PRIVATE
        QiVision
        GTest::gtest
        GTest::gtest_main
)

target_include_directories(unit_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(unit_test)

# =============================================================================
# Accuracy Tests (for precision validation)
# =============================================================================
set(ACCURACY_TEST_SOURCES
    accuracy/FittingAccuracyTest.cpp
    accuracy/SubPixelAccuracyTest.cpp
    accuracy/CaliperAccuracyTest.cpp
)

add_executable(accuracy_test ${ACCURACY_TEST_SOURCES})

target_link_libraries(accuracy_test
    PRIVATE
        QiVision
        GTest::gtest
        GTest::gtest_main
)

target_include_directories(accuracy_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

# Register accuracy tests with CTest
gtest_discover_tests(accuracy_test)

# =============================================================================
# Test Data Directory
# =============================================================================
# Copy test data to build directory
# file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
