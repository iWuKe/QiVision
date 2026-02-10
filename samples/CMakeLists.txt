# =============================================================================
# QiVision Samples
# =============================================================================

# Helper function to add sample executable
function(add_sample name source_file)
    add_executable(${name} ${source_file})
    target_link_libraries(${name} PRIVATE QiVision)
    target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/include)
    set_target_properties(${name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/samples
    )
endfunction()

# Matching - Shape matching
add_sample(matching_shape_match matching/shape_match.cpp)
add_sample(matching_fast_shape_match matching/fast_shape_match.cpp)
add_sample(matching_fast_shape_match_gui matching/fast_shape_match_gui.cpp)
add_sample(matching_shape_match_clip_gui matching/shape_match_clip_gui.cpp)
add_sample(matching_test_create_model matching/test_create_model.cpp)
add_sample(matching_test_scaled_match matching/test_scaled_match.cpp)
add_sample(matching_ncc_match matching/ncc_match.cpp)
add_sample(matching_component_model matching/component_model_demo.cpp)

# Measure - Metrology & Caliper
add_sample(measure_circle_metrology measure/circle_metrology.cpp)
add_sample(measure_line_metrology measure/line_metrology.cpp)
add_sample(measure_caliper_pairs measure/caliper_pairs.cpp)
add_sample(measure_caliper_circle_manual measure/caliper_circle_manual.cpp)
add_sample(measure_ellipse_fit_synthetic measure/ellipse_fit_synthetic.cpp)

# Blob - Blob analysis
add_sample(blob_analysis blob/blob_analysis.cpp)

# Color - Color detection
add_sample(color_detect_demo color/color_detect_demo.cpp)

# Calib/Transform - Polar transform
add_sample(calib_polar_transform calib/polar_transform_test.cpp)

# Barcode - Barcode reading (requires ZXing-cpp)
if(QIVISION_BUILD_BARCODE)
    add_sample(barcode_read barcode/barcode_read.cpp)
endif()

# OCR - Text recognition (requires ONNXRuntime)
if(QIVISION_BUILD_OCR)
    add_sample(ocr_demo ocr/ocr_demo.cpp)
    add_sample(ocr_polar_circular_barcode ocr/polar_ocr_circular_barcode.cpp)
endif()

# Build all samples
set(SAMPLE_TARGETS
    matching_shape_match
    matching_fast_shape_match
    matching_fast_shape_match_gui
    matching_shape_match_clip_gui
    matching_test_create_model
    matching_test_scaled_match
    matching_ncc_match
    matching_component_model
    measure_circle_metrology
    measure_line_metrology
    measure_caliper_pairs
    measure_caliper_circle_manual
    measure_ellipse_fit_synthetic
    blob_analysis
    color_detect_demo
    calib_polar_transform
)
if(QIVISION_BUILD_BARCODE)
    list(APPEND SAMPLE_TARGETS barcode_read)
endif()
if(QIVISION_BUILD_OCR)
    list(APPEND SAMPLE_TARGETS ocr_demo ocr_polar_circular_barcode)
endif()
add_custom_target(samples DEPENDS ${SAMPLE_TARGETS})
