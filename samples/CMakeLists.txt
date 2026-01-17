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

# Core - Basic image operations
add_sample(core_basic_image core/basic_image.cpp)

# Matching - Shape matching
add_sample(matching_shape_match matching/shape_match.cpp)

# Build all samples
add_custom_target(samples DEPENDS
    core_basic_image
    matching_shape_match
)
