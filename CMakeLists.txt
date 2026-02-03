cmake_minimum_required(VERSION 3.16)

project(QiVision
    VERSION 0.1.0
    DESCRIPTION "Industrial Machine Vision Library"
    LANGUAGES CXX
)

# =============================================================================
# C++ Standard
# =============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =============================================================================
# Build Options
# =============================================================================
option(QIVISION_BUILD_SAMPLES "Build sample programs" ON)
option(QIVISION_BUILD_TESTS "Build unit tests" OFF)
option(QIVISION_ENABLE_SIMD "Enable SIMD optimizations" ON)
option(QIVISION_ENABLE_OPENMP "Enable OpenMP for parallelization" ON)

# Optional modules (default ON for Windows/Linux, OFF otherwise)
set(QIVISION_BUILD_GUI_DEFAULT OFF)
set(QIVISION_BUILD_DISPLAY_DEFAULT OFF)
if(WIN32 OR UNIX)
    set(QIVISION_BUILD_GUI_DEFAULT ON)
    set(QIVISION_BUILD_DISPLAY_DEFAULT ON)
endif()
option(QIVISION_BUILD_GUI "Build GUI module" ${QIVISION_BUILD_GUI_DEFAULT})
option(QIVISION_BUILD_DISPLAY "Build Display module" ${QIVISION_BUILD_DISPLAY_DEFAULT})
option(QIVISION_BUILD_BARCODE "Build Barcode module (requires ZXing-cpp)" ON)
option(QIVISION_BUILD_OCR "Build OCR module (requires ONNXRuntime)" OFF)

# =============================================================================
# Generated Configuration Header
# =============================================================================
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/include/QiVision)
configure_file(
    ${CMAKE_SOURCE_DIR}/include/QiVision/QiVisionConfig.h.in
    ${CMAKE_BINARY_DIR}/include/QiVision/QiVisionConfig.h
    @ONLY
)

# =============================================================================
# Compiler Settings
# =============================================================================
if(MSVC)
    add_compile_options(/W4 /utf-8)
    add_compile_definitions(_USE_MATH_DEFINES)  # Enable M_PI etc.
    add_compile_definitions(NOMINMAX)           # Prevent Windows min/max macros
    if(QIVISION_ENABLE_SIMD)
        add_compile_options(/arch:AVX2)
    endif()
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(QIVISION_ENABLE_SIMD)
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag(-mavx2 COMPILER_SUPPORTS_AVX2)
        check_cxx_compiler_flag(-msse4.1 COMPILER_SUPPORTS_SSE4)
        if(COMPILER_SUPPORTS_AVX2)
            add_compile_options(-mavx2 -mfma)
        elseif(COMPILER_SUPPORTS_SSE4)
            add_compile_options(-msse4.1)
        endif()
    endif()
endif()

# =============================================================================
# OpenMP Support
# =============================================================================
if(QIVISION_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
    else()
        message(WARNING "OpenMP not found, disabling parallel optimization")
        set(QIVISION_ENABLE_OPENMP OFF)
    endif()
endif()

if(ANDROID)
    if(QIVISION_ENABLE_OPENMP)
        message(STATUS "Android: disabling OpenMP by default")
        set(QIVISION_ENABLE_OPENMP OFF CACHE BOOL "Enable OpenMP for parallelization" FORCE)
    endif()
endif()

# =============================================================================
# ZXing-cpp for Barcode Support
# =============================================================================
if(QIVISION_BUILD_BARCODE)
    include(FetchContent)
    FetchContent_Declare(
        zxing-cpp
        GIT_REPOSITORY https://github.com/zxing-cpp/zxing-cpp.git
        GIT_TAG        v2.2.1
    )
    set(BUILD_WRITERS OFF CACHE BOOL "" FORCE)
    set(BUILD_READERS ON CACHE BOOL "" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_BLACKBOX_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_UNIT_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_PYTHON_MODULE OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(zxing-cpp)
    message(STATUS "ZXing-cpp: enabled for barcode support")
endif()

# =============================================================================
# ONNXRuntime for OCR Support
# =============================================================================
if(QIVISION_BUILD_OCR)
    # Try to find ONNXRuntime
    # Users can set ONNXRUNTIME_ROOT to specify the installation path
    find_path(ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        PATHS
            ${ONNXRUNTIME_ROOT}/include
            /usr/include/onnxruntime
            /usr/local/include/onnxruntime
            $ENV{ONNXRUNTIME_ROOT}/include
    )

    find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        PATHS
            ${ONNXRUNTIME_ROOT}/lib
            /usr/lib
            /usr/local/lib
            /usr/lib/x86_64-linux-gnu
            $ENV{ONNXRUNTIME_ROOT}/lib
    )

    if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
        message(STATUS "ONNXRuntime found:")
        message(STATUS "  Include: ${ONNXRUNTIME_INCLUDE_DIR}")
        message(STATUS "  Library: ${ONNXRUNTIME_LIBRARY}")
        set(QIVISION_HAS_ONNXRUNTIME TRUE)
    else()
        message(WARNING "ONNXRuntime not found. OCR module will be disabled.")
        message(STATUS "  To enable OCR, install ONNXRuntime or set ONNXRUNTIME_ROOT")
        message(STATUS "  Ubuntu: sudo apt install libonnxruntime-dev")
        message(STATUS "  Or download from: https://github.com/microsoft/onnxruntime/releases")
        set(QIVISION_BUILD_OCR OFF)
        set(QIVISION_HAS_ONNXRUNTIME FALSE)
    endif()
endif()

# =============================================================================
# Output Directories
# =============================================================================
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# =============================================================================
# Main Library
# =============================================================================
add_subdirectory(src)

# =============================================================================
# Samples
# =============================================================================
if(QIVISION_BUILD_SAMPLES)
    add_subdirectory(samples)
endif()

# =============================================================================
# Tests
# =============================================================================
if(QIVISION_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# =============================================================================
# Installation
# =============================================================================
include(GNUInstallDirs)

install(DIRECTORY include/QiVision
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    PATTERN "QiVisionConfig.h.in" EXCLUDE
)

install(FILES ${CMAKE_BINARY_DIR}/include/QiVision/QiVisionConfig.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/QiVision
)

install(TARGETS QiVision
    EXPORT QiVisionTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# =============================================================================
# Summary
# =============================================================================
message(STATUS "")
message(STATUS "QiVision ${PROJECT_VERSION}")
message(STATUS "  Build type:    ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:  ${CMAKE_CXX_STANDARD}")
message(STATUS "  SIMD:          ${QIVISION_ENABLE_SIMD}")
message(STATUS "  Barcode:       ${QIVISION_BUILD_BARCODE}")
message(STATUS "  OCR:           ${QIVISION_BUILD_OCR}")
message(STATUS "  Samples:       ${QIVISION_BUILD_SAMPLES}")
message(STATUS "  Tests:         ${QIVISION_BUILD_TESTS}")
message(STATUS "")
