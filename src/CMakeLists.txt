# =============================================================================
# QiVision Library
# =============================================================================

set(QIVISION_SOURCES
    # Core
    Core/Types.cpp
    Core/QImage.cpp
    Core/QRegion.cpp
    Core/QMatrix.cpp
    Core/QContour.cpp
    Core/QContourArray.cpp
    # Platform
    Platform/Memory.cpp
    Platform/SIMD.cpp
    Platform/Random.cpp
    Platform/Thread.cpp
    Platform/Timer.cpp
    Platform/FileIO.cpp
    # Internal
    Internal/Gaussian.cpp
    Internal/Interpolate.cpp
    Internal/Gradient.cpp
    Internal/Convolution.cpp
    Internal/Edge1D.cpp
    Internal/Hessian.cpp
    Internal/Steger.cpp
    Internal/NonMaxSuppression.cpp
    Internal/EdgeLinking.cpp
    Internal/Canny.cpp
    Internal/Pyramid.cpp
    Internal/Profiler.cpp
    Internal/Histogram.cpp
    Internal/Threshold.cpp
    Internal/Matrix.cpp
    Internal/Solver.cpp
    Internal/Fitting.cpp
    Internal/SubPixel.cpp
    Internal/Geometry2d.cpp
    Internal/Distance.cpp
    Internal/Intersection.cpp
    Internal/GeomConstruct.cpp
    Internal/GeomRelation.cpp
    Internal/ContourProcess.cpp
    Internal/ContourAnalysis.cpp
    Internal/ContourSelect.cpp
    Internal/ContourConvert.cpp
)

set(QIVISION_HEADERS
    # Main header
    ${CMAKE_SOURCE_DIR}/include/QiVision/QiVision.h
    # Core
    ${CMAKE_SOURCE_DIR}/include/QiVision/Core/Types.h
    ${CMAKE_SOURCE_DIR}/include/QiVision/Core/Constants.h
    ${CMAKE_SOURCE_DIR}/include/QiVision/Core/Exception.h
    ${CMAKE_SOURCE_DIR}/include/QiVision/Core/QImage.h
    ${CMAKE_SOURCE_DIR}/include/QiVision/Core/QRegion.h
    ${CMAKE_SOURCE_DIR}/include/QiVision/Core/QMatrix.h
    ${CMAKE_SOURCE_DIR}/include/QiVision/Core/QContour.h
    ${CMAKE_SOURCE_DIR}/include/QiVision/Core/QContourArray.h
    # Platform
    ${CMAKE_SOURCE_DIR}/include/QiVision/Platform/Memory.h
    ${CMAKE_SOURCE_DIR}/include/QiVision/Platform/SIMD.h
    ${CMAKE_SOURCE_DIR}/include/QiVision/Platform/Random.h
    ${CMAKE_SOURCE_DIR}/include/QiVision/Platform/Thread.h
    ${CMAKE_SOURCE_DIR}/include/QiVision/Platform/Timer.h
    ${CMAKE_SOURCE_DIR}/include/QiVision/Platform/FileIO.h
    # Internal
    ${CMAKE_SOURCE_DIR}/include/QiVision/Internal/Gaussian.h
    ${CMAKE_SOURCE_DIR}/include/QiVision/Internal/Interpolate.h
    ${CMAKE_SOURCE_DIR}/include/QiVision/Internal/Gradient.h
    ${CMAKE_SOURCE_DIR}/include/QiVision/Internal/Convolution.h
    ${CMAKE_SOURCE_DIR}/include/QiVision/Internal/Edge1D.h
    ${CMAKE_SOURCE_DIR}/include/QiVision/Internal/Hessian.h
    ${CMAKE_SOURCE_DIR}/include/QiVision/Internal/Steger.h
    ${CMAKE_SOURCE_DIR}/include/QiVision/Internal/NonMaxSuppression.h
    ${CMAKE_SOURCE_DIR}/include/QiVision/Internal/EdgeLinking.h
    ${CMAKE_SOURCE_DIR}/include/QiVision/Internal/Canny.h
    ${CMAKE_SOURCE_DIR}/include/QiVision/Internal/Pyramid.h
    ${CMAKE_SOURCE_DIR}/include/QiVision/Internal/Profiler.h
    ${CMAKE_SOURCE_DIR}/include/QiVision/Internal/Histogram.h
    ${CMAKE_SOURCE_DIR}/include/QiVision/Internal/Threshold.h
    ${CMAKE_SOURCE_DIR}/include/QiVision/Internal/Matrix.h
    ${CMAKE_SOURCE_DIR}/include/QiVision/Internal/Solver.h
    ${CMAKE_SOURCE_DIR}/include/QiVision/Internal/Fitting.h
    ${CMAKE_SOURCE_DIR}/include/QiVision/Internal/SubPixel.h
    ${CMAKE_SOURCE_DIR}/include/QiVision/Internal/Geometry2d.h
    ${CMAKE_SOURCE_DIR}/include/QiVision/Internal/Distance.h
    ${CMAKE_SOURCE_DIR}/include/QiVision/Internal/Intersection.h
    ${CMAKE_SOURCE_DIR}/include/QiVision/Internal/GeomConstruct.h
    ${CMAKE_SOURCE_DIR}/include/QiVision/Internal/GeomRelation.h
    ${CMAKE_SOURCE_DIR}/include/QiVision/Internal/ContourProcess.h
    ${CMAKE_SOURCE_DIR}/include/QiVision/Internal/ContourAnalysis.h
    ${CMAKE_SOURCE_DIR}/include/QiVision/Internal/ContourSelect.h
    ${CMAKE_SOURCE_DIR}/include/QiVision/Internal/ContourConvert.h
)

# Create library
add_library(QiVision ${QIVISION_SOURCES} ${QIVISION_HEADERS})

# Include directories
target_include_directories(QiVision
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_SOURCE_DIR}/third_party
)

# Link libraries
if(QIVISION_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(QiVision PUBLIC OpenMP::OpenMP_CXX)
    endif()
endif()

# Compiler definitions
target_compile_definitions(QiVision
    PRIVATE
        $<$<CONFIG:Debug>:QIVISION_DEBUG>
)

# Set library properties
set_target_properties(QiVision PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    DEBUG_POSTFIX "d"
)
