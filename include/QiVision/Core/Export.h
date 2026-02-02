#pragma once

/**
 * @file Export.h
 * @brief Export/import macros for shared library support
 *
 * Build system should define one of:
 *   - QIVISION_BUILD_SHARED: when building QiVision as shared library
 *   - QIVISION_USE_SHARED: when using QiVision as shared library
 *   - QIVISION_STATIC: when building/using as static library (default)
 */

#if defined(_WIN32) || defined(_WIN64)
    #if defined(QIVISION_BUILD_SHARED)
        #define QIVISION_API __declspec(dllexport)
    #elif defined(QIVISION_USE_SHARED)
        #define QIVISION_API __declspec(dllimport)
    #else
        #define QIVISION_API
    #endif
    #define QIVISION_CALL __cdecl
#else
    #if defined(QIVISION_BUILD_SHARED)
        #define QIVISION_API __attribute__((visibility("default")))
    #else
        #define QIVISION_API
    #endif
    #define QIVISION_CALL
#endif
