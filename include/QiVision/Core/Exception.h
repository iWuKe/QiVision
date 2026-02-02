#pragma once

#include <QiVision/Core/Export.h>

/**
 * @file Exception.h
 * @brief Exception classes for QiVision
 */

#include <stdexcept>
#include <string>

namespace Qi::Vision {

/**
 * @brief Base exception class for QiVision
 */
class QIVISION_API Exception : public std::runtime_error {
public:
    explicit Exception(const std::string& message)
        : std::runtime_error(message) {}

    explicit Exception(const char* message)
        : std::runtime_error(message) {}
};

/**
 * @brief Invalid argument exception
 */
class QIVISION_API InvalidArgumentException : public Exception {
public:
    explicit InvalidArgumentException(const std::string& message)
        : Exception("Invalid argument: " + message) {}
};

/**
 * @brief Out of range exception
 */
class QIVISION_API OutOfRangeException : public Exception {
public:
    explicit OutOfRangeException(const std::string& message)
        : Exception("Out of range: " + message) {}
};

/**
 * @brief Insufficient data for algorithm (e.g., not enough points for fitting)
 */
class QIVISION_API InsufficientDataException : public Exception {
public:
    explicit InsufficientDataException(const std::string& message)
        : Exception("Insufficient data: " + message) {}
};

/**
 * @brief Algorithm failed to converge
 */
class QIVISION_API ConvergenceException : public Exception {
public:
    explicit ConvergenceException(const std::string& message)
        : Exception("Convergence failed: " + message) {}
};

/**
 * @brief File I/O exception
 */
class QIVISION_API IOException : public Exception {
public:
    explicit IOException(const std::string& message)
        : Exception("I/O error: " + message) {}
};

/**
 * @brief Unsupported operation or format
 */
class QIVISION_API UnsupportedException : public Exception {
public:
    explicit UnsupportedException(const std::string& message)
        : Exception("Unsupported: " + message) {}
};

/**
 * @brief Version mismatch for serialization
 */
class QIVISION_API VersionMismatchException : public Exception {
public:
    explicit VersionMismatchException(const std::string& message)
        : Exception("Version mismatch: " + message) {}
};

} // namespace Qi::Vision
