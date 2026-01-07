#pragma once

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
class Exception : public std::runtime_error {
public:
    explicit Exception(const std::string& message)
        : std::runtime_error(message) {}

    explicit Exception(const char* message)
        : std::runtime_error(message) {}
};

/**
 * @brief Invalid argument exception
 */
class InvalidArgumentException : public Exception {
public:
    explicit InvalidArgumentException(const std::string& message)
        : Exception("Invalid argument: " + message) {}
};

/**
 * @brief Out of range exception
 */
class OutOfRangeException : public Exception {
public:
    explicit OutOfRangeException(const std::string& message)
        : Exception("Out of range: " + message) {}
};

/**
 * @brief Insufficient data for algorithm (e.g., not enough points for fitting)
 */
class InsufficientDataException : public Exception {
public:
    explicit InsufficientDataException(const std::string& message)
        : Exception("Insufficient data: " + message) {}
};

/**
 * @brief Algorithm failed to converge
 */
class ConvergenceException : public Exception {
public:
    explicit ConvergenceException(const std::string& message)
        : Exception("Convergence failed: " + message) {}
};

/**
 * @brief File I/O exception
 */
class IOException : public Exception {
public:
    explicit IOException(const std::string& message)
        : Exception("I/O error: " + message) {}
};

/**
 * @brief Unsupported operation or format
 */
class UnsupportedException : public Exception {
public:
    explicit UnsupportedException(const std::string& message)
        : Exception("Unsupported: " + message) {}
};

/**
 * @brief Version mismatch for serialization
 */
class VersionMismatchException : public Exception {
public:
    explicit VersionMismatchException(const std::string& message)
        : Exception("Version mismatch: " + message) {}
};

} // namespace Qi::Vision
