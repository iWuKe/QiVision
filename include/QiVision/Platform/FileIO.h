#pragma once

#include <QiVision/Core/Export.h>

/**
 * @file FileIO.h
 * @brief Cross-platform file I/O utilities
 *
 * Provides:
 * - Binary file read/write
 * - Text file read/write with UTF-8 support
 * - Path utilities
 * - File/directory existence checks
 *
 * Note: Image I/O is handled by QImage using stb_image.
 * This module is for general file operations and model serialization.
 */

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <fstream>

namespace Qi::Vision::Platform {

// ============================================================================
// Path Utilities
// ============================================================================

/**
 * @brief Check if file exists
 */
QIVISION_API bool FileExists(const std::string& path);

/**
 * @brief Check if directory exists
 */
QIVISION_API bool DirectoryExists(const std::string& path);

/**
 * @brief Get file size in bytes
 * @return File size, or -1 if file doesn't exist
 */
QIVISION_API int64_t GetFileSize(const std::string& path);

/**
 * @brief Get file extension (including dot)
 * @return Extension like ".txt", or empty string if none
 */
QIVISION_API std::string GetExtension(const std::string& path);

/**
 * @brief Get filename from path (without directory)
 */
QIVISION_API std::string GetFileName(const std::string& path);

/**
 * @brief Get directory from path (without filename)
 */
QIVISION_API std::string GetDirectory(const std::string& path);

/**
 * @brief Join path components
 */
QIVISION_API std::string JoinPath(const std::string& dir, const std::string& name);

/**
 * @brief Normalize path separators to platform-native
 */
QIVISION_API std::string NormalizePath(const std::string& path);

/**
 * @brief Create directory (and parents if needed)
 * @return true if created or already exists
 */
QIVISION_API bool CreateDirectory(const std::string& path);

/**
 * @brief Delete file
 * @return true if deleted or didn't exist
 */
QIVISION_API bool DeleteFile(const std::string& path);

// ============================================================================
// Binary File I/O
// ============================================================================

/**
 * @brief Read entire file into byte vector
 * @param path File path
 * @param data Output vector (will be resized)
 * @return true on success
 */
QIVISION_API bool ReadBinaryFile(const std::string& path, std::vector<uint8_t>& data);

/**
 * @brief Write byte vector to file
 * @param path File path
 * @param data Data to write
 * @return true on success
 */
QIVISION_API bool WriteBinaryFile(const std::string& path, const std::vector<uint8_t>& data);

/**
 * @brief Write raw bytes to file
 * @param path File path
 * @param data Pointer to data
 * @param size Size in bytes
 * @return true on success
 */
QIVISION_API bool WriteBinaryFile(const std::string& path, const void* data, size_t size);

// ============================================================================
// Text File I/O (UTF-8)
// ============================================================================

/**
 * @brief Read entire text file into string (UTF-8)
 * @param path File path
 * @param content Output string
 * @return true on success
 */
QIVISION_API bool ReadTextFile(const std::string& path, std::string& content);

/**
 * @brief Write string to text file (UTF-8)
 * @param path File path
 * @param content Content to write
 * @return true on success
 */
QIVISION_API bool WriteTextFile(const std::string& path, const std::string& content);

/**
 * @brief Read text file lines into vector
 * @param path File path
 * @param lines Output vector of lines
 * @param trimLines If true, trim whitespace from each line
 * @return true on success
 */
QIVISION_API bool ReadTextLines(const std::string& path, std::vector<std::string>& lines,
                   bool trimLines = true);

/**
 * @brief Write lines to text file
 * @param path File path
 * @param lines Lines to write
 * @param lineEnding Line ending to use ("\n" or "\r\n")
 * @return true on success
 */
QIVISION_API bool WriteTextLines(const std::string& path, const std::vector<std::string>& lines,
                    const std::string& lineEnding = "\n");

// ============================================================================
// Serialization Helpers
// ============================================================================

/**
 * @brief Binary writer helper
 *
 * Provides convenient methods for writing primitive types and vectors.
 */
class QIVISION_API BinaryWriter {
public:
    /**
     * @brief Construct writer for file
     * @param path Output file path
     */
    explicit BinaryWriter(const std::string& path);

    /**
     * @brief Check if file is open
     */
    bool IsOpen() const { return stream_.is_open(); }

    /**
     * @brief Close file
     */
    void Close() { stream_.close(); }

    /**
     * @brief Write primitive type
     */
    template<typename T>
    void Write(T value);

    /**
     * @brief Write array of primitives
     */
    template<typename T>
    void WriteArray(const T* data, size_t count);

    /**
     * @brief Write vector
     */
    template<typename T>
    void WriteVector(const std::vector<T>& vec);

    /**
     * @brief Write string (length-prefixed)
     */
    void WriteString(const std::string& str);

    /**
     * @brief Write raw bytes
     */
    void WriteBytes(const void* data, size_t size);

private:
    std::ofstream stream_;
};

/**
 * @brief Binary reader helper
 *
 * Provides convenient methods for reading primitive types and vectors.
 */
class QIVISION_API BinaryReader {
public:
    /**
     * @brief Construct reader for file
     * @param path Input file path
     */
    explicit BinaryReader(const std::string& path);

    /**
     * @brief Check if file is open
     */
    bool IsOpen() const { return stream_.is_open(); }

    /**
     * @brief Check if at end of file
     */
    bool IsEof() const { return stream_.eof(); }

    /**
     * @brief Close file
     */
    void Close() { stream_.close(); }

    /**
     * @brief Read primitive type
     */
    template<typename T>
    T Read();

    /**
     * @brief Read array of primitives
     */
    template<typename T>
    void ReadArray(T* data, size_t count);

    /**
     * @brief Read vector (reads count then elements)
     */
    template<typename T>
    std::vector<T> ReadVector();

    /**
     * @brief Read string (length-prefixed)
     */
    std::string ReadString();

    /**
     * @brief Read raw bytes
     */
    void ReadBytes(void* data, size_t size);

private:
    std::ifstream stream_;
};

// ============================================================================
// Template Implementations
// ============================================================================

template<typename T>
QIVISION_API void BinaryWriter::Write(T value) {
    stream_.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template<typename T>
QIVISION_API void BinaryWriter::WriteArray(const T* data, size_t count) {
    if (count > 0) {
        stream_.write(reinterpret_cast<const char*>(data), count * sizeof(T));
    }
}

template<typename T>
QIVISION_API void BinaryWriter::WriteVector(const std::vector<T>& vec) {
    uint64_t size = vec.size();
    Write(size);
    if (!vec.empty()) {
        WriteArray(vec.data(), vec.size());
    }
}

template<typename T>
T BinaryReader::Read() {
    T value{};
    stream_.read(reinterpret_cast<char*>(&value), sizeof(T));
    return value;
}

template<typename T>
QIVISION_API void BinaryReader::ReadArray(T* data, size_t count) {
    if (count > 0) {
        stream_.read(reinterpret_cast<char*>(data), count * sizeof(T));
    }
}

template<typename T>
QIVISION_API std::vector<T> BinaryReader::ReadVector() {
    uint64_t size = Read<uint64_t>();
    std::vector<T> vec(static_cast<size_t>(size));
    if (size > 0) {
        ReadArray(vec.data(), static_cast<size_t>(size));
    }
    return vec;
}

} // namespace Qi::Vision::Platform
