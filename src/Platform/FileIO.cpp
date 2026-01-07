/**
 * @file FileIO.cpp
 * @brief File I/O implementation
 */

#include <QiVision/Platform/FileIO.h>

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>

// Platform-specific includes
#ifdef _WIN32
#include <direct.h>
#include <io.h>
#define ACCESS _access
#define MKDIR(path) _mkdir(path)
#else
#include <sys/stat.h>
#include <unistd.h>
#define ACCESS access
#define MKDIR(path) mkdir(path, 0755)
#endif

namespace Qi::Vision::Platform {

// ============================================================================
// Path Utilities
// ============================================================================

bool FileExists(const std::string& path) {
    if (path.empty()) return false;
    return ACCESS(path.c_str(), 0) == 0;
}

bool DirectoryExists(const std::string& path) {
    if (path.empty()) return false;

#ifdef _WIN32
    struct _stat info;
    if (_stat(path.c_str(), &info) != 0) return false;
    return (info.st_mode & _S_IFDIR) != 0;
#else
    struct stat info;
    if (stat(path.c_str(), &info) != 0) return false;
    return S_ISDIR(info.st_mode);
#endif
}

int64_t GetFileSize(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return -1;
    return static_cast<int64_t>(file.tellg());
}

std::string GetExtension(const std::string& path) {
    size_t dotPos = path.rfind('.');
    size_t slashPos = path.find_last_of("/\\");

    // Dot must be after last slash (if any)
    if (dotPos == std::string::npos ||
        (slashPos != std::string::npos && dotPos < slashPos)) {
        return "";
    }

    return path.substr(dotPos);
}

std::string GetFileName(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return path;
    }
    return path.substr(pos + 1);
}

std::string GetDirectory(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return "";
    }
    return path.substr(0, pos);
}

std::string JoinPath(const std::string& dir, const std::string& name) {
    if (dir.empty()) return name;
    if (name.empty()) return dir;

    char lastChar = dir.back();
    if (lastChar == '/' || lastChar == '\\') {
        return dir + name;
    }

#ifdef _WIN32
    return dir + "\\" + name;
#else
    return dir + "/" + name;
#endif
}

std::string NormalizePath(const std::string& path) {
    std::string result = path;

#ifdef _WIN32
    std::replace(result.begin(), result.end(), '/', '\\');
#else
    std::replace(result.begin(), result.end(), '\\', '/');
#endif

    return result;
}

bool CreateDirectory(const std::string& path) {
    if (path.empty()) return false;
    if (DirectoryExists(path)) return true;

    // Create parent directories first
    std::string parent = GetDirectory(path);
    if (!parent.empty() && !DirectoryExists(parent)) {
        if (!CreateDirectory(parent)) {
            return false;
        }
    }

    return MKDIR(path.c_str()) == 0;
}

bool DeleteFile(const std::string& path) {
    if (!FileExists(path)) return true;
    return std::remove(path.c_str()) == 0;
}

// ============================================================================
// Binary File I/O
// ============================================================================

bool ReadBinaryFile(const std::string& path, std::vector<uint8_t>& data) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return false;
    }

    std::streamsize size = file.tellg();
    if (size <= 0) {
        data.clear();
        return true;
    }

    file.seekg(0, std::ios::beg);
    data.resize(static_cast<size_t>(size));

    if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
        return false;
    }

    return true;
}

bool WriteBinaryFile(const std::string& path, const std::vector<uint8_t>& data) {
    return WriteBinaryFile(path, data.data(), data.size());
}

bool WriteBinaryFile(const std::string& path, const void* data, size_t size) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    if (size > 0) {
        file.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    }

    return file.good();
}

// ============================================================================
// Text File I/O
// ============================================================================

bool ReadTextFile(const std::string& path, std::string& content) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    content = buffer.str();

    return true;
}

bool WriteTextFile(const std::string& path, const std::string& content) {
    std::ofstream file(path);
    if (!file.is_open()) {
        return false;
    }

    file << content;
    return file.good();
}

// Helper: trim whitespace from string
static std::string TrimString(const std::string& str) {
    const char* whitespace = " \t\r\n";
    size_t start = str.find_first_not_of(whitespace);
    if (start == std::string::npos) return "";
    size_t end = str.find_last_not_of(whitespace);
    return str.substr(start, end - start + 1);
}

bool ReadTextLines(const std::string& path, std::vector<std::string>& lines,
                   bool trimLines) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }

    lines.clear();
    std::string line;
    while (std::getline(file, line)) {
        if (trimLines) {
            lines.push_back(TrimString(line));
        } else {
            lines.push_back(line);
        }
    }

    return true;
}

bool WriteTextLines(const std::string& path, const std::vector<std::string>& lines,
                    const std::string& lineEnding) {
    std::ofstream file(path);
    if (!file.is_open()) {
        return false;
    }

    for (const auto& line : lines) {
        file << line << lineEnding;
    }

    return file.good();
}

// ============================================================================
// BinaryWriter
// ============================================================================

BinaryWriter::BinaryWriter(const std::string& path)
    : stream_(path, std::ios::binary) {
}

void BinaryWriter::WriteString(const std::string& str) {
    uint64_t len = str.size();
    Write(len);
    if (!str.empty()) {
        stream_.write(str.data(), static_cast<std::streamsize>(str.size()));
    }
}

void BinaryWriter::WriteBytes(const void* data, size_t size) {
    if (size > 0) {
        stream_.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    }
}

// ============================================================================
// BinaryReader
// ============================================================================

BinaryReader::BinaryReader(const std::string& path)
    : stream_(path, std::ios::binary) {
}

std::string BinaryReader::ReadString() {
    uint64_t len = Read<uint64_t>();
    if (len == 0) return "";

    std::string str(static_cast<size_t>(len), '\0');
    stream_.read(&str[0], static_cast<std::streamsize>(len));
    return str;
}

void BinaryReader::ReadBytes(void* data, size_t size) {
    if (size > 0) {
        stream_.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
    }
}

} // namespace Qi::Vision::Platform
