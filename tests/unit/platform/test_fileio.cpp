/**
 * @file test_fileio.cpp
 * @brief Unit tests for Platform/FileIO.h
 */

#include <QiVision/Platform/FileIO.h>
#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>
#include <fstream>

using namespace Qi::Vision::Platform;

class FileIOTest : public ::testing::Test {
protected:
    std::string testDir_;
    std::string testFile_;

    void SetUp() override {
        testDir_ = "/tmp/qivision_test";
        testFile_ = testDir_ + "/test_file.txt";

        // Create test directory
        CreateDirectory(testDir_);
    }

    void TearDown() override {
        // Cleanup test files
        DeleteFile(testFile_);
        DeleteFile(testDir_ + "/test.bin");
        DeleteFile(testDir_ + "/test_lines.txt");
        DeleteFile(testDir_ + "/binary_writer.bin");
        // Note: Directory cleanup not implemented for simplicity
    }
};

// ============================================================================
// Path Utilities Tests
// ============================================================================

TEST_F(FileIOTest, FileExistsTrue) {
    // Create a file
    std::ofstream f(testFile_);
    f << "test";
    f.close();

    EXPECT_TRUE(FileExists(testFile_));
}

TEST_F(FileIOTest, FileExistsFalse) {
    EXPECT_FALSE(FileExists(testDir_ + "/nonexistent.txt"));
}

TEST_F(FileIOTest, FileExistsEmpty) {
    EXPECT_FALSE(FileExists(""));
}

TEST_F(FileIOTest, DirectoryExistsTrue) {
    EXPECT_TRUE(DirectoryExists(testDir_));
}

TEST_F(FileIOTest, DirectoryExistsFalse) {
    EXPECT_FALSE(DirectoryExists(testDir_ + "/nonexistent_dir"));
}

TEST_F(FileIOTest, GetFileSize) {
    std::string content = "Hello, World!";
    std::ofstream f(testFile_);
    f << content;
    f.close();

    int64_t size = GetFileSize(testFile_);
    EXPECT_EQ(size, static_cast<int64_t>(content.size()));
}

TEST_F(FileIOTest, GetFileSizeNonexistent) {
    EXPECT_EQ(GetFileSize(testDir_ + "/nonexistent.txt"), -1);
}

TEST_F(FileIOTest, GetExtension) {
    EXPECT_EQ(GetExtension("file.txt"), ".txt");
    EXPECT_EQ(GetExtension("file.tar.gz"), ".gz");
    EXPECT_EQ(GetExtension("file"), "");
    EXPECT_EQ(GetExtension("/path/to/file.jpg"), ".jpg");
    EXPECT_EQ(GetExtension("/path/.hidden"), ".hidden");
}

TEST_F(FileIOTest, GetFileName) {
    EXPECT_EQ(GetFileName("file.txt"), "file.txt");
    EXPECT_EQ(GetFileName("/path/to/file.txt"), "file.txt");
    EXPECT_EQ(GetFileName("C:\\path\\to\\file.txt"), "file.txt");
    EXPECT_EQ(GetFileName("/path/"), "");
}

TEST_F(FileIOTest, GetDirectory) {
    EXPECT_EQ(GetDirectory("file.txt"), "");
    EXPECT_EQ(GetDirectory("/path/to/file.txt"), "/path/to");
    EXPECT_EQ(GetDirectory("C:\\path\\to\\file.txt"), "C:\\path\\to");
}

TEST_F(FileIOTest, JoinPath) {
    std::string joined = JoinPath(testDir_, "subdir");
    EXPECT_FALSE(joined.empty());
    EXPECT_NE(joined.find("subdir"), std::string::npos);
}

TEST_F(FileIOTest, JoinPathEmpty) {
    EXPECT_EQ(JoinPath("", "file.txt"), "file.txt");
    EXPECT_EQ(JoinPath("/path", ""), "/path");
}

TEST_F(FileIOTest, JoinPathTrailingSlash) {
    std::string joined = JoinPath("/path/", "file.txt");
    EXPECT_EQ(joined, "/path/file.txt");
}

TEST_F(FileIOTest, NormalizePath) {
    std::string normalized = NormalizePath("/path\\to/file.txt");
    // Should have consistent separators
#ifdef _WIN32
    EXPECT_EQ(normalized, "\\path\\to\\file.txt");
#else
    EXPECT_EQ(normalized, "/path/to/file.txt");
#endif
}

TEST_F(FileIOTest, CreateDirectoryNew) {
    std::string newDir = testDir_ + "/new_subdir";
    EXPECT_TRUE(CreateDirectory(newDir));
    EXPECT_TRUE(DirectoryExists(newDir));

    // Cleanup
    rmdir(newDir.c_str());
}

TEST_F(FileIOTest, CreateDirectoryExisting) {
    EXPECT_TRUE(CreateDirectory(testDir_));
}

TEST_F(FileIOTest, DeleteFileExisting) {
    std::ofstream f(testFile_);
    f << "test";
    f.close();

    EXPECT_TRUE(DeleteFile(testFile_));
    EXPECT_FALSE(FileExists(testFile_));
}

TEST_F(FileIOTest, DeleteFileNonexistent) {
    EXPECT_TRUE(DeleteFile(testDir_ + "/nonexistent.txt"));
}

// ============================================================================
// Binary File I/O Tests
// ============================================================================

TEST_F(FileIOTest, ReadWriteBinaryFile) {
    std::string binPath = testDir_ + "/test.bin";
    std::vector<uint8_t> dataWrite = {0x01, 0x02, 0x03, 0xFF, 0x00, 0xAB};

    EXPECT_TRUE(WriteBinaryFile(binPath, dataWrite));
    EXPECT_TRUE(FileExists(binPath));

    std::vector<uint8_t> dataRead;
    EXPECT_TRUE(ReadBinaryFile(binPath, dataRead));

    EXPECT_EQ(dataRead, dataWrite);
}

TEST_F(FileIOTest, ReadWriteBinaryFileEmpty) {
    std::string binPath = testDir_ + "/test.bin";
    std::vector<uint8_t> empty;

    EXPECT_TRUE(WriteBinaryFile(binPath, empty));

    std::vector<uint8_t> dataRead;
    EXPECT_TRUE(ReadBinaryFile(binPath, dataRead));
    EXPECT_TRUE(dataRead.empty());
}

TEST_F(FileIOTest, ReadBinaryFileNonexistent) {
    std::vector<uint8_t> data;
    EXPECT_FALSE(ReadBinaryFile(testDir_ + "/nonexistent.bin", data));
}

TEST_F(FileIOTest, WriteBinaryFileRawPointer) {
    std::string binPath = testDir_ + "/test.bin";
    uint8_t data[] = {0xDE, 0xAD, 0xBE, 0xEF};

    EXPECT_TRUE(WriteBinaryFile(binPath, data, sizeof(data)));

    std::vector<uint8_t> dataRead;
    EXPECT_TRUE(ReadBinaryFile(binPath, dataRead));
    EXPECT_EQ(dataRead.size(), sizeof(data));
    EXPECT_EQ(std::memcmp(dataRead.data(), data, sizeof(data)), 0);
}

// ============================================================================
// Text File I/O Tests
// ============================================================================

TEST_F(FileIOTest, ReadWriteTextFile) {
    std::string content = "Hello, World!\nThis is a test.\n日本語テスト";

    EXPECT_TRUE(WriteTextFile(testFile_, content));

    std::string readContent;
    EXPECT_TRUE(ReadTextFile(testFile_, readContent));

    EXPECT_EQ(readContent, content);
}

TEST_F(FileIOTest, ReadWriteTextFileEmpty) {
    EXPECT_TRUE(WriteTextFile(testFile_, ""));

    std::string content;
    EXPECT_TRUE(ReadTextFile(testFile_, content));
    EXPECT_TRUE(content.empty());
}

TEST_F(FileIOTest, ReadTextFileNonexistent) {
    std::string content;
    EXPECT_FALSE(ReadTextFile(testDir_ + "/nonexistent.txt", content));
}

TEST_F(FileIOTest, ReadWriteTextLines) {
    std::string linesPath = testDir_ + "/test_lines.txt";
    std::vector<std::string> linesWrite = {
        "First line",
        "Second line",
        "Third line with spaces  ",
        ""
    };

    EXPECT_TRUE(WriteTextLines(linesPath, linesWrite));

    std::vector<std::string> linesRead;
    EXPECT_TRUE(ReadTextLines(linesPath, linesRead, true));

    // With trimming, trailing spaces should be removed
    EXPECT_EQ(linesRead.size(), linesWrite.size());
    EXPECT_EQ(linesRead[0], "First line");
    EXPECT_EQ(linesRead[2], "Third line with spaces");
}

TEST_F(FileIOTest, ReadTextLinesNoTrim) {
    std::string linesPath = testDir_ + "/test_lines.txt";
    std::vector<std::string> linesWrite = {"  spaces  "};

    EXPECT_TRUE(WriteTextLines(linesPath, linesWrite));

    std::vector<std::string> linesRead;
    EXPECT_TRUE(ReadTextLines(linesPath, linesRead, false));

    EXPECT_EQ(linesRead[0], "  spaces  ");
}

// ============================================================================
// BinaryWriter/Reader Tests
// ============================================================================

TEST_F(FileIOTest, BinaryWriterReader) {
    std::string binPath = testDir_ + "/binary_writer.bin";

    // Write
    {
        BinaryWriter writer(binPath);
        EXPECT_TRUE(writer.IsOpen());

        writer.Write<int32_t>(42);
        writer.Write<double>(3.14159);
        writer.Write<uint8_t>(255);
        writer.WriteString("Hello");

        std::vector<float> floats = {1.0f, 2.0f, 3.0f};
        writer.WriteVector(floats);
    }

    // Read
    {
        BinaryReader reader(binPath);
        EXPECT_TRUE(reader.IsOpen());

        EXPECT_EQ(reader.Read<int32_t>(), 42);
        EXPECT_DOUBLE_EQ(reader.Read<double>(), 3.14159);
        EXPECT_EQ(reader.Read<uint8_t>(), 255);
        EXPECT_EQ(reader.ReadString(), "Hello");

        auto floats = reader.ReadVector<float>();
        EXPECT_EQ(floats.size(), 3u);
        EXPECT_FLOAT_EQ(floats[0], 1.0f);
        EXPECT_FLOAT_EQ(floats[1], 2.0f);
        EXPECT_FLOAT_EQ(floats[2], 3.0f);
    }
}

TEST_F(FileIOTest, BinaryWriterEmptyVector) {
    std::string binPath = testDir_ + "/binary_writer.bin";

    {
        BinaryWriter writer(binPath);
        std::vector<int> empty;
        writer.WriteVector(empty);
    }

    {
        BinaryReader reader(binPath);
        auto vec = reader.ReadVector<int>();
        EXPECT_TRUE(vec.empty());
    }
}

TEST_F(FileIOTest, BinaryWriterEmptyString) {
    std::string binPath = testDir_ + "/binary_writer.bin";

    {
        BinaryWriter writer(binPath);
        writer.WriteString("");
    }

    {
        BinaryReader reader(binPath);
        EXPECT_EQ(reader.ReadString(), "");
    }
}

TEST_F(FileIOTest, BinaryReaderNonexistent) {
    BinaryReader reader(testDir_ + "/nonexistent.bin");
    EXPECT_FALSE(reader.IsOpen());
}

TEST_F(FileIOTest, BinaryWriterArray) {
    std::string binPath = testDir_ + "/binary_writer.bin";

    int32_t data[] = {1, 2, 3, 4, 5};

    {
        BinaryWriter writer(binPath);
        writer.WriteArray(data, 5);
    }

    {
        BinaryReader reader(binPath);
        int32_t readData[5];
        reader.ReadArray(readData, 5);

        for (int i = 0; i < 5; ++i) {
            EXPECT_EQ(readData[i], data[i]);
        }
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(FileIOTest, PathWithSpaces) {
    std::string spacePath = testDir_ + "/file with spaces.txt";

    EXPECT_TRUE(WriteTextFile(spacePath, "content"));
    EXPECT_TRUE(FileExists(spacePath));

    std::string content;
    EXPECT_TRUE(ReadTextFile(spacePath, content));
    EXPECT_EQ(content, "content");

    DeleteFile(spacePath);
}

TEST_F(FileIOTest, LargeBinaryFile) {
    std::string binPath = testDir_ + "/test.bin";

    // 1MB of data
    std::vector<uint8_t> largeData(1024 * 1024);
    for (size_t i = 0; i < largeData.size(); ++i) {
        largeData[i] = static_cast<uint8_t>(i % 256);
    }

    EXPECT_TRUE(WriteBinaryFile(binPath, largeData));
    EXPECT_EQ(GetFileSize(binPath), static_cast<int64_t>(largeData.size()));

    std::vector<uint8_t> readData;
    EXPECT_TRUE(ReadBinaryFile(binPath, readData));
    EXPECT_EQ(readData, largeData);
}
