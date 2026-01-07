#include <gtest/gtest.h>
#include <QiVision/QiVision.h>
#include <QiVision/Platform/SIMD.h>

#include <iostream>

int main(int argc, char** argv) {
    // Print library info
    std::cout << "========================================\n";
    std::cout << "QiVision Unit Tests\n";
    std::cout << "Version: " << Qi::Vision::GetVersion() << "\n";
    std::cout << "SIMD: " << Qi::Vision::Platform::GetSIMDLevelName() << "\n";
    std::cout << "========================================\n\n";

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
