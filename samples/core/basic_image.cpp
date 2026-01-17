/**
 * @file 01_basic_image.cpp
 * @brief 示例：基本图像操作 / Example: Basic Image Operations
 */

#include <QiVision/QiVision.h>
#include <cstdio>
#include <cstring>

using namespace Qi::Vision;

int main() {
    printf("=== QiVision Sample: Basic Image Operations ===\n\n");

    // 1. 创建灰度图像 / Create grayscale image
    printf("1. Creating a 640x480 grayscale image...\n");
    QImage gray(640, 480, PixelType::UInt8, ChannelType::Gray);
    printf("   Size: %dx%d, Channels: %d\n",
           gray.Width(), gray.Height(), gray.Channels());

    // 2. 填充像素值 / Fill with pixel values
    printf("\n2. Filling image with gradient...\n");
    for (int32_t y = 0; y < gray.Height(); ++y) {
        uint8_t* row = static_cast<uint8_t*>(gray.RowPtr(y));
        for (int32_t x = 0; x < gray.Width(); ++x) {
            row[x] = static_cast<uint8_t>((x + y) % 256);
        }
    }

    // 3. 访问单个像素 / Access single pixel
    printf("\n3. Pixel access:\n");
    printf("   Pixel at (100, 100) = %d\n", gray.At(100, 100));
    printf("   Pixel at (200, 150) = %d\n", gray.At(200, 150));

    // 4. 保存图像 / Save image
    const char* outputPath = "output_gradient.png";
    printf("\n4. Saving image to '%s'...\n", outputPath);
    if (gray.SaveToFile(outputPath)) {
        printf("   Success!\n");
    } else {
        printf("   Failed to save.\n");
    }

    // 5. 加载图像 / Load image
    printf("\n5. Loading image back...\n");
    QImage loaded = QImage::FromFile(outputPath);
    if (!loaded.Empty()) {
        printf("   Loaded: %dx%d, %d channels\n",
               loaded.Width(), loaded.Height(), loaded.Channels());
    } else {
        printf("   Could not load image.\n");
    }

    // 6. 图像克隆 / Clone image
    printf("\n6. Cloning image...\n");
    QImage cloned = gray.Clone();
    printf("   Original data ptr: %p\n", gray.Data());
    printf("   Cloned data ptr:   %p (different = deep copy)\n", cloned.Data());

    // 7. 图像信息 / Image info
    printf("\n7. Image properties:\n");
    printf("   Stride (bytes per row): %zu\n", gray.Stride());
    printf("   Is empty: %s\n", gray.Empty() ? "yes" : "no");
    printf("   Is full domain: %s\n", gray.IsFullDomain() ? "yes" : "no");

    printf("\n=== Done ===\n");
    return 0;
}
