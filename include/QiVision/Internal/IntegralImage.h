#pragma once

/**
 * @file IntegralImage.h
 * @brief Integral image (summed area table) for fast region computations
 *
 * Provides:
 * - Integral image computation (sum, squared sum)
 * - O(1) rectangular region sum queries
 * - Used for fast NCC template matching
 *
 * Integral image I(x,y) = sum of all pixels above and to the left of (x,y)
 * Region sum = I(x2,y2) - I(x1-1,y2) - I(x2,y1-1) + I(x1-1,y1-1)
 *
 * @see NCCModel for usage in template matching
 */

#include <QiVision/Core/QImage.h>
#include <QiVision/Core/Types.h>

#include <cstdint>
#include <memory>

namespace Qi::Vision::Internal {

// =============================================================================
// IntegralImage Class
// =============================================================================

/**
 * @brief Integral image for fast rectangular region sum computation
 *
 * @code
 * IntegralImage integral;
 * integral.Compute(image);
 *
 * // Get sum of pixels in rectangle (x1, y1) to (x2, y2)
 * double sum = integral.GetRectSum(x1, y1, x2, y2);
 *
 * // Get mean of pixels in rectangle
 * double mean = integral.GetRectMean(x1, y1, x2, y2);
 *
 * // Get variance of pixels in rectangle (needs squared integral)
 * integral.Compute(image, true);  // with squared integral
 * double variance = integral.GetRectVariance(x1, y1, x2, y2);
 * @endcode
 */
class IntegralImage {
public:
    IntegralImage();
    ~IntegralImage();
    IntegralImage(const IntegralImage& other);
    IntegralImage(IntegralImage&& other) noexcept;
    IntegralImage& operator=(const IntegralImage& other);
    IntegralImage& operator=(IntegralImage&& other) noexcept;

    // =========================================================================
    // Construction
    // =========================================================================

    /**
     * @brief Compute integral image from source image
     * @param image Input grayscale image (uint8 or float)
     * @param computeSquared Also compute squared integral (for variance)
     * @return true if successful
     */
    bool Compute(const QImage& image, bool computeSquared = false);

    /**
     * @brief Clear the integral image
     */
    void Clear();

    /**
     * @brief Check if integral image is valid
     */
    bool IsValid() const;

    /**
     * @brief Check if squared integral is available
     */
    bool HasSquaredIntegral() const;

    // =========================================================================
    // Properties
    // =========================================================================

    int32_t Width() const;
    int32_t Height() const;

    // =========================================================================
    // Sum Queries
    // =========================================================================

    /**
     * @brief Get sum of pixels in rectangular region
     * @param x1, y1 Top-left corner (inclusive)
     * @param x2, y2 Bottom-right corner (inclusive)
     * @return Sum of pixel values in the region
     */
    double GetRectSum(int32_t x1, int32_t y1, int32_t x2, int32_t y2) const;

    /**
     * @brief Get sum of squared pixels in rectangular region
     * @note Requires computeSquared=true in Compute()
     */
    double GetRectSumSquared(int32_t x1, int32_t y1, int32_t x2, int32_t y2) const;

    /**
     * @brief Get mean of pixels in rectangular region
     */
    double GetRectMean(int32_t x1, int32_t y1, int32_t x2, int32_t y2) const;

    /**
     * @brief Get variance of pixels in rectangular region
     * @note Requires computeSquared=true in Compute()
     * @return Variance = E[X^2] - E[X]^2
     */
    double GetRectVariance(int32_t x1, int32_t y1, int32_t x2, int32_t y2) const;

    /**
     * @brief Get standard deviation of pixels in rectangular region
     */
    double GetRectStdDev(int32_t x1, int32_t y1, int32_t x2, int32_t y2) const;

    /**
     * @brief Get pixel count in rectangular region
     */
    int64_t GetRectCount(int32_t x1, int32_t y1, int32_t x2, int32_t y2) const;

    // =========================================================================
    // Raw Access
    // =========================================================================

    /**
     * @brief Get integral value at position
     * @param x X coordinate (column)
     * @param y Y coordinate (row)
     * @return Integral value (sum of all pixels above and left of this position)
     */
    double GetIntegralAt(int32_t x, int32_t y) const;

    /**
     * @brief Get squared integral value at position
     */
    double GetSquaredIntegralAt(int32_t x, int32_t y) const;

    /**
     * @brief Get raw integral data (double precision, row-major)
     * @return Pointer to integral data, or nullptr if invalid
     * @note Size is (width+1) * (height+1)
     */
    const double* GetIntegralData() const;

    /**
     * @brief Get raw squared integral data
     */
    const double* GetSquaredIntegralData() const;

    /**
     * @brief Get integral image dimensions
     * @return Width of integral image (original width + 1)
     */
    int32_t IntegralWidth() const;
    int32_t IntegralHeight() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Compute integral image into pre-allocated buffer
 * @param src Source image data (uint8 or float)
 * @param srcWidth Source width
 * @param srcHeight Source height
 * @param srcStride Source stride in elements
 * @param integral Output integral buffer (must be (width+1)*(height+1) doubles)
 * @param intStride Integral stride (typically srcWidth + 1)
 */
template<typename T>
void ComputeIntegralImage(const T* src, int32_t srcWidth, int32_t srcHeight,
                          int32_t srcStride, double* integral, int32_t intStride);

/**
 * @brief Get rectangular sum from integral image buffer
 * @param integral Integral image data
 * @param intWidth Integral image width
 * @param intHeight Integral image height
 * @param x1, y1 Top-left corner (in source coordinates)
 * @param x2, y2 Bottom-right corner (in source coordinates)
 * @return Sum of pixel values in region
 */
double GetRectSumFromIntegral(const double* integral, int32_t intWidth, int32_t intHeight,
                               int32_t x1, int32_t y1, int32_t x2, int32_t y2);

} // namespace Qi::Vision::Internal
