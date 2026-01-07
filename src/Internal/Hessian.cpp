/**
 * @file Hessian.cpp
 * @brief Hessian matrix computation implementation
 */

#include <QiVision/Internal/Hessian.h>
#include <QiVision/Internal/Interpolate.h>

#include <algorithm>
#include <cmath>

namespace Qi::Vision::Internal {

// ============================================================================
// 2x2 Eigenvalue Decomposition
// ============================================================================

void EigenDecompose2x2(double a, double b, double c,
                        double& lambda1, double& lambda2,
                        double& nx, double& ny) {
    // For 2x2 symmetric matrix A = [a b; b c]
    // Eigenvalues: λ = (a+c)/2 ± sqrt(((a-c)/2)² + b²)

    double trace = a + c;
    double det = a * c - b * b;

    (void)det;  // Unused but computed for reference
    double halfTrace = trace * 0.5;
    double diff = (a - c) * 0.5;
    double sqrtTerm = std::sqrt(diff * diff + b * b);

    // Two eigenvalues
    double l1 = halfTrace + sqrtTerm;
    double l2 = halfTrace - sqrtTerm;

    // Sort by absolute value: |lambda1| >= |lambda2|
    if (std::abs(l1) >= std::abs(l2)) {
        lambda1 = l1;
        lambda2 = l2;
    } else {
        lambda1 = l2;
        lambda2 = l1;
    }

    // Compute eigenvector for lambda1
    // (A - λI)v = 0
    // [a-λ  b  ][nx]   [0]
    // [b    c-λ][ny] = [0]

    // From first row: (a-λ1)*nx + b*ny = 0
    // => nx = -b, ny = (a - λ1)  (or use second row)

    double aMinusL = a - lambda1;
    double cMinusL = c - lambda1;

    // Choose the row with larger coefficients for numerical stability
    if (std::abs(aMinusL) + std::abs(b) >= std::abs(b) + std::abs(cMinusL)) {
        // Use first row: (a-λ1)*nx + b*ny = 0
        if (std::abs(b) > 1e-10) {
            nx = 1.0;
            ny = -aMinusL / b;
        } else if (std::abs(aMinusL) > 1e-10) {
            nx = 0.0;
            ny = 1.0;
        } else {
            // Degenerate case: a ≈ λ1 and b ≈ 0
            nx = 1.0;
            ny = 0.0;
        }
    } else {
        // Use second row: b*nx + (c-λ1)*ny = 0
        if (std::abs(cMinusL) > 1e-10) {
            nx = 1.0;
            ny = -b / cMinusL;
        } else if (std::abs(b) > 1e-10) {
            nx = 0.0;
            ny = 1.0;
        } else {
            nx = 1.0;
            ny = 0.0;
        }
    }

    // Normalize
    double len = std::sqrt(nx * nx + ny * ny);
    if (len > 1e-10) {
        nx /= len;
        ny /= len;
    }
}

void EigenDecompose2x2Full(double a, double b, double c,
                            double& lambda1, double& lambda2,
                            double& nx, double& ny,
                            double& tx, double& ty) {
    // First compute eigenvalues and principal eigenvector
    EigenDecompose2x2(a, b, c, lambda1, lambda2, nx, ny);

    // Tangent vector is perpendicular to normal
    // If n = (nx, ny), then t = (-ny, nx) or (ny, -nx)
    // We choose (-ny, nx) for consistent handedness
    tx = -ny;
    ty = nx;
}

// ============================================================================
// Whole Image Hessian (QImage version)
// ============================================================================

void ComputeHessianImage(const QImage& image, double sigma,
                          QImage& dxxImage, QImage& dxyImage, QImage& dyyImage,
                          BorderMode border) {
    if (!image.IsValid()) {
        dxxImage = QImage();
        dxyImage = QImage();
        dyyImage = QImage();
        return;
    }

    int32_t width = image.Width();
    int32_t height = image.Height();

    // Create output images as float
    dxxImage = QImage(width, height, PixelType::Float32);
    dxyImage = QImage(width, height, PixelType::Float32);
    dyyImage = QImage(width, height, PixelType::Float32);

    // Get pointers
    float* dxxPtr = static_cast<float*>(dxxImage.Data());
    float* dxyPtr = static_cast<float*>(dxyImage.Data());
    float* dyyPtr = static_cast<float*>(dyyImage.Data());

    // Process based on input type
    switch (image.Type()) {
        case PixelType::UInt8: {
            const uint8_t* data = static_cast<const uint8_t*>(image.Data());
            std::vector<float> dxx, dxy, dyy;
            ComputeHessianImage(data, width, height, sigma, dxx, dxy, dyy, border);
            std::copy(dxx.begin(), dxx.end(), dxxPtr);
            std::copy(dxy.begin(), dxy.end(), dxyPtr);
            std::copy(dyy.begin(), dyy.end(), dyyPtr);
            break;
        }
        case PixelType::UInt16: {
            const uint16_t* data = static_cast<const uint16_t*>(image.Data());
            std::vector<float> dxx, dxy, dyy;
            ComputeHessianImage(data, width, height, sigma, dxx, dxy, dyy, border);
            std::copy(dxx.begin(), dxx.end(), dxxPtr);
            std::copy(dxy.begin(), dxy.end(), dxyPtr);
            std::copy(dyy.begin(), dyy.end(), dyyPtr);
            break;
        }
        case PixelType::Float32: {
            const float* data = static_cast<const float*>(image.Data());
            std::vector<float> dxx, dxy, dyy;
            ComputeHessianImage(data, width, height, sigma, dxx, dxy, dyy, border);
            std::copy(dxx.begin(), dxx.end(), dxxPtr);
            std::copy(dxy.begin(), dxy.end(), dxyPtr);
            std::copy(dyy.begin(), dyy.end(), dyyPtr);
            break;
        }
        default:
            // Unsupported format - return empty images
            dxxImage = QImage();
            dxyImage = QImage();
            dyyImage = QImage();
            break;
    }
}

// ============================================================================
// Eigenvalue Images
// ============================================================================

void ComputeEigenvalueImages(const float* dxx, const float* dxy, const float* dyy,
                              int32_t width, int32_t height,
                              std::vector<float>& lambda1,
                              std::vector<float>& lambda2) {
    size_t size = static_cast<size_t>(width * height);
    lambda1.resize(size);
    lambda2.resize(size);

    for (size_t i = 0; i < size; ++i) {
        double a = static_cast<double>(dxx[i]);
        double b = static_cast<double>(dxy[i]);
        double c = static_cast<double>(dyy[i]);

        // Eigenvalues: λ = (a+c)/2 ± sqrt(((a-c)/2)² + b²)
        double halfTrace = (a + c) * 0.5;
        double diff = (a - c) * 0.5;
        double sqrtTerm = std::sqrt(diff * diff + b * b);

        double l1 = halfTrace + sqrtTerm;
        double l2 = halfTrace - sqrtTerm;

        // Sort by absolute value
        if (std::abs(l1) >= std::abs(l2)) {
            lambda1[i] = static_cast<float>(l1);
            lambda2[i] = static_cast<float>(l2);
        } else {
            lambda1[i] = static_cast<float>(l2);
            lambda2[i] = static_cast<float>(l1);
        }
    }
}

void ComputeEigenvectorImages(const float* dxx, const float* dxy, const float* dyy,
                               int32_t width, int32_t height,
                               std::vector<float>& nx,
                               std::vector<float>& ny) {
    size_t size = static_cast<size_t>(width * height);
    nx.resize(size);
    ny.resize(size);

    for (size_t i = 0; i < size; ++i) {
        double a = static_cast<double>(dxx[i]);
        double b = static_cast<double>(dxy[i]);
        double c = static_cast<double>(dyy[i]);

        double l1, l2, nxVal, nyVal;
        EigenDecompose2x2(a, b, c, l1, l2, nxVal, nyVal);

        nx[i] = static_cast<float>(nxVal);
        ny[i] = static_cast<float>(nyVal);
    }
}

// ============================================================================
// Ridge/Valley Response
// ============================================================================

void ComputeRidgeResponse(const float* lambda1, int32_t width, int32_t height,
                           std::vector<float>& response) {
    size_t size = static_cast<size_t>(width * height);
    response.resize(size);

    for (size_t i = 0; i < size; ++i) {
        // Ridge: λ1 < 0, response = -λ1
        response[i] = (lambda1[i] < 0) ? -lambda1[i] : 0.0f;
    }
}

void ComputeValleyResponse(const float* lambda1, int32_t width, int32_t height,
                            std::vector<float>& response) {
    size_t size = static_cast<size_t>(width * height);
    response.resize(size);

    for (size_t i = 0; i < size; ++i) {
        // Valley: λ1 > 0, response = λ1
        response[i] = (lambda1[i] > 0) ? lambda1[i] : 0.0f;
    }
}

void ComputeLineResponse(const float* lambda1, int32_t width, int32_t height,
                          std::vector<float>& response) {
    size_t size = static_cast<size_t>(width * height);
    response.resize(size);

    for (size_t i = 0; i < size; ++i) {
        // Combined response: |λ1|
        response[i] = std::abs(lambda1[i]);
    }
}

// ============================================================================
// Non-Maximum Suppression Helper
// ============================================================================

bool IsRidgeMaximum(const float* lambda1, const float* nx, const float* ny,
                     int32_t width, int32_t height, int32_t x, int32_t y) {
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
        return false;
    }

    size_t idx = static_cast<size_t>(y * width + x);

    // Must be a ridge point (λ1 < 0)
    float l1 = lambda1[idx];
    if (l1 >= 0) {
        return false;
    }

    // Get principal direction (normal to ridge)
    float normalX = nx[idx];
    float normalY = ny[idx];

    // Response value at current point
    float currentResponse = -l1;  // Ridge response = -λ1

    // Check neighbors along the normal direction
    // We use bilinear interpolation for subpixel positions

    // Forward neighbor (along +n direction)
    double fx1 = x + normalX;
    double fy1 = y + normalY;

    // Backward neighbor (along -n direction)
    double fx2 = x - normalX;
    double fy2 = y - normalY;

    // Clamp to image bounds
    fx1 = std::max(0.0, std::min(static_cast<double>(width - 1), fx1));
    fy1 = std::max(0.0, std::min(static_cast<double>(height - 1), fy1));
    fx2 = std::max(0.0, std::min(static_cast<double>(width - 1), fx2));
    fy2 = std::max(0.0, std::min(static_cast<double>(height - 1), fy2));

    // Bilinear interpolation of response at neighbor positions
    auto interpolateResponse = [&](double fx, double fy) -> float {
        int32_t x0 = static_cast<int32_t>(std::floor(fx));
        int32_t y0 = static_cast<int32_t>(std::floor(fy));
        int32_t x1 = std::min(x0 + 1, width - 1);
        int32_t y1 = std::min(y0 + 1, height - 1);

        double dx = fx - x0;
        double dy = fy - y0;

        float v00 = -std::min(0.0f, lambda1[y0 * width + x0]);
        float v10 = -std::min(0.0f, lambda1[y0 * width + x1]);
        float v01 = -std::min(0.0f, lambda1[y1 * width + x0]);
        float v11 = -std::min(0.0f, lambda1[y1 * width + x1]);

        return static_cast<float>(
            (1-dx)*(1-dy)*v00 + dx*(1-dy)*v10 +
            (1-dx)*dy*v01 + dx*dy*v11
        );
    };

    float response1 = interpolateResponse(fx1, fy1);
    float response2 = interpolateResponse(fx2, fy2);

    // Current point is maximum if its response is greater than both neighbors
    return (currentResponse >= response1) && (currentResponse >= response2);
}

} // namespace Qi::Vision::Internal
