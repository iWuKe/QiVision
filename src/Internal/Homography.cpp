#include <QiVision/Internal/Homography.h>
#include <QiVision/Internal/Solver.h>
#include <QiVision/Platform/Random.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace Qi::Vision::Internal {

// =============================================================================
// Constants
// =============================================================================

constexpr double EPSILON = 1e-10;
constexpr double PI = 3.14159265358979323846;

// =============================================================================
// Helper Functions
// =============================================================================

namespace {

// Clamp coordinate to valid range
inline int32_t ClampCoord(int32_t val, int32_t maxVal) {
    return std::max(0, std::min(val, maxVal - 1));
}

// Get pixel with border handling
template<typename T>
T GetPixelWithBorder(const T* data, int32_t width, int32_t height,
                     int32_t x, int32_t y, BorderMode mode, T borderValue) {
    if (x >= 0 && x < width && y >= 0 && y < height) {
        return data[y * width + x];
    }

    switch (mode) {
        case BorderMode::Constant:
            return borderValue;
        case BorderMode::Replicate:
            x = ClampCoord(x, width);
            y = ClampCoord(y, height);
            return data[y * width + x];
        case BorderMode::Reflect:
            if (x < 0) x = -x;
            if (x >= width) x = 2 * width - x - 2;
            if (y < 0) y = -y;
            if (y >= height) y = 2 * height - y - 2;
            x = ClampCoord(x, width);
            y = ClampCoord(y, height);
            return data[y * width + x];
        case BorderMode::Reflect101:
            if (x < 0) x = -x - 1;
            if (x >= width) x = 2 * width - x - 1;
            if (y < 0) y = -y - 1;
            if (y >= height) y = 2 * height - y - 1;
            x = ClampCoord(x, width);
            y = ClampCoord(y, height);
            return data[y * width + x];
        case BorderMode::Wrap:
            x = ((x % width) + width) % width;
            y = ((y % height) + height) % height;
            return data[y * width + x];
        default:
            return borderValue;
    }
}

// Bilinear interpolation helper
template<typename T>
double BilinearSample(const T* data, int32_t width, int32_t height,
                      double x, double y, BorderMode mode, double borderValue) {
    int32_t x0 = static_cast<int32_t>(std::floor(x));
    int32_t y0 = static_cast<int32_t>(std::floor(y));
    int32_t x1 = x0 + 1;
    int32_t y1 = y0 + 1;

    double fx = x - x0;
    double fy = y - y0;

    T bv = static_cast<T>(borderValue);
    double v00 = GetPixelWithBorder(data, width, height, x0, y0, mode, bv);
    double v10 = GetPixelWithBorder(data, width, height, x1, y0, mode, bv);
    double v01 = GetPixelWithBorder(data, width, height, x0, y1, mode, bv);
    double v11 = GetPixelWithBorder(data, width, height, x1, y1, mode, bv);

    return v00 * (1 - fx) * (1 - fy) +
           v10 * fx * (1 - fy) +
           v01 * (1 - fx) * fy +
           v11 * fx * fy;
}

// Bicubic kernel
inline double CubicKernel(double x) {
    x = std::abs(x);
    if (x < 1.0) {
        return (1.5 * x - 2.5) * x * x + 1.0;
    } else if (x < 2.0) {
        return ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0;
    }
    return 0.0;
}

// Bicubic interpolation helper
template<typename T>
double BicubicSample(const T* data, int32_t width, int32_t height,
                     double x, double y, BorderMode mode, double borderValue) {
    int32_t x0 = static_cast<int32_t>(std::floor(x)) - 1;
    int32_t y0 = static_cast<int32_t>(std::floor(y)) - 1;

    double fx = x - std::floor(x);
    double fy = y - std::floor(y);

    T bv = static_cast<T>(borderValue);
    double result = 0.0;

    for (int j = 0; j < 4; ++j) {
        double wy = CubicKernel(fy - (j - 1));
        for (int i = 0; i < 4; ++i) {
            double wx = CubicKernel(fx - (i - 1));
            double v = GetPixelWithBorder(data, width, height, x0 + i, y0 + j, mode, bv);
            result += v * wx * wy;
        }
    }

    return result;
}

// Nearest neighbor helper
template<typename T>
double NearestSample(const T* data, int32_t width, int32_t height,
                     double x, double y, BorderMode mode, double borderValue) {
    int32_t xi = static_cast<int32_t>(std::round(x));
    int32_t yi = static_cast<int32_t>(std::round(y));
    T bv = static_cast<T>(borderValue);
    return GetPixelWithBorder(data, width, height, xi, yi, mode, bv);
}

// Generic perspective warp function
template<typename T>
void WarpPerspectiveGeneric(const T* src, int32_t srcWidth, int32_t srcHeight,
                            T* dst, int32_t dstWidth, int32_t dstHeight,
                            const Homography& invH,
                            InterpolationMethod method,
                            BorderMode borderMode, double borderValue) {
    for (int32_t y = 0; y < dstHeight; ++y) {
        for (int32_t x = 0; x < dstWidth; ++x) {
            // Map destination to source using inverse homography
            Point2d srcPt = invH.Transform(static_cast<double>(x),
                                           static_cast<double>(y));

            double value;
            switch (method) {
                case InterpolationMethod::Nearest:
                    value = NearestSample(src, srcWidth, srcHeight,
                                         srcPt.x, srcPt.y, borderMode, borderValue);
                    break;
                case InterpolationMethod::Bicubic:
                    value = BicubicSample(src, srcWidth, srcHeight,
                                         srcPt.x, srcPt.y, borderMode, borderValue);
                    break;
                case InterpolationMethod::Bilinear:
                default:
                    value = BilinearSample(src, srcWidth, srcHeight,
                                          srcPt.x, srcPt.y, borderMode, borderValue);
                    break;
            }

            // Clamp and store
            if constexpr (std::is_integral_v<T>) {
                value = std::round(value);
                value = std::max(static_cast<double>(std::numeric_limits<T>::min()),
                                std::min(static_cast<double>(std::numeric_limits<T>::max()), value));
            }
            dst[y * dstWidth + x] = static_cast<T>(value);
        }
    }
}

// Normalize points for DLT (improves numerical stability)
void NormalizePoints(const std::vector<Point2d>& points,
                     std::vector<Point2d>& normalized,
                     Mat33& T) {
    // Compute centroid
    double cx = 0, cy = 0;
    for (const auto& p : points) {
        cx += p.x;
        cy += p.y;
    }
    cx /= points.size();
    cy /= points.size();

    // Compute mean distance from centroid
    double meanDist = 0;
    for (const auto& p : points) {
        double dx = p.x - cx;
        double dy = p.y - cy;
        meanDist += std::sqrt(dx * dx + dy * dy);
    }
    meanDist /= points.size();

    // Scale factor to make mean distance = sqrt(2)
    double scale = (meanDist > EPSILON) ? std::sqrt(2.0) / meanDist : 1.0;

    // Build normalization matrix
    T = Mat33::Identity();
    T(0, 0) = scale;
    T(0, 2) = -scale * cx;
    T(1, 1) = scale;
    T(1, 2) = -scale * cy;

    // Apply normalization
    normalized.resize(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        normalized[i].x = scale * (points[i].x - cx);
        normalized[i].y = scale * (points[i].y - cy);
    }
}

// Count inliers for RANSAC
int32_t CountInliers(const std::vector<Point2d>& srcPoints,
                     const std::vector<Point2d>& dstPoints,
                     const Homography& H,
                     double threshold,
                     std::vector<bool>* mask = nullptr) {
    int32_t count = 0;
    double thresholdSq = threshold * threshold;

    if (mask) {
        mask->resize(srcPoints.size());
    }

    for (size_t i = 0; i < srcPoints.size(); ++i) {
        Point2d transformed = H.Transform(srcPoints[i]);
        double dx = transformed.x - dstPoints[i].x;
        double dy = transformed.y - dstPoints[i].y;
        double distSq = dx * dx + dy * dy;

        bool isInlier = distSq <= thresholdSq;
        if (isInlier) count++;
        if (mask) (*mask)[i] = isInlier;
    }

    return count;
}

// Calculate required RANSAC iterations
int32_t CalculateRANSACIterations(int32_t sampleSize, double confidence,
                                   double inlierRatio, int32_t maxIterations) {
    if (inlierRatio >= 1.0) return 1;
    if (inlierRatio <= 0.0) return maxIterations;

    double num = std::log(1.0 - confidence);
    double denom = std::log(1.0 - std::pow(inlierRatio, sampleSize));

    if (std::abs(denom) < EPSILON) return maxIterations;

    int32_t iterations = static_cast<int32_t>(std::ceil(num / denom));
    return std::min(iterations, maxIterations);
}

} // anonymous namespace

// =============================================================================
// Homography Class Implementation
// =============================================================================

Homography::Homography() {
    std::fill(data_, data_ + 9, 0.0);
    data_[0] = data_[4] = data_[8] = 1.0;  // Identity
}

Homography::Homography(double h00, double h01, double h02,
                       double h10, double h11, double h12,
                       double h20, double h21, double h22) {
    data_[0] = h00; data_[1] = h01; data_[2] = h02;
    data_[3] = h10; data_[4] = h11; data_[5] = h12;
    data_[6] = h20; data_[7] = h21; data_[8] = h22;
}

Homography::Homography(const Mat33& mat) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            data_[i * 3 + j] = mat(i, j);
        }
    }
}

Homography::Homography(const double* data) {
    std::copy(data, data + 9, data_);
}

Mat33 Homography::ToMat33() const {
    Mat33 result;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result(i, j) = data_[i * 3 + j];
        }
    }
    return result;
}

Homography Homography::Identity() {
    return Homography();
}

Homography Homography::FromAffine(const QMatrix& affine) {
    return Homography(
        affine.M00(), affine.M01(), affine.M02(),
        affine.M10(), affine.M11(), affine.M12(),
        0.0, 0.0, 1.0
    );
}

std::optional<Homography> Homography::From4Points(
    const std::array<Point2d, 4>& srcPoints,
    const std::array<Point2d, 4>& dstPoints) {

    std::vector<Point2d> src(srcPoints.begin(), srcPoints.end());
    std::vector<Point2d> dst(dstPoints.begin(), dstPoints.end());

    return EstimateHomography(src, dst);
}

Point2d Homography::Transform(const Point2d& p) const {
    return Transform(p.x, p.y);
}

Point2d Homography::Transform(double x, double y) const {
    double w = data_[6] * x + data_[7] * y + data_[8];

    if (std::abs(w) < EPSILON) {
        // Point at infinity
        return {std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity()};
    }

    double invW = 1.0 / w;
    return {
        (data_[0] * x + data_[1] * y + data_[2]) * invW,
        (data_[3] * x + data_[4] * y + data_[5]) * invW
    };
}

std::vector<Point2d> Homography::Transform(const std::vector<Point2d>& points) const {
    std::vector<Point2d> result;
    result.reserve(points.size());
    for (const auto& p : points) {
        result.push_back(Transform(p));
    }
    return result;
}

double Homography::Determinant() const {
    return data_[0] * (data_[4] * data_[8] - data_[5] * data_[7])
         - data_[1] * (data_[3] * data_[8] - data_[5] * data_[6])
         + data_[2] * (data_[3] * data_[7] - data_[4] * data_[6]);
}

bool Homography::IsInvertible(double tolerance) const {
    return std::abs(Determinant()) > tolerance;
}

Homography Homography::Inverse() const {
    double det = Determinant();
    if (std::abs(det) < EPSILON) {
        return Homography();  // Return identity for singular matrix
    }

    double invDet = 1.0 / det;

    // Adjugate matrix elements
    double a00 = data_[4] * data_[8] - data_[5] * data_[7];
    double a01 = data_[2] * data_[7] - data_[1] * data_[8];
    double a02 = data_[1] * data_[5] - data_[2] * data_[4];

    double a10 = data_[5] * data_[6] - data_[3] * data_[8];
    double a11 = data_[0] * data_[8] - data_[2] * data_[6];
    double a12 = data_[2] * data_[3] - data_[0] * data_[5];

    double a20 = data_[3] * data_[7] - data_[4] * data_[6];
    double a21 = data_[1] * data_[6] - data_[0] * data_[7];
    double a22 = data_[0] * data_[4] - data_[1] * data_[3];

    return Homography(
        a00 * invDet, a01 * invDet, a02 * invDet,
        a10 * invDet, a11 * invDet, a12 * invDet,
        a20 * invDet, a21 * invDet, a22 * invDet
    );
}

Homography Homography::Normalized() const {
    if (std::abs(data_[8]) < EPSILON) {
        return *this;
    }

    double invH22 = 1.0 / data_[8];
    Homography result;
    for (int i = 0; i < 9; ++i) {
        result.data_[i] = data_[i] * invH22;
    }
    return result;
}

Homography Homography::operator*(const Homography& other) const {
    Homography result;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result.data_[i * 3 + j] = 0;
            for (int k = 0; k < 3; ++k) {
                result.data_[i * 3 + j] += data_[i * 3 + k] * other.data_[k * 3 + j];
            }
        }
    }
    return result;
}

bool Homography::IsAffine(double tolerance) const {
    return std::abs(data_[6]) < tolerance && std::abs(data_[7]) < tolerance;
}

std::optional<QMatrix> Homography::ToAffine(double tolerance) const {
    if (!IsAffine(tolerance)) {
        return std::nullopt;
    }

    // Normalize by h22
    double scale = (std::abs(data_[8]) > EPSILON) ? 1.0 / data_[8] : 1.0;

    return QMatrix(
        data_[0] * scale, data_[1] * scale, data_[2] * scale,
        data_[3] * scale, data_[4] * scale, data_[5] * scale
    );
}

// =============================================================================
// Homography Estimation
// =============================================================================

std::optional<Homography> EstimateHomography(
    const std::vector<Point2d>& srcPoints,
    const std::vector<Point2d>& dstPoints) {

    if (srcPoints.size() < 4 || srcPoints.size() != dstPoints.size()) {
        return std::nullopt;
    }

    size_t n = srcPoints.size();

    // Normalize points for numerical stability
    std::vector<Point2d> srcNorm, dstNorm;
    Mat33 Tsrc, Tdst;
    NormalizePoints(srcPoints, srcNorm, Tsrc);
    NormalizePoints(dstPoints, dstNorm, Tdst);

    // Build DLT matrix A (2n x 9)
    // For each correspondence: two equations
    // [-x -y -1  0  0  0  x*x' y*x' x'] * h = 0
    // [ 0  0  0 -x -y -1  x*y' y*y' y'] * h = 0

    MatX A(2 * static_cast<int>(n), 9);

    for (size_t i = 0; i < n; ++i) {
        double x = srcNorm[i].x;
        double y = srcNorm[i].y;
        double xp = dstNorm[i].x;
        double yp = dstNorm[i].y;

        int row = 2 * static_cast<int>(i);

        // First equation
        A(row, 0) = -x;
        A(row, 1) = -y;
        A(row, 2) = -1;
        A(row, 3) = 0;
        A(row, 4) = 0;
        A(row, 5) = 0;
        A(row, 6) = x * xp;
        A(row, 7) = y * xp;
        A(row, 8) = xp;

        // Second equation
        A(row + 1, 0) = 0;
        A(row + 1, 1) = 0;
        A(row + 1, 2) = 0;
        A(row + 1, 3) = -x;
        A(row + 1, 4) = -y;
        A(row + 1, 5) = -1;
        A(row + 1, 6) = x * yp;
        A(row + 1, 7) = y * yp;
        A(row + 1, 8) = yp;
    }

    // Solve using SVD (find null space of A)
    VecX h = SolveHomogeneous(A);
    if (h.Size() != 9) {
        return std::nullopt;
    }

    // Reconstruct normalized homography
    Mat33 Hnorm;
    Hnorm(0, 0) = h[0]; Hnorm(0, 1) = h[1]; Hnorm(0, 2) = h[2];
    Hnorm(1, 0) = h[3]; Hnorm(1, 1) = h[4]; Hnorm(1, 2) = h[5];
    Hnorm(2, 0) = h[6]; Hnorm(2, 1) = h[7]; Hnorm(2, 2) = h[8];

    // Denormalize: H = Tdst^-1 * Hnorm * Tsrc
    // Tdst = [s 0 -s*cx; 0 s -s*cy; 0 0 1]
    // TdstInv = [1/s 0 cx; 0 1/s cy; 0 0 1]
    Mat33 TdstInv = Mat33::Identity();  // Initialize to identity first
    double scale = Tdst(0, 0);
    if (std::abs(scale) > EPSILON) {
        double invScale = 1.0 / scale;
        TdstInv(0, 0) = invScale;
        TdstInv(1, 1) = invScale;
        // Tdst(0, 2) = -scale * cx, so cx = -Tdst(0, 2) / scale
        TdstInv(0, 2) = -Tdst(0, 2) * invScale;
        TdstInv(1, 2) = -Tdst(1, 2) * invScale;
    }

    Mat33 H = TdstInv * Hnorm * Tsrc;

    // Normalize so H(2,2) = 1
    Homography result(H);
    return result.Normalized();
}

std::optional<Homography> EstimateHomographyRANSAC(
    const std::vector<Point2d>& srcPoints,
    const std::vector<Point2d>& dstPoints,
    double threshold,
    double confidence,
    int32_t maxIterations,
    std::vector<bool>* inlierMask) {

    if (srcPoints.size() < 4 || srcPoints.size() != dstPoints.size()) {
        return std::nullopt;
    }

    size_t n = srcPoints.size();
    auto& rng = Platform::Random::Instance();

    Homography bestH;
    int32_t bestInliers = 0;
    std::vector<bool> bestMask;

    int32_t iterations = maxIterations;
    const int32_t sampleSize = 4;

    for (int32_t iter = 0; iter < iterations; ++iter) {
        // Random sample of 4 points
        std::vector<size_t> indices(n);
        for (size_t i = 0; i < n; ++i) indices[i] = i;

        // Fisher-Yates shuffle for first 4
        for (int i = 0; i < sampleSize; ++i) {
            size_t j = i + rng.Index(n - i);
            std::swap(indices[i], indices[j]);
        }

        // Extract sample points
        std::vector<Point2d> sampleSrc(sampleSize), sampleDst(sampleSize);
        for (int i = 0; i < sampleSize; ++i) {
            sampleSrc[i] = srcPoints[indices[i]];
            sampleDst[i] = dstPoints[indices[i]];
        }

        // Estimate from minimal sample
        auto H = EstimateHomography(sampleSrc, sampleDst);
        if (!H) continue;

        // Count inliers
        std::vector<bool> mask;
        int32_t numInliers = CountInliers(srcPoints, dstPoints, *H, threshold, &mask);

        if (numInliers > bestInliers) {
            bestInliers = numInliers;
            bestH = *H;
            bestMask = std::move(mask);

            // Update iteration count
            double inlierRatio = static_cast<double>(numInliers) / n;
            iterations = CalculateRANSACIterations(sampleSize, confidence,
                                                    inlierRatio, maxIterations);
        }
    }

    if (bestInliers < 4) {
        return std::nullopt;
    }

    // Refine with all inliers
    std::vector<Point2d> inlierSrc, inlierDst;
    for (size_t i = 0; i < n; ++i) {
        if (bestMask[i]) {
            inlierSrc.push_back(srcPoints[i]);
            inlierDst.push_back(dstPoints[i]);
        }
    }

    auto refinedH = EstimateHomography(inlierSrc, inlierDst);
    if (refinedH) {
        bestH = *refinedH;
        CountInliers(srcPoints, dstPoints, bestH, threshold, &bestMask);
    }

    if (inlierMask) {
        *inlierMask = std::move(bestMask);
    }

    return bestH;
}

double ComputeHomographyError(
    const std::vector<Point2d>& srcPoints,
    const std::vector<Point2d>& dstPoints,
    const Homography& H) {

    if (srcPoints.empty() || srcPoints.size() != dstPoints.size()) {
        return std::numeric_limits<double>::max();
    }

    double sumSqError = 0;
    for (size_t i = 0; i < srcPoints.size(); ++i) {
        Point2d transformed = H.Transform(srcPoints[i]);
        double dx = transformed.x - dstPoints[i].x;
        double dy = transformed.y - dstPoints[i].y;
        sumSqError += dx * dx + dy * dy;
    }

    return std::sqrt(sumSqError / srcPoints.size());
}

std::vector<double> ComputePointHomographyErrors(
    const std::vector<Point2d>& srcPoints,
    const std::vector<Point2d>& dstPoints,
    const Homography& H) {

    std::vector<double> errors;
    if (srcPoints.size() != dstPoints.size()) {
        return errors;
    }

    errors.reserve(srcPoints.size());
    for (size_t i = 0; i < srcPoints.size(); ++i) {
        Point2d transformed = H.Transform(srcPoints[i]);
        double dx = transformed.x - dstPoints[i].x;
        double dy = transformed.y - dstPoints[i].y;
        errors.push_back(std::sqrt(dx * dx + dy * dy));
    }

    return errors;
}

// =============================================================================
// Image Warping
// =============================================================================

void ComputePerspectiveOutputSize(int32_t srcWidth, int32_t srcHeight,
                                   const Homography& H,
                                   int32_t& dstWidth, int32_t& dstHeight,
                                   double& offsetX, double& offsetY) {
    // Transform corners
    Point2d corners[4] = {
        H.Transform(0, 0),
        H.Transform(srcWidth - 1, 0),
        H.Transform(srcWidth - 1, srcHeight - 1),
        H.Transform(0, srcHeight - 1)
    };

    // Find bounding box
    double minX = corners[0].x, maxX = corners[0].x;
    double minY = corners[0].y, maxY = corners[0].y;

    for (int i = 1; i < 4; ++i) {
        if (std::isfinite(corners[i].x) && std::isfinite(corners[i].y)) {
            minX = std::min(minX, corners[i].x);
            maxX = std::max(maxX, corners[i].x);
            minY = std::min(minY, corners[i].y);
            maxY = std::max(maxY, corners[i].y);
        }
    }

    dstWidth = static_cast<int32_t>(std::ceil(maxX - minX)) + 1;
    dstHeight = static_cast<int32_t>(std::ceil(maxY - minY)) + 1;
    offsetX = -minX;
    offsetY = -minY;
}

QImage WarpPerspective(const QImage& src,
                       const Homography& H,
                       int32_t dstWidth,
                       int32_t dstHeight,
                       InterpolationMethod method,
                       BorderMode borderMode,
                       double borderValue) {
    if (src.Empty()) return QImage();

    // Calculate output size if not specified
    double offsetX = 0, offsetY = 0;
    if (dstWidth <= 0 || dstHeight <= 0) {
        ComputePerspectiveOutputSize(src.Width(), src.Height(), H,
                                     dstWidth, dstHeight, offsetX, offsetY);
    }

    // Create adjusted homography with offset
    Homography offsetH(1, 0, offsetX, 0, 1, offsetY, 0, 0, 1);
    Homography adjustedH = offsetH * H;

    // Compute inverse for backward mapping
    if (!adjustedH.IsInvertible()) {
        return QImage();
    }
    Homography invH = adjustedH.Inverse();

    // Get channel type
    ChannelType chType = src.GetChannelType();

    // Create output image
    QImage dst(dstWidth, dstHeight, src.Type(), chType);

    int32_t srcW = src.Width();
    int32_t srcH = src.Height();
    int32_t channels = src.Channels();

    // Process each channel
    for (int32_t c = 0; c < channels; ++c) {
        switch (src.Type()) {
            case PixelType::UInt8: {
                const uint8_t* srcBase = static_cast<const uint8_t*>(src.Data());
                uint8_t* dstBase = static_cast<uint8_t*>(dst.Data());
                const uint8_t* srcData = srcBase + c * srcW * srcH;
                uint8_t* dstData = dstBase + c * dstWidth * dstHeight;
                WarpPerspectiveGeneric(srcData, srcW, srcH, dstData, dstWidth, dstHeight,
                                       invH, method, borderMode, borderValue);
                break;
            }
            case PixelType::UInt16: {
                const uint16_t* srcBase = static_cast<const uint16_t*>(src.Data());
                uint16_t* dstBase = static_cast<uint16_t*>(dst.Data());
                const uint16_t* srcData = srcBase + c * srcW * srcH;
                uint16_t* dstData = dstBase + c * dstWidth * dstHeight;
                WarpPerspectiveGeneric(srcData, srcW, srcH, dstData, dstWidth, dstHeight,
                                       invH, method, borderMode, borderValue);
                break;
            }
            case PixelType::Int16: {
                const int16_t* srcBase = static_cast<const int16_t*>(src.Data());
                int16_t* dstBase = static_cast<int16_t*>(dst.Data());
                const int16_t* srcData = srcBase + c * srcW * srcH;
                int16_t* dstData = dstBase + c * dstWidth * dstHeight;
                WarpPerspectiveGeneric(srcData, srcW, srcH, dstData, dstWidth, dstHeight,
                                       invH, method, borderMode, borderValue);
                break;
            }
            case PixelType::Float32: {
                const float* srcBase = static_cast<const float*>(src.Data());
                float* dstBase = static_cast<float*>(dst.Data());
                const float* srcData = srcBase + c * srcW * srcH;
                float* dstData = dstBase + c * dstWidth * dstHeight;
                WarpPerspectiveGeneric(srcData, srcW, srcH, dstData, dstWidth, dstHeight,
                                       invH, method, borderMode, borderValue);
                break;
            }
        }
    }

    return dst;
}

// =============================================================================
// Contour Transformation
// =============================================================================

QContour PerspectiveTransformContour(const QContour& contour, const Homography& H) {
    if (contour.Empty()) return QContour();

    auto points = contour.GetPoints();
    std::vector<Point2d> transformed;
    transformed.reserve(points.size());

    for (const auto& p : points) {
        transformed.push_back(H.Transform(p));
    }

    return QContour(transformed, contour.IsClosed());
}

std::vector<QContour> PerspectiveTransformContours(
    const std::vector<QContour>& contours,
    const Homography& H) {

    std::vector<QContour> result;
    result.reserve(contours.size());

    for (const auto& contour : contours) {
        result.push_back(PerspectiveTransformContour(contour, H));
    }

    return result;
}

// =============================================================================
// Homography Decomposition
// =============================================================================

std::vector<HomographyDecomposition> DecomposeHomography(const Homography& H) {
    (void)H;
    // Note: This is a simplified decomposition that assumes H is between
    // normalized image coordinates. A full implementation would use SVD.

    std::vector<HomographyDecomposition> results;

    // For now, return empty - full implementation requires SVD and more math
    // This would be a complex implementation following Faugeras or Ma et al.

    return results;
}

std::vector<HomographyDecomposition> FilterDecompositionsByVisibility(
    const std::vector<HomographyDecomposition>& decompositions,
    const std::vector<Point2d>& srcPoints) {
    (void)srcPoints;

    std::vector<HomographyDecomposition> valid;

    for (const auto& d : decompositions) {
        if (d.valid) {
            valid.push_back(d);
        }
    }

    return valid;
}

// =============================================================================
// Utility Functions
// =============================================================================

std::optional<Homography> RectifyQuadrilateral(
    const std::array<Point2d, 4>& quad,
    double width,
    double height) {

    std::array<Point2d, 4> rect = {{
        {0, 0},
        {width, 0},
        {width, height},
        {0, height}
    }};

    return Homography::From4Points(quad, rect);
}

std::optional<Homography> RectangleToQuadrilateral(
    double width,
    double height,
    const std::array<Point2d, 4>& quad) {

    std::array<Point2d, 4> rect = {{
        {0, 0},
        {width, 0},
        {width, height},
        {0, height}
    }};

    return Homography::From4Points(rect, quad);
}

Rect2d TransformBoundingBoxPerspective(const Rect2d& bbox, const Homography& H) {
    Point2d corners[4] = {
        H.Transform(bbox.x, bbox.y),
        H.Transform(bbox.x + bbox.width, bbox.y),
        H.Transform(bbox.x + bbox.width, bbox.y + bbox.height),
        H.Transform(bbox.x, bbox.y + bbox.height)
    };

    double minX = corners[0].x, maxX = corners[0].x;
    double minY = corners[0].y, maxY = corners[0].y;

    for (int i = 1; i < 4; ++i) {
        if (std::isfinite(corners[i].x) && std::isfinite(corners[i].y)) {
            minX = std::min(minX, corners[i].x);
            maxX = std::max(maxX, corners[i].x);
            minY = std::min(minY, corners[i].y);
            maxY = std::max(maxY, corners[i].y);
        }
    }

    return Rect2d(minX, minY, maxX - minX, maxY - minY);
}

bool IsValidHomography(const Homography& H, int32_t srcWidth, int32_t srcHeight) {
    // Check invertibility
    if (!H.IsInvertible()) {
        return false;
    }

    // Transform corners
    Point2d corners[4] = {
        H.Transform(0, 0),
        H.Transform(srcWidth - 1, 0),
        H.Transform(srcWidth - 1, srcHeight - 1),
        H.Transform(0, srcHeight - 1)
    };

    // Check for infinite points
    for (int i = 0; i < 4; ++i) {
        if (!std::isfinite(corners[i].x) || !std::isfinite(corners[i].y)) {
            return false;
        }
    }

    // Check orientation (cross product should have same sign for convex quad)
    auto cross = [](const Point2d& a, const Point2d& b, const Point2d& c) {
        return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    };

    double sign = 0;
    for (int i = 0; i < 4; ++i) {
        double c = cross(corners[i], corners[(i + 1) % 4], corners[(i + 2) % 4]);
        if (i == 0) {
            sign = c;
        } else if (sign * c < 0) {
            return false;  // Quad is self-intersecting
        }
    }

    return true;
}

double SampsonError(const Point2d& src, const Point2d& dst, const Homography& H) {
    // Compute Sampson error (first-order approximation to geometric error)
    double x = src.x, y = src.y;
    double xp = dst.x, yp = dst.y;

    const double* h = H.Data();

    // H * p
    double w = h[6] * x + h[7] * y + h[8];
    double Hx = h[0] * x + h[1] * y + h[2];
    double Hy = h[3] * x + h[4] * y + h[5];

    // Error
    double ex = Hx / w - xp;
    double ey = Hy / w - yp;

    // Jacobian derivatives
    double invW2 = 1.0 / (w * w);
    double J11 = (h[0] * w - h[6] * Hx) * invW2;
    double J12 = (h[1] * w - h[7] * Hx) * invW2;
    double J21 = (h[3] * w - h[6] * Hy) * invW2;
    double J22 = (h[4] * w - h[7] * Hy) * invW2;

    // Sampson error = e^T * (J * J^T)^-1 * e
    double a = J11 * J11 + J12 * J12 + 1;  // +1 for x' derivative
    double b = J11 * J21 + J12 * J22;
    double c = J21 * J21 + J22 * J22 + 1;  // +1 for y' derivative

    double det = a * c - b * b;
    if (std::abs(det) < EPSILON) {
        return ex * ex + ey * ey;  // Fall back to algebraic error
    }

    return (c * ex * ex - 2 * b * ex * ey + a * ey * ey) / det;
}

Homography RefineHomographyLM(
    const std::vector<Point2d>& srcPoints,
    const std::vector<Point2d>& dstPoints,
    const Homography& H,
    int32_t maxIterations) {

    // Simple Gauss-Newton refinement
    // For a full LM implementation, we would need more infrastructure

    if (srcPoints.size() < 4 || srcPoints.size() != dstPoints.size()) {
        return H;
    }

    Homography current = H.Normalized();
    size_t n = srcPoints.size();

    for (int32_t iter = 0; iter < maxIterations; ++iter) {
        // Build Jacobian and residual
        MatX J(2 * static_cast<int>(n), 8);  // 8 parameters (h22 = 1)
        VecX r(2 * static_cast<int>(n));

        for (size_t i = 0; i < n; ++i) {
            double x = srcPoints[i].x;
            double y = srcPoints[i].y;
            double xp = dstPoints[i].x;
            double yp = dstPoints[i].y;

            const double* h = current.Data();
            double w = h[6] * x + h[7] * y + h[8];
            double invW = 1.0 / w;

            double Hx = h[0] * x + h[1] * y + h[2];
            double Hy = h[3] * x + h[4] * y + h[5];

            double projX = Hx * invW;
            double projY = Hy * invW;

            int row = 2 * static_cast<int>(i);

            // Residuals
            r[row] = projX - xp;
            r[row + 1] = projY - yp;

            // Jacobian for x equation
            J(row, 0) = x * invW;
            J(row, 1) = y * invW;
            J(row, 2) = invW;
            J(row, 3) = 0;
            J(row, 4) = 0;
            J(row, 5) = 0;
            J(row, 6) = -x * projX * invW;
            J(row, 7) = -y * projX * invW;

            // Jacobian for y equation
            J(row + 1, 0) = 0;
            J(row + 1, 1) = 0;
            J(row + 1, 2) = 0;
            J(row + 1, 3) = x * invW;
            J(row + 1, 4) = y * invW;
            J(row + 1, 5) = invW;
            J(row + 1, 6) = -x * projY * invW;
            J(row + 1, 7) = -y * projY * invW;
        }

        // Solve normal equations: (J^T * J) * delta = -J^T * r
        MatX JtJ = J.Transpose() * J;
        VecX Jtr = J.Transpose() * r;

        VecX delta = SolveLU(JtJ, Jtr * (-1.0));
        if (delta.Size() != 8) {
            break;
        }

        // Update parameters
        current = Homography(
            current(0, 0) + delta[0], current(0, 1) + delta[1], current(0, 2) + delta[2],
            current(1, 0) + delta[3], current(1, 1) + delta[4], current(1, 2) + delta[5],
            current(2, 0) + delta[6], current(2, 1) + delta[7], current(2, 2)
        );

        // Normalize
        current = current.Normalized();

        // Check convergence
        double deltaSum = 0;
        for (int i = 0; i < 8; ++i) {
            deltaSum += std::abs(delta[i]);
        }
        if (deltaSum < 1e-10) {
            break;
        }
    }

    return current;
}

} // namespace Qi::Vision::Internal
