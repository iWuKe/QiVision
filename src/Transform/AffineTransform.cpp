/**
 * @file AffineTransform.cpp
 * @brief Implementation of public affine transformation API
 */

#include <QiVision/Transform/AffineTransform.h>
#include <QiVision/Internal/AffineTransform.h>
#include <QiVision/Internal/Interpolate.h>
#include <QiVision/Core/Exception.h>
#include <QiVision/Core/Validate.h>

#include <algorithm>
#include <cctype>
#include <cmath>

namespace Qi::Vision::Transform {

namespace {

// Transform-specific: sets dst to empty on empty src, requires UInt8
inline bool RequireImageU8(const QImage& src, QImage& dst, const char* funcName) {
    if (!Validate::RequireImageU8(src, funcName)) {
        dst = QImage();
        return false;
    }
    return true;
}

inline void RequireFinite(double value, const char* name, const char* funcName) {
    if (!std::isfinite(value)) {
        throw InvalidArgumentException(std::string(funcName) + ": " + name + " is invalid");
    }
}

inline void RequirePositiveFinite(double value, const char* name, const char* funcName) {
    RequireFinite(value, name, funcName);
    Validate::RequirePositive(value, name, funcName);
}

void RequireMatrixFinite(const QMatrix& matrix, const char* funcName) {
    double elements[6];
    matrix.GetElements(elements);
    for (double v : elements) {
        if (!std::isfinite(v)) {
            throw InvalidArgumentException(std::string(funcName) + ": invalid matrix");
        }
    }
}

void RequirePointValid(const Point2d& point, const char* funcName) {
    if (!point.IsValid()) {
        throw InvalidArgumentException(std::string(funcName) + ": invalid point");
    }
}

// Convert string to interpolation method
Internal::InterpolationMethod ToInternalInterp(const std::string& interp) {
    std::string lower = interp;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (lower == "nearest") {
        return Internal::InterpolationMethod::Nearest;
    } else if (lower == "bicubic") {
        return Internal::InterpolationMethod::Bicubic;
    }
    if (lower.empty() || lower == "bilinear") {
        return Internal::InterpolationMethod::Bilinear;
    }
    throw InvalidArgumentException("Unknown interpolation: " + interp);
}

// Convert string to border mode
Internal::BorderMode ToInternalBorderMode(const std::string& mode) {
    std::string lower = mode;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (lower == "replicate") {
        return Internal::BorderMode::Replicate;
    } else if (lower == "reflect" || lower == "reflect101") {
        return Internal::BorderMode::Reflect101;
    } else if (lower == "wrap") {
        return Internal::BorderMode::Wrap;
    }
    if (lower.empty() || lower == "constant") {
        return Internal::BorderMode::Constant;
    }
    throw InvalidArgumentException("Unknown border mode: " + mode);
}

} // anonymous namespace

// =============================================================================
// Image Transformation
// =============================================================================

void AffineTransImage(
    const QImage& src,
    QImage& dst,
    const QMatrix& matrix,
    const std::string& interpolation,
    const std::string& borderMode,
    double borderValue)
{
    if (!RequireImageU8(src, dst, "AffineTransImage")) {
        return;
    }
    RequireMatrixFinite(matrix, "AffineTransImage");
    RequireFinite(borderValue, "borderValue", "AffineTransImage");
    dst = Internal::WarpAffine(
        src,
        matrix,
        0, 0,  // Auto-calculate size
        ToInternalInterp(interpolation),
        ToInternalBorderMode(borderMode),
        borderValue
    );
}

void AffineTransImage(
    const QImage& src,
    QImage& dst,
    const QMatrix& matrix,
    int32_t dstWidth,
    int32_t dstHeight,
    const std::string& interpolation,
    const std::string& borderMode,
    double borderValue)
{
    if (!RequireImageU8(src, dst, "AffineTransImage")) {
        return;
    }
    RequireMatrixFinite(matrix, "AffineTransImage");
    Validate::RequireNonNegative(dstWidth, "dstWidth", "AffineTransImage");
    Validate::RequireNonNegative(dstHeight, "dstHeight", "AffineTransImage");
    RequireFinite(borderValue, "borderValue", "AffineTransImage");
    dst = Internal::WarpAffine(
        src,
        matrix,
        dstWidth, dstHeight,
        ToInternalInterp(interpolation),
        ToInternalBorderMode(borderMode),
        borderValue
    );
}

void RotateImage(
    const QImage& src,
    QImage& dst,
    double angle,
    const std::string& interpolation)
{
    if (!RequireImageU8(src, dst, "RotateImage")) {
        return;
    }
    RequireFinite(angle, "angle", "RotateImage");
    dst = Internal::RotateImage(
        src,
        angle,
        true,  // resize to fit
        ToInternalInterp(interpolation),
        Internal::BorderMode::Constant,
        0.0
    );
}

void RotateImage(
    const QImage& src,
    QImage& dst,
    double angle,
    double centerRow,
    double centerCol,
    const std::string& interpolation)
{
    if (!RequireImageU8(src, dst, "RotateImage")) {
        return;
    }
    RequireFinite(angle, "angle", "RotateImage");
    RequireFinite(centerRow, "centerRow", "RotateImage");
    RequireFinite(centerCol, "centerCol", "RotateImage");
    dst = Internal::RotateImage(
        src,
        angle,
        centerCol,  // Internal uses x, y order
        centerRow,
        true,  // resize to fit
        ToInternalInterp(interpolation),
        Internal::BorderMode::Constant,
        0.0
    );
}

void ScaleImage(
    const QImage& src,
    QImage& dst,
    double scaleX,
    double scaleY,
    const std::string& interpolation)
{
    if (!RequireImageU8(src, dst, "ScaleImage")) {
        return;
    }
    RequirePositiveFinite(scaleX, "scaleX", "ScaleImage");
    RequirePositiveFinite(scaleY, "scaleY", "ScaleImage");
    dst = Internal::ScaleImageFactor(
        src,
        scaleX,
        scaleY,
        ToInternalInterp(interpolation)
    );
}

void ZoomImageSize(
    const QImage& src,
    QImage& dst,
    int32_t dstWidth,
    int32_t dstHeight,
    const std::string& interpolation)
{
    if (!RequireImageU8(src, dst, "ZoomImageSize")) {
        return;
    }
    Validate::RequireNonNegative(dstWidth, "dstWidth", "ZoomImageSize");
    Validate::RequireNonNegative(dstHeight, "dstHeight", "ZoomImageSize");
    dst = Internal::ScaleImage(
        src,
        dstWidth,
        dstHeight,
        ToInternalInterp(interpolation)
    );
}

// =============================================================================
// Matrix Creation
// =============================================================================

QMatrix HomMat2dIdentity() {
    return QMatrix::Identity();
}

QMatrix HomMat2dRotate(double phi, double cy, double cx) {
    RequireFinite(phi, "phi", "HomMat2dRotate");
    RequireFinite(cy, "cy", "HomMat2dRotate");
    RequireFinite(cx, "cx", "HomMat2dRotate");
    return QMatrix::Rotation(phi, cx, cy);
}

QMatrix HomMat2dScale(double sy, double sx, double cy, double cx) {
    RequireFinite(sy, "sy", "HomMat2dScale");
    RequireFinite(sx, "sx", "HomMat2dScale");
    RequireFinite(cy, "cy", "HomMat2dScale");
    RequireFinite(cx, "cx", "HomMat2dScale");
    return QMatrix::Scaling(sx, sy, Point2d{cx, cy});
}

QMatrix HomMat2dTranslate(const QMatrix& homMat2d, double ty, double tx) {
    RequireMatrixFinite(homMat2d, "HomMat2dTranslate");
    RequireFinite(ty, "ty", "HomMat2dTranslate");
    RequireFinite(tx, "tx", "HomMat2dTranslate");
    return QMatrix::Translation(tx, ty) * homMat2d;
}

QMatrix HomMat2dTranslateOnly(double ty, double tx) {
    RequireFinite(ty, "ty", "HomMat2dTranslateOnly");
    RequireFinite(tx, "tx", "HomMat2dTranslateOnly");
    return QMatrix::Translation(tx, ty);
}

QMatrix HomMat2dCompose(const QMatrix& homMat2d1, const QMatrix& homMat2d2) {
    RequireMatrixFinite(homMat2d1, "HomMat2dCompose");
    RequireMatrixFinite(homMat2d2, "HomMat2dCompose");
    return homMat2d1 * homMat2d2;
}

QMatrix HomMat2dInvert(const QMatrix& homMat2d) {
    RequireMatrixFinite(homMat2d, "HomMat2dInvert");
    return homMat2d.Inverse();
}

QMatrix HomMat2dRotateLocal(const QMatrix& homMat2d, double phi, double cy, double cx) {
    RequireMatrixFinite(homMat2d, "HomMat2dRotateLocal");
    RequireFinite(phi, "phi", "HomMat2dRotateLocal");
    RequireFinite(cy, "cy", "HomMat2dRotateLocal");
    RequireFinite(cx, "cx", "HomMat2dRotateLocal");
    QMatrix rotation = QMatrix::Rotation(phi, cx, cy);
    return rotation * homMat2d;
}

QMatrix HomMat2dScaleLocal(const QMatrix& homMat2d, double sy, double sx, double cy, double cx) {
    RequireMatrixFinite(homMat2d, "HomMat2dScaleLocal");
    RequireFinite(sy, "sy", "HomMat2dScaleLocal");
    RequireFinite(sx, "sx", "HomMat2dScaleLocal");
    RequireFinite(cy, "cy", "HomMat2dScaleLocal");
    RequireFinite(cx, "cx", "HomMat2dScaleLocal");
    QMatrix scaling = QMatrix::Scaling(sx, sy, Point2d{cx, cy});
    return scaling * homMat2d;
}

// =============================================================================
// Point Transformation
// =============================================================================

Point2d AffineTransPoint2d(const QMatrix& homMat2d, const Point2d& point) {
    RequireMatrixFinite(homMat2d, "AffineTransPoint2d");
    RequirePointValid(point, "AffineTransPoint2d");
    return homMat2d.Transform(point);
}

void AffineTransPoint2d(
    const QMatrix& homMat2d,
    double py, double px,
    double& qy, double& qx)
{
    RequireMatrixFinite(homMat2d, "AffineTransPoint2d");
    RequireFinite(py, "py", "AffineTransPoint2d");
    RequireFinite(px, "px", "AffineTransPoint2d");
    Point2d result = homMat2d.Transform(px, py);
    qx = result.x;
    qy = result.y;
}

std::vector<Point2d> AffineTransPoint2d(
    const QMatrix& homMat2d,
    const std::vector<Point2d>& points)
{
    std::vector<Point2d> result;
    result.reserve(points.size());
    RequireMatrixFinite(homMat2d, "AffineTransPoint2d");
    for (const auto& p : points) {
        if (!p.IsValid()) {
            throw InvalidArgumentException("AffineTransPoint2d: invalid point");
        }
        result.push_back(homMat2d.Transform(p));
    }
    return result;
}

void AffineTransPoint2d(
    const QMatrix& homMat2d,
    const std::vector<double>& py, const std::vector<double>& px,
    std::vector<double>& qy, std::vector<double>& qx)
{
    size_t n = std::min(py.size(), px.size());
    qy.resize(n);
    qx.resize(n);

    RequireMatrixFinite(homMat2d, "AffineTransPoint2d");
    for (size_t i = 0; i < n; ++i) {
        RequireFinite(py[i], "py", "AffineTransPoint2d");
        RequireFinite(px[i], "px", "AffineTransPoint2d");
        Point2d result = homMat2d.Transform(px[i], py[i]);
        qx[i] = result.x;
        qy[i] = result.y;
    }
}

// =============================================================================
// Transform Estimation
// =============================================================================

bool VectorToHomMat2d(
    const std::vector<Point2d>& srcPoints,
    const std::vector<Point2d>& dstPoints,
    QMatrix& homMat2d)
{
    for (const auto& p : srcPoints) {
        if (!p.IsValid()) {
            throw InvalidArgumentException("VectorToHomMat2d: invalid source point");
        }
    }
    for (const auto& p : dstPoints) {
        if (!p.IsValid()) {
            throw InvalidArgumentException("VectorToHomMat2d: invalid destination point");
        }
    }
    auto result = Internal::EstimateAffine(srcPoints, dstPoints);
    if (result) {
        homMat2d = *result;
        return true;
    }
    return false;
}

bool VectorToRigid(
    const std::vector<Point2d>& srcPoints,
    const std::vector<Point2d>& dstPoints,
    QMatrix& homMat2d)
{
    for (const auto& p : srcPoints) {
        if (!p.IsValid()) {
            throw InvalidArgumentException("VectorToRigid: invalid source point");
        }
    }
    for (const auto& p : dstPoints) {
        if (!p.IsValid()) {
            throw InvalidArgumentException("VectorToRigid: invalid destination point");
        }
    }
    auto result = Internal::EstimateRigid(srcPoints, dstPoints);
    if (result) {
        homMat2d = *result;
        return true;
    }
    return false;
}

bool VectorToSimilarity(
    const std::vector<Point2d>& srcPoints,
    const std::vector<Point2d>& dstPoints,
    QMatrix& homMat2d)
{
    for (const auto& p : srcPoints) {
        if (!p.IsValid()) {
            throw InvalidArgumentException("VectorToSimilarity: invalid source point");
        }
    }
    for (const auto& p : dstPoints) {
        if (!p.IsValid()) {
            throw InvalidArgumentException("VectorToSimilarity: invalid destination point");
        }
    }
    auto result = Internal::EstimateSimilarity(srcPoints, dstPoints);
    if (result) {
        homMat2d = *result;
        return true;
    }
    return false;
}

// =============================================================================
// Matrix Analysis
// =============================================================================

bool HomMat2dToAffinePar(
    const QMatrix& homMat2d,
    double& ty, double& tx,
    double& phi,
    double& sy, double& sx,
    double& theta)
{
    RequireMatrixFinite(homMat2d, "HomMat2dToAffinePar");
    double shear = 0.0;
    bool success = Internal::DecomposeAffine(homMat2d, tx, ty, phi, sx, sy, shear);
    // Convert shear to angle
    theta = std::atan(shear);
    return success;
}

bool HomMat2dIsRigid(const QMatrix& homMat2d, double tolerance) {
    RequireMatrixFinite(homMat2d, "HomMat2dIsRigid");
    RequireFinite(tolerance, "tolerance", "HomMat2dIsRigid");
    return Internal::IsRigidTransform(homMat2d, tolerance);
}

bool HomMat2dIsSimilarity(const QMatrix& homMat2d, double tolerance) {
    RequireMatrixFinite(homMat2d, "HomMat2dIsSimilarity");
    RequireFinite(tolerance, "tolerance", "HomMat2dIsSimilarity");
    return Internal::IsSimilarityTransform(homMat2d, tolerance);
}

double HomMat2dDeterminant(const QMatrix& homMat2d) {
    RequireMatrixFinite(homMat2d, "HomMat2dDeterminant");
    return homMat2d.Determinant();
}

} // namespace Qi::Vision::Transform
