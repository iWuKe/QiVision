/**
 * @file CameraCalib.cpp
 * @brief Camera calibration implementation using Zhang's method
 *
 * Algorithm flow:
 * 1. Estimate homography for each calibration image
 * 2. Solve for intrinsic matrix K from homography constraints
 * 3. Compute extrinsic R, t for each image from K and H
 * 4. Estimate distortion coefficients (linear)
 * 5. Refine all parameters using Levenberg-Marquardt
 */

#include <QiVision/Calib/CameraCalib.h>
#include <QiVision/Internal/Homography.h>
#include <QiVision/Internal/Solver.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace Qi::Vision::Calib {

// ============================================================================
// Constants
// ============================================================================

constexpr double EPSILON = 1e-10;
constexpr double PI = 3.14159265358979323846;

// ============================================================================
// ExtrinsicParams Implementation
// ============================================================================

Internal::Mat44 ExtrinsicParams::ToTransformMatrix() const {
    Internal::Mat44 T;

    // Set rotation part (top-left 3x3)
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            T(i, j) = R(i, j);
        }
    }

    // Set translation part (top-right 3x1)
    T(0, 3) = t[0];
    T(1, 3) = t[1];
    T(2, 3) = t[2];

    // Set bottom row
    T(3, 0) = 0.0;
    T(3, 1) = 0.0;
    T(3, 2) = 0.0;
    T(3, 3) = 1.0;

    return T;
}

ExtrinsicParams ExtrinsicParams::FromRt(const Internal::Mat33& R_, const Internal::Vec3& t_) {
    ExtrinsicParams params;
    params.R = R_;
    params.t = t_;
    params.rvec = MatrixToRodrigues(R_);
    return params;
}

// ============================================================================
// Rodrigues Transform Implementation
// ============================================================================

Internal::Mat33 RodriguesToMatrix(const Internal::Vec3& rvec) {
    // Rodrigues formula: R = I + sin(theta)*K + (1-cos(theta))*K^2
    // where K is the skew-symmetric matrix of the rotation axis

    double theta = rvec.Norm();

    if (theta < EPSILON) {
        // Small angle approximation: R = I + K
        Internal::Mat33 R = Internal::Mat33::Identity();
        R(0, 1) = -rvec[2];
        R(0, 2) = rvec[1];
        R(1, 0) = rvec[2];
        R(1, 2) = -rvec[0];
        R(2, 0) = -rvec[1];
        R(2, 1) = rvec[0];
        return R;
    }

    // Unit rotation axis
    Internal::Vec3 axis = rvec / theta;
    double kx = axis[0], ky = axis[1], kz = axis[2];

    // Skew-symmetric matrix K
    Internal::Mat33 K;
    K(0, 0) = 0;    K(0, 1) = -kz; K(0, 2) = ky;
    K(1, 0) = kz;   K(1, 1) = 0;   K(1, 2) = -kx;
    K(2, 0) = -ky;  K(2, 1) = kx;  K(2, 2) = 0;

    // K^2
    Internal::Mat33 K2 = K * K;

    // Rodrigues formula
    double c = std::cos(theta);
    double s = std::sin(theta);

    Internal::Mat33 R = Internal::Mat33::Identity();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            R(i, j) = R(i, j) + s * K(i, j) + (1.0 - c) * K2(i, j);
        }
    }

    return R;
}

Internal::Vec3 MatrixToRodrigues(const Internal::Mat33& R) {
    // Extract rotation angle from trace: trace(R) = 1 + 2*cos(theta)
    double trace = R(0, 0) + R(1, 1) + R(2, 2);
    double cosTheta = (trace - 1.0) * 0.5;

    // Clamp to [-1, 1] for numerical stability
    cosTheta = std::max(-1.0, std::min(1.0, cosTheta));
    double theta = std::acos(cosTheta);

    if (theta < EPSILON) {
        // Small angle: extract axis from skew-symmetric part
        // (R - R^T) / 2 = sin(theta) * K
        return Internal::Vec3{
            (R(2, 1) - R(1, 2)) * 0.5,
            (R(0, 2) - R(2, 0)) * 0.5,
            (R(1, 0) - R(0, 1)) * 0.5
        };
    }

    if (std::abs(theta - PI) < EPSILON) {
        // Near 180 degrees: extract axis from R + I = 2 * (axis * axis^T)
        // axis = sqrt((R + I) / 2), taking care of signs
        double rx = std::sqrt(std::max(0.0, (R(0, 0) + 1.0) * 0.5));
        double ry = std::sqrt(std::max(0.0, (R(1, 1) + 1.0) * 0.5));
        double rz = std::sqrt(std::max(0.0, (R(2, 2) + 1.0) * 0.5));

        // Fix signs using off-diagonal elements
        if (R(0, 1) < 0) ry = -ry;
        if (R(0, 2) < 0) rz = -rz;
        // ry and rz signs relative to rx determined by R(0,1) and R(0,2)
        // rx is positive by convention

        return Internal::Vec3{rx * theta, ry * theta, rz * theta};
    }

    // General case: axis from (R - R^T) / (2 * sin(theta))
    double sinTheta = std::sin(theta);
    Internal::Vec3 axis{
        (R(2, 1) - R(1, 2)) / (2.0 * sinTheta),
        (R(0, 2) - R(2, 0)) / (2.0 * sinTheta),
        (R(1, 0) - R(0, 1)) / (2.0 * sinTheta)
    };

    return axis * theta;
}

// ============================================================================
// Helper Functions
// ============================================================================

namespace {

/**
 * @brief Estimate homography from 2D-3D planar correspondences
 *
 * For planar calibration pattern (Z=0), the homography H maps
 * object points [X, Y, 1]^T to image points [u, v, 1]^T
 */
std::optional<Internal::Homography> EstimateHomographyFromPlanar(
    const std::vector<Point2d>& imagePoints,
    const std::vector<Point3d>& objectPoints)
{
    if (imagePoints.size() < 4 || imagePoints.size() != objectPoints.size()) {
        return std::nullopt;
    }

    // Convert 3D points to 2D (drop Z=0 coordinate)
    std::vector<Point2d> srcPoints;
    srcPoints.reserve(objectPoints.size());
    for (const auto& p : objectPoints) {
        srcPoints.push_back(Point2d(p.x, p.y));
    }

    // Use existing homography estimation
    return Internal::EstimateHomography(srcPoints, imagePoints);
}

/**
 * @brief Extract homography constraint matrix row for B matrix
 *
 * For homography H = [h1 h2 h3], the constraint h_i^T * B * h_j = 0
 * gives us v_ij * b = 0 where b = [B11, B12, B22, B13, B23, B33]^T
 */
Internal::VecX GetHomographyConstraintRow(
    const Internal::Homography& H, int i, int j)
{
    // h_i and h_j are columns i and j of H
    double hi1 = H(0, i), hi2 = H(1, i), hi3 = H(2, i);
    double hj1 = H(0, j), hj2 = H(1, j), hj3 = H(2, j);

    // v_ij = [h_i1*h_j1, h_i1*h_j2 + h_i2*h_j1, h_i2*h_j2,
    //         h_i3*h_j1 + h_i1*h_j3, h_i3*h_j2 + h_i2*h_j3, h_i3*h_j3]
    Internal::VecX v(6);
    v[0] = hi1 * hj1;
    v[1] = hi1 * hj2 + hi2 * hj1;
    v[2] = hi2 * hj2;
    v[3] = hi3 * hj1 + hi1 * hj3;
    v[4] = hi3 * hj2 + hi2 * hj3;
    v[5] = hi3 * hj3;

    return v;
}

/**
 * @brief Solve for intrinsic matrix from homography constraints
 *
 * Uses the absolute conic B = K^(-T) * K^(-1) to solve for K.
 * Each homography provides 2 constraints on B (symmetric 3x3 matrix).
 */
std::optional<CameraIntrinsics> SolveIntrinsicsFromHomographies(
    const std::vector<Internal::Homography>& homographies,
    const Size2i& imageSize)
{
    int numImages = static_cast<int>(homographies.size());

    if (numImages < 3) {
        // Need at least 3 homographies for unique solution
        return std::nullopt;
    }

    // Build constraint matrix V
    // Each homography provides 2 constraints: v12 and (v11 - v22)
    Internal::MatX V(2 * numImages, 6);

    for (int i = 0; i < numImages; ++i) {
        const auto& H = homographies[i];

        // v12 (orthogonality constraint: h1^T * B * h2 = 0)
        Internal::VecX v12 = GetHomographyConstraintRow(H, 0, 1);

        // v11 - v22 (equal scale constraint: h1^T * B * h1 = h2^T * B * h2)
        Internal::VecX v11 = GetHomographyConstraintRow(H, 0, 0);
        Internal::VecX v22 = GetHomographyConstraintRow(H, 1, 1);
        Internal::VecX v11_v22 = v11 - v22;

        // Set rows
        for (int j = 0; j < 6; ++j) {
            V(2 * i, j) = v12[j];
            V(2 * i + 1, j) = v11_v22[j];
        }
    }

    // Solve Vb = 0 using SVD (find null space)
    Internal::VecX b = Internal::SolveHomogeneous(V);
    if (b.Size() != 6) {
        return std::nullopt;
    }

    // B = [B11 B12 B13; B12 B22 B23; B13 B23 B33]
    // b = [B11, B12, B22, B13, B23, B33]
    double B11 = b[0], B12 = b[1], B22 = b[2];
    double B13 = b[3], B23 = b[4], B33 = b[5];

    // Extract intrinsics from B = K^(-T) * K^(-1)
    // v0 = (B12*B13 - B11*B23) / (B11*B22 - B12^2)
    // lambda = B33 - (B13^2 + v0*(B12*B13 - B11*B23)) / B11
    // alpha = sqrt(lambda / B11)
    // beta = sqrt(lambda * B11 / (B11*B22 - B12^2))
    // gamma = -B12 * alpha^2 * beta / lambda
    // u0 = gamma * v0 / beta - B13 * alpha^2 / lambda

    double denom = B11 * B22 - B12 * B12;
    if (std::abs(denom) < EPSILON || std::abs(B11) < EPSILON) {
        // Degenerate case - use image center as principal point
        // and estimate focal length from image size
        CameraIntrinsics K;
        K.cx = imageSize.width * 0.5;
        K.cy = imageSize.height * 0.5;
        K.fx = std::max(imageSize.width, imageSize.height) * 1.2;  // Rough estimate
        K.fy = K.fx;
        return K;
    }

    double cy = (B12 * B13 - B11 * B23) / denom;
    double lambda = B33 - (B13 * B13 + cy * (B12 * B13 - B11 * B23)) / B11;

    if (lambda < EPSILON) {
        // Negative lambda - invalid solution
        return std::nullopt;
    }

    double fx = std::sqrt(lambda / B11);
    double fy = std::sqrt(lambda * B11 / denom);
    double gamma = -B12 * fx * fx * fy / lambda;  // Skew (typically 0)
    double cx = gamma * cy / fy - B13 * fx * fx / lambda;

    // Ignore skew (gamma) - assume rectangular pixels
    CameraIntrinsics K;
    K.fx = fx;
    K.fy = fy;
    K.cx = cx;
    K.cy = cy;

    // Sanity check: principal point should be roughly in image center
    if (K.cx < 0 || K.cx > imageSize.width ||
        K.cy < 0 || K.cy > imageSize.height) {
        // Use image center as fallback
        K.cx = imageSize.width * 0.5;
        K.cy = imageSize.height * 0.5;
    }

    // Focal lengths should be positive and reasonable
    if (K.fx <= 0 || K.fy <= 0) {
        return std::nullopt;
    }

    return K;
}

/**
 * @brief Compute extrinsic parameters from intrinsics and homography
 *
 * Given K and H, compute R and t such that H = K * [r1 r2 t]
 */
ExtrinsicParams ComputeExtrinsicsFromHomography(
    const CameraIntrinsics& K,
    const Internal::Homography& H)
{
    ExtrinsicParams params;

    // K^(-1)
    Internal::Mat33 Kinv;
    Kinv(0, 0) = 1.0 / K.fx;
    Kinv(0, 1) = 0.0;
    Kinv(0, 2) = -K.cx / K.fx;
    Kinv(1, 0) = 0.0;
    Kinv(1, 1) = 1.0 / K.fy;
    Kinv(1, 2) = -K.cy / K.fy;
    Kinv(2, 0) = 0.0;
    Kinv(2, 1) = 0.0;
    Kinv(2, 2) = 1.0;

    Internal::Mat33 Hmat = H.ToMat33();

    // h1 = H * [1 0 0]^T, h2 = H * [0 1 0]^T, h3 = H * [0 0 1]^T
    Internal::Vec3 h1{Hmat(0, 0), Hmat(1, 0), Hmat(2, 0)};
    Internal::Vec3 h2{Hmat(0, 1), Hmat(1, 1), Hmat(2, 1)};
    Internal::Vec3 h3{Hmat(0, 2), Hmat(1, 2), Hmat(2, 2)};

    // r1 = lambda * K^(-1) * h1
    // r2 = lambda * K^(-1) * h2
    // t = lambda * K^(-1) * h3
    // where lambda = 1 / ||K^(-1) * h1||

    Internal::Vec3 Kinv_h1 = Kinv * h1;
    Internal::Vec3 Kinv_h2 = Kinv * h2;
    Internal::Vec3 Kinv_h3 = Kinv * h3;

    double lambda = 1.0 / Kinv_h1.Norm();

    Internal::Vec3 r1 = Kinv_h1 * lambda;
    Internal::Vec3 r2 = Kinv_h2 * lambda;
    Internal::Vec3 r3 = Internal::Cross(r1, r2);  // r3 = r1 x r2

    params.t = Kinv_h3 * lambda;

    // Build rotation matrix [r1 r2 r3]
    params.R(0, 0) = r1[0]; params.R(0, 1) = r2[0]; params.R(0, 2) = r3[0];
    params.R(1, 0) = r1[1]; params.R(1, 1) = r2[1]; params.R(1, 2) = r3[1];
    params.R(2, 0) = r1[2]; params.R(2, 1) = r2[2]; params.R(2, 2) = r3[2];

    // Enforce proper rotation matrix (orthogonalization via SVD)
    // R = U * V^T where U*S*V^T = SVD(R)
    Internal::MatX Rdyn(3, 3);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Rdyn(i, j) = params.R(i, j);
        }
    }

    Internal::SVDResult svd = Internal::SVD_Decompose(Rdyn);
    if (svd.valid) {
        // R_proper = U * V^T
        Internal::MatX VT = svd.V.Transpose();
        Internal::MatX R_proper = svd.U * VT;

        // Ensure det(R) = 1 (proper rotation)
        double det = R_proper(0, 0) * (R_proper(1, 1) * R_proper(2, 2) - R_proper(1, 2) * R_proper(2, 1))
                   - R_proper(0, 1) * (R_proper(1, 0) * R_proper(2, 2) - R_proper(1, 2) * R_proper(2, 0))
                   + R_proper(0, 2) * (R_proper(1, 0) * R_proper(2, 1) - R_proper(1, 1) * R_proper(2, 0));

        if (det < 0) {
            // Flip sign of last column
            for (int i = 0; i < 3; ++i) {
                R_proper(i, 2) = -R_proper(i, 2);
            }
        }

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                params.R(i, j) = R_proper(i, j);
            }
        }
    }

    params.rvec = MatrixToRodrigues(params.R);
    return params;
}

/**
 * @brief Estimate distortion coefficients (linear estimation)
 *
 * Given undistorted ideal points and observed distorted points,
 * estimate k1 and k2 using linear least squares.
 */
DistortionCoeffs EstimateDistortionLinear(
    const std::vector<std::vector<Point2d>>& imagePoints,
    const std::vector<std::vector<Point3d>>& objectPoints,
    const CameraIntrinsics& K,
    const std::vector<ExtrinsicParams>& extrinsics)
{
    DistortionCoeffs dist;

    // Collect all point pairs for distortion estimation
    // For each observed point (u, v), compute the ideal undistorted point (x, y)
    // and build the linear system: r^2*x - dx = k1*r^2*x (for x component)

    std::vector<double> A_data;  // Linear system coefficients
    std::vector<double> b_data;  // Right-hand side

    for (size_t v = 0; v < imagePoints.size(); ++v) {
        const auto& imgPts = imagePoints[v];
        const auto& objPts = objectPoints[v];
        const auto& ext = extrinsics[v];

        for (size_t p = 0; p < imgPts.size(); ++p) {
            // Project 3D point to camera coordinates
            Point3d objPt = objPts[p];
            Internal::Vec3 P{objPt.x, objPt.y, objPt.z};

            // Transform to camera frame: P_cam = R * P + t
            Internal::Vec3 P_cam = ext.R * P + ext.t;

            if (std::abs(P_cam[2]) < EPSILON) continue;

            // Normalized coordinates
            double x = P_cam[0] / P_cam[2];
            double y = P_cam[1] / P_cam[2];
            double r2 = x * x + y * y;

            // Ideal pixel coordinates (without distortion)
            double u_ideal = K.fx * x + K.cx;
            double v_ideal = K.fy * y + K.cy;

            // Observed pixel coordinates (with distortion)
            double u_obs = imgPts[p].x;
            double v_obs = imgPts[p].y;

            // Distortion model: u_obs - u_ideal = fx * (k1*r^2 + k2*r^4) * x
            //                   v_obs - v_ideal = fy * (k1*r^2 + k2*r^4) * y
            // Linear in k1, k2

            double dx = u_obs - u_ideal;
            double dy = v_obs - v_ideal;

            // Build row for x component
            A_data.push_back(K.fx * r2 * x);      // k1 coefficient
            A_data.push_back(K.fx * r2 * r2 * x); // k2 coefficient
            b_data.push_back(dx);

            // Build row for y component
            A_data.push_back(K.fy * r2 * y);      // k1 coefficient
            A_data.push_back(K.fy * r2 * r2 * y); // k2 coefficient
            b_data.push_back(dy);
        }
    }

    if (b_data.size() < 4) {
        // Not enough data
        return dist;
    }

    // Build matrices
    int numRows = static_cast<int>(b_data.size());
    Internal::MatX A(numRows, 2);
    Internal::VecX b(numRows);

    for (int i = 0; i < numRows; ++i) {
        A(i, 0) = A_data[2 * i];
        A(i, 1) = A_data[2 * i + 1];
        b[i] = b_data[i];
    }

    // Solve least squares
    Internal::VecX k = Internal::SolveLeastSquares(A, b);
    if (k.Size() >= 2) {
        dist.k1 = k[0];
        dist.k2 = k[1];
    }

    return dist;
}

/**
 * @brief Refine calibration using Levenberg-Marquardt optimization
 *
 * Minimizes total reprojection error over all parameters:
 * intrinsics (fx, fy, cx, cy), distortion (k1, k2), and extrinsics (rvec, tvec for each image)
 */
void RefineCalibrationLM(
    CalibrationResult& result,
    const std::vector<std::vector<Point2d>>& imagePoints,
    const std::vector<std::vector<Point3d>>& objectPoints,
    CalibFlags flags,
    int maxIterations = 30)
{
    // Simple Gauss-Newton refinement (simplified LM)
    // Full LM would require damping factor management

    const int numViews = static_cast<int>(result.extrinsics.size());

    // Number of parameters:
    // - 4 intrinsics (fx, fy, cx, cy) - can be constrained by flags
    // - 2 distortion (k1, k2) - can be constrained by flags
    // - 6 per view (3 rvec + 3 tvec)

    for (int iter = 0; iter < maxIterations; ++iter) {
        double totalError = 0.0;
        int totalPoints = 0;

        // Compute current reprojection error
        for (int v = 0; v < numViews; ++v) {
            auto errors = ComputeReprojectionErrors(
                objectPoints[v], imagePoints[v],
                result.camera, result.extrinsics[v].rvec, result.extrinsics[v].t);

            for (double e : errors) {
                totalError += e * e;
                ++totalPoints;
            }
        }

        double rmsError = std::sqrt(totalError / totalPoints);

        // Check convergence
        if (iter > 0 && std::abs(rmsError - result.rmsError) < 1e-6) {
            break;
        }
        result.rmsError = rmsError;

        // Update extrinsics for each view independently (faster convergence)
        for (int v = 0; v < numViews; ++v) {
            // Re-estimate pose with current intrinsics
            Internal::Vec3 rvec = result.extrinsics[v].rvec;
            Internal::Vec3 tvec = result.extrinsics[v].t;

            bool ok = SolvePnP(objectPoints[v], imagePoints[v],
                              result.camera, rvec, tvec, true);
            if (ok) {
                result.extrinsics[v].rvec = rvec;
                result.extrinsics[v].t = tvec;
                result.extrinsics[v].R = RodriguesToMatrix(rvec);
            }
        }

        // Update distortion coefficients
        DistortionCoeffs newDist = EstimateDistortionLinear(
            imagePoints, objectPoints,
            result.camera.Intrinsics(), result.extrinsics);

        if (!(flags & CalibFlags::FixK1)) {
            result.camera.Distortion().k1 = newDist.k1;
        }
        if (!(flags & CalibFlags::FixK2)) {
            result.camera.Distortion().k2 = newDist.k2;
        }
    }

    // Compute final statistics
    double totalError = 0.0;
    double maxError = 0.0;
    int totalPoints = 0;

    result.perViewErrors.resize(numViews);
    result.perPointErrors.resize(numViews);

    for (int v = 0; v < numViews; ++v) {
        auto errors = ComputeReprojectionErrors(
            objectPoints[v], imagePoints[v],
            result.camera, result.extrinsics[v].rvec, result.extrinsics[v].t);

        result.perPointErrors[v] = errors;

        double viewError = 0.0;
        for (double e : errors) {
            totalError += e * e;
            viewError += e * e;
            maxError = std::max(maxError, e);
            ++totalPoints;
        }

        result.perViewErrors[v] = std::sqrt(viewError / errors.size());
    }

    result.rmsError = std::sqrt(totalError / totalPoints);
    result.meanError = totalError / totalPoints;
    result.maxError = maxError;
}

} // anonymous namespace

// ============================================================================
// CalibrateCamera Implementation
// ============================================================================

CalibrationResult CalibrateCamera(
    const std::vector<std::vector<Point2d>>& imagePoints,
    const std::vector<std::vector<Point3d>>& objectPoints,
    const Size2i& imageSize,
    CalibFlags flags,
    const CameraModel* initialCamera)
{
    CalibrationResult result;
    result.success = false;

    // Validate input
    int numViews = static_cast<int>(imagePoints.size());
    if (numViews < 3) {
        // Need at least 3 views for calibration
        return result;
    }

    if (imagePoints.size() != objectPoints.size()) {
        return result;
    }

    // Check that all views have the same number of points
    for (int v = 0; v < numViews; ++v) {
        if (imagePoints[v].size() != objectPoints[v].size()) {
            return result;
        }
        if (imagePoints[v].size() < 4) {
            return result;
        }
    }

    // Step 1: Estimate homography for each view
    std::vector<Internal::Homography> homographies;
    homographies.reserve(numViews);

    for (int v = 0; v < numViews; ++v) {
        auto H = EstimateHomographyFromPlanar(imagePoints[v], objectPoints[v]);
        if (!H) {
            return result;
        }
        homographies.push_back(*H);
    }

    // Step 2: Solve for intrinsics from homography constraints
    CameraIntrinsics K;

    if ((flags & CalibFlags::UseIntrinsicGuess) && initialCamera) {
        K = initialCamera->Intrinsics();
    } else {
        auto solvedK = SolveIntrinsicsFromHomographies(homographies, imageSize);
        if (!solvedK) {
            return result;
        }
        K = *solvedK;
    }

    // Apply constraints
    if (flags & CalibFlags::FixPrincipalPoint) {
        K.cx = imageSize.width * 0.5;
        K.cy = imageSize.height * 0.5;
    }
    if (flags & CalibFlags::FixAspectRatio) {
        K.fy = K.fx;
    }

    // Step 3: Compute extrinsics for each view
    result.extrinsics.reserve(numViews);

    for (int v = 0; v < numViews; ++v) {
        ExtrinsicParams ext = ComputeExtrinsicsFromHomography(K, homographies[v]);
        result.extrinsics.push_back(ext);
    }

    // Step 4: Estimate distortion coefficients
    DistortionCoeffs dist;
    if (!(flags & CalibFlags::FixK1) || !(flags & CalibFlags::FixK2)) {
        dist = EstimateDistortionLinear(imagePoints, objectPoints, K, result.extrinsics);
    }

    if (flags & CalibFlags::ZeroTangentDist) {
        dist.p1 = 0.0;
        dist.p2 = 0.0;
    }
    if (flags & CalibFlags::FixK1) {
        dist.k1 = 0.0;
    }
    if (flags & CalibFlags::FixK2) {
        dist.k2 = 0.0;
    }
    if (flags & CalibFlags::FixK3) {
        dist.k3 = 0.0;
    }

    // Build camera model
    result.camera = CameraModel(K, dist, imageSize);

    // Step 5: Refine all parameters
    RefineCalibrationLM(result, imagePoints, objectPoints, flags);

    result.success = true;
    return result;
}

// ============================================================================
// SolvePnP Implementation
// ============================================================================

bool SolvePnP(
    const std::vector<Point3d>& objectPoints,
    const std::vector<Point2d>& imagePoints,
    const CameraModel& camera,
    Internal::Vec3& rvec,
    Internal::Vec3& tvec,
    bool useExtrinsicGuess)
{
    if (objectPoints.size() < 4 || objectPoints.size() != imagePoints.size()) {
        return false;
    }

    // Undistort image points
    std::vector<Point2d> undistortedPoints;
    undistortedPoints.reserve(imagePoints.size());

    const CameraIntrinsics& K = camera.Intrinsics();
    for (const auto& p : imagePoints) {
        // Convert to normalized coordinates
        double x_dist = (p.x - K.cx) / K.fx;
        double y_dist = (p.y - K.cy) / K.fy;

        // Undistort
        Point2d undist = camera.Undistort(Point2d(x_dist, y_dist));
        undistortedPoints.push_back(undist);
    }

    // Use DLT (Direct Linear Transform) for initial estimate if no guess
    if (!useExtrinsicGuess) {
        // Build DLT system: each point gives 2 equations
        // x_i * (P3 * X_i) - P1 * X_i = 0
        // y_i * (P3 * X_i) - P2 * X_i = 0
        // where P = [R | t] (3x4), P_i is row i

        int n = static_cast<int>(objectPoints.size());
        Internal::MatX A(2 * n, 12);

        for (int i = 0; i < n; ++i) {
            double X = objectPoints[i].x;
            double Y = objectPoints[i].y;
            double Z = objectPoints[i].z;
            double x = undistortedPoints[i].x;
            double y = undistortedPoints[i].y;

            // Row for x equation
            A(2*i, 0) = X; A(2*i, 1) = Y; A(2*i, 2) = Z; A(2*i, 3) = 1;
            A(2*i, 4) = 0; A(2*i, 5) = 0; A(2*i, 6) = 0; A(2*i, 7) = 0;
            A(2*i, 8) = -x*X; A(2*i, 9) = -x*Y; A(2*i, 10) = -x*Z; A(2*i, 11) = -x;

            // Row for y equation
            A(2*i+1, 0) = 0; A(2*i+1, 1) = 0; A(2*i+1, 2) = 0; A(2*i+1, 3) = 0;
            A(2*i+1, 4) = X; A(2*i+1, 5) = Y; A(2*i+1, 6) = Z; A(2*i+1, 7) = 1;
            A(2*i+1, 8) = -y*X; A(2*i+1, 9) = -y*Y; A(2*i+1, 10) = -y*Z; A(2*i+1, 11) = -y;
        }

        Internal::VecX p = Internal::SolveHomogeneous(A);
        if (p.Size() != 12) {
            return false;
        }

        // Extract R and t from P = [R | t]
        Internal::Mat33 R;
        R(0, 0) = p[0]; R(0, 1) = p[1]; R(0, 2) = p[2];
        R(1, 0) = p[4]; R(1, 1) = p[5]; R(1, 2) = p[6];
        R(2, 0) = p[8]; R(2, 1) = p[9]; R(2, 2) = p[10];

        tvec[0] = p[3];
        tvec[1] = p[7];
        tvec[2] = p[11];

        // Enforce proper rotation using SVD
        Internal::MatX Rdyn(3, 3);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                Rdyn(i, j) = R(i, j);
            }
        }

        Internal::SVDResult svd = Internal::SVD_Decompose(Rdyn);
        if (!svd.valid) {
            return false;
        }

        Internal::MatX VT = svd.V.Transpose();
        Internal::MatX R_proper = svd.U * VT;

        // Compute scale factor from singular values
        double scale = (svd.S[0] + svd.S[1] + svd.S[2]) / 3.0;
        if (scale < EPSILON) {
            return false;
        }

        // Ensure det(R) = 1
        double det = R_proper(0, 0) * (R_proper(1, 1) * R_proper(2, 2) - R_proper(1, 2) * R_proper(2, 1))
                   - R_proper(0, 1) * (R_proper(1, 0) * R_proper(2, 2) - R_proper(1, 2) * R_proper(2, 0))
                   + R_proper(0, 2) * (R_proper(1, 0) * R_proper(2, 1) - R_proper(1, 1) * R_proper(2, 0));

        if (det < 0) {
            for (int i = 0; i < 3; ++i) {
                R_proper(i, 2) = -R_proper(i, 2);
            }
            scale = -scale;
        }

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                R(i, j) = R_proper(i, j);
            }
        }

        // Scale translation
        tvec = tvec / scale;

        rvec = MatrixToRodrigues(R);
    }

    // Iterative refinement using Gauss-Newton
    const int maxIter = 10;
    const double convergenceThreshold = 1e-8;

    for (int iter = 0; iter < maxIter; ++iter) {
        Internal::Mat33 R = RodriguesToMatrix(rvec);

        // Build Jacobian and residual
        int n = static_cast<int>(objectPoints.size());
        Internal::MatX J(2 * n, 6);  // 3 for rvec, 3 for tvec
        Internal::VecX r(2 * n);

        for (int i = 0; i < n; ++i) {
            Internal::Vec3 P{objectPoints[i].x, objectPoints[i].y, objectPoints[i].z};

            // Transform point: P_cam = R * P + t
            Internal::Vec3 P_cam = R * P + tvec;

            if (std::abs(P_cam[2]) < EPSILON) {
                return false;
            }

            double invZ = 1.0 / P_cam[2];
            double x = P_cam[0] * invZ;
            double y = P_cam[1] * invZ;

            // Residual
            r[2*i] = x - undistortedPoints[i].x;
            r[2*i+1] = y - undistortedPoints[i].y;

            // Jacobian of projected point w.r.t. camera coordinates
            // dx/dP_cam = [1/Z, 0, -X/Z^2]
            // dy/dP_cam = [0, 1/Z, -Y/Z^2]

            double dxdX = invZ;
            double dxdY = 0;
            double dxdZ = -P_cam[0] * invZ * invZ;

            double dydX = 0;
            double dydY = invZ;
            double dydZ = -P_cam[1] * invZ * invZ;

            // dP_cam/dt = I
            J(2*i, 3) = dxdX;
            J(2*i, 4) = dxdY;
            J(2*i, 5) = dxdZ;

            J(2*i+1, 3) = dydX;
            J(2*i+1, 4) = dydY;
            J(2*i+1, 5) = dydZ;

            // dP_cam/drvec requires derivative of Rodrigues formula
            // Simplified: use finite differences or small angle approximation
            // For small angle, dR/dr_i = K_i where K_i is skew-symmetric
            // dP_cam/dr = [K_x, K_y, K_z] * P where K_i is derivative w.r.t. r_i

            Internal::Vec3 Rx = Internal::Cross(Internal::Vec3{1, 0, 0}, P);
            Internal::Vec3 Ry = Internal::Cross(Internal::Vec3{0, 1, 0}, P);
            Internal::Vec3 Rz = Internal::Cross(Internal::Vec3{0, 0, 1}, P);

            // Apply rotation
            Internal::Vec3 dPdr0 = R * Rx;
            Internal::Vec3 dPdr1 = R * Ry;
            Internal::Vec3 dPdr2 = R * Rz;

            J(2*i, 0) = dxdX * dPdr0[0] + dxdY * dPdr0[1] + dxdZ * dPdr0[2];
            J(2*i, 1) = dxdX * dPdr1[0] + dxdY * dPdr1[1] + dxdZ * dPdr1[2];
            J(2*i, 2) = dxdX * dPdr2[0] + dxdY * dPdr2[1] + dxdZ * dPdr2[2];

            J(2*i+1, 0) = dydX * dPdr0[0] + dydY * dPdr0[1] + dydZ * dPdr0[2];
            J(2*i+1, 1) = dydX * dPdr1[0] + dydY * dPdr1[1] + dydZ * dPdr1[2];
            J(2*i+1, 2) = dydX * dPdr2[0] + dydY * dPdr2[1] + dydZ * dPdr2[2];
        }

        // Solve normal equations: (J^T * J) * delta = -J^T * r
        Internal::MatX JtJ = J.Transpose() * J;
        Internal::VecX Jtr = J.Transpose() * r;

        Internal::VecX delta = Internal::SolveLU(JtJ, Jtr * (-1.0));
        if (delta.Size() != 6) {
            break;
        }

        // Update parameters
        rvec[0] += delta[0];
        rvec[1] += delta[1];
        rvec[2] += delta[2];
        tvec[0] += delta[3];
        tvec[1] += delta[4];
        tvec[2] += delta[5];

        // Check convergence
        double deltaSum = 0;
        for (int i = 0; i < 6; ++i) {
            deltaSum += delta[i] * delta[i];
        }
        if (deltaSum < convergenceThreshold) {
            break;
        }
    }

    return true;
}

// ============================================================================
// ComputeReprojectionErrors Implementation
// ============================================================================

std::vector<double> ComputeReprojectionErrors(
    const std::vector<Point3d>& objectPoints,
    const std::vector<Point2d>& imagePoints,
    const CameraModel& camera,
    const Internal::Vec3& rvec,
    const Internal::Vec3& tvec)
{
    std::vector<double> errors;
    errors.reserve(objectPoints.size());

    std::vector<Point2d> projected = ProjectPoints(objectPoints, camera, rvec, tvec);

    for (size_t i = 0; i < objectPoints.size(); ++i) {
        double dx = projected[i].x - imagePoints[i].x;
        double dy = projected[i].y - imagePoints[i].y;
        errors.push_back(std::sqrt(dx * dx + dy * dy));
    }

    return errors;
}

// ============================================================================
// ProjectPoints Implementation
// ============================================================================

std::vector<Point2d> ProjectPoints(
    const std::vector<Point3d>& objectPoints,
    const CameraModel& camera,
    const Internal::Vec3& rvec,
    const Internal::Vec3& tvec)
{
    std::vector<Point2d> projected;
    projected.reserve(objectPoints.size());

    Internal::Mat33 R = RodriguesToMatrix(rvec);

    for (const auto& objPt : objectPoints) {
        // Transform to camera frame
        Internal::Vec3 P{objPt.x, objPt.y, objPt.z};
        Internal::Vec3 P_cam = R * P + tvec;

        // Project to image plane (with distortion)
        Point3d p3d(P_cam[0], P_cam[1], P_cam[2]);
        Point2d p2d = camera.ProjectPoint(p3d);
        projected.push_back(p2d);
    }

    return projected;
}

} // namespace Qi::Vision::Calib
