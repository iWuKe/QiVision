/**
 * @file SubPixel.cpp
 * @brief Implementation of subpixel refinement algorithms
 */

#include <QiVision/Internal/SubPixel.h>
#include <QiVision/Internal/Solver.h>

#include <cmath>
#include <algorithm>
#include <limits>

namespace Qi::Vision::Internal {

// =============================================================================
// 1D Subpixel Refinement Functions
// =============================================================================

SubPixelResult1D RefineSubPixel1D(const double* signal, size_t size,
                                   int32_t index,
                                   SubPixelMethod1D method,
                                   int32_t windowHalfSize) {
    SubPixelResult1D result;
    result.integerPosition = index;
    result.subpixelPosition = static_cast<double>(index);

    // Check bounds
    if (index < 0 || static_cast<size_t>(index) >= size) {
        result.success = false;
        result.confidence = 0.0;
        return result;
    }

    switch (method) {
        case SubPixelMethod1D::Parabolic: {
            // Need at least 3 points
            if (index < 1 || static_cast<size_t>(index) >= size - 1) {
                result.success = false;
                result.confidence = 0.0;
                return result;
            }

            double v0 = signal[index - 1];
            double v1 = signal[index];
            double v2 = signal[index + 1];

            double offset = RefineParabolic1D(v0, v1, v2);
            double curvature = ComputeCurvature1D(v0, v1, v2);
            double peakValue = ParabolicPeakValue(v0, v1, v2, offset);

            result.success = true;
            result.offset = offset;
            result.subpixelPosition = index + offset;
            result.peakValue = peakValue;
            result.curvature = curvature;
            result.confidence = ComputeSubPixelConfidence1D(curvature, peakValue, 0.0, offset);
            break;
        }

        case SubPixelMethod1D::Gaussian:
            return RefineGaussian1D(signal, size, index);

        case SubPixelMethod1D::Centroid:
            return RefineCentroid1D(signal, size, index, windowHalfSize, false);

        case SubPixelMethod1D::Quartic:
            return RefineQuartic1D(signal, size, index);

        case SubPixelMethod1D::Linear: {
            // Linear interpolation - find zero crossing or inflection
            if (index < 1 || static_cast<size_t>(index) >= size - 1) {
                result.success = false;
                result.confidence = 0.0;
                return result;
            }

            double v0 = signal[index - 1];
            double v1 = signal[index];
            double v2 = signal[index + 1];

            // Estimate local slope
            double slope = (v2 - v0) * 0.5;
            if (std::abs(slope) < 1e-10) {
                result.success = true;
                result.offset = 0.0;
                result.subpixelPosition = static_cast<double>(index);
                result.peakValue = v1;
                result.confidence = 0.5;
            } else {
                // Interpolate to find where derivative is zero
                double offset = -v1 / slope;
                offset = Clamp(offset, -SUBPIXEL_MAX_OFFSET, SUBPIXEL_MAX_OFFSET);
                result.success = true;
                result.offset = offset;
                result.subpixelPosition = index + offset;
                result.peakValue = v1 + slope * offset;
                result.confidence = 0.6;
            }
            break;
        }

        default:
            result.success = false;
            result.confidence = 0.0;
            break;
    }

    return result;
}

SubPixelResult1D RefineGaussian1D(const double* signal, size_t size, int32_t index) {
    SubPixelResult1D result;
    result.integerPosition = index;
    result.subpixelPosition = static_cast<double>(index);

    // Need at least 3 points
    if (index < 1 || static_cast<size_t>(index) >= size - 1) {
        result.success = false;
        result.confidence = 0.0;
        return result;
    }

    double v0 = signal[index - 1];
    double v1 = signal[index];
    double v2 = signal[index + 1];

    // For Gaussian fitting: y = A * exp(-x^2 / (2*sigma^2))
    // Take log: ln(y) = ln(A) - x^2 / (2*sigma^2)
    // This is a parabola in log domain

    // Ensure positive values
    const double minVal = 1e-10;
    v0 = std::max(v0, minVal);
    v1 = std::max(v1, minVal);
    v2 = std::max(v2, minVal);

    double l0 = std::log(v0);
    double l1 = std::log(v1);
    double l2 = std::log(v2);

    // Parabolic fit in log domain
    double denom = 2.0 * (l0 - 2.0 * l1 + l2);
    if (std::abs(denom) < SUBPIXEL_MIN_CURVATURE) {
        // Flat region, use parabolic instead
        double offset = RefineParabolic1D(v0, v1, v2);
        result.success = true;
        result.offset = offset;
        result.subpixelPosition = index + offset;
        result.peakValue = v1;
        result.curvature = 0.0;
        result.confidence = 0.3;
        return result;
    }

    double offset = (l0 - l2) / denom;
    offset = Clamp(offset, -SUBPIXEL_MAX_OFFSET, SUBPIXEL_MAX_OFFSET);

    // Compute sigma from curvature in log domain
    // d2(ln y)/dx2 = -1/sigma^2
    double logCurvature = l0 - 2.0 * l1 + l2;
    double sigma = std::sqrt(-1.0 / logCurvature);
    sigma = Clamp(sigma, GAUSSIAN_FIT_MIN_SIGMA, GAUSSIAN_FIT_MAX_SIGMA);

    // Compute peak value in original domain
    double logPeak = l1 - logCurvature * offset * offset * 0.5;
    double peakValue = std::exp(logPeak);

    result.success = true;
    result.offset = offset;
    result.subpixelPosition = index + offset;
    result.peakValue = peakValue;
    result.curvature = sigma;  // Store sigma in curvature field
    result.confidence = ComputeSubPixelConfidence1D(-1.0 / (sigma * sigma), peakValue, 0.0, offset);

    return result;
}

SubPixelResult1D RefineCentroid1D(const double* signal, size_t size,
                                   int32_t index, int32_t halfWindow,
                                   bool useAbsValues) {
    SubPixelResult1D result;
    result.integerPosition = index;
    result.subpixelPosition = static_cast<double>(index);

    // Check bounds
    int32_t start = index - halfWindow;
    int32_t end = index + halfWindow;
    if (start < 0 || static_cast<size_t>(end) >= size) {
        result.success = false;
        result.confidence = 0.0;
        return result;
    }

    double sumX = 0.0;
    double sumW = 0.0;
    double peakValue = signal[index];

    for (int32_t i = start; i <= end; ++i) {
        double value = useAbsValues ? std::abs(signal[i]) : signal[i];
        // Shift values to be positive for weighting
        double weight = std::max(0.0, value);
        sumX += weight * i;
        sumW += weight;
    }

    if (sumW < 1e-10) {
        result.success = false;
        result.confidence = 0.0;
        return result;
    }

    double centroid = sumX / sumW;
    double offset = centroid - index;
    offset = Clamp(offset, -SUBPIXEL_MAX_OFFSET, SUBPIXEL_MAX_OFFSET);

    result.success = true;
    result.offset = offset;
    result.subpixelPosition = index + offset;
    result.peakValue = peakValue;
    result.curvature = 0.0;  // Not computed for centroid
    result.confidence = 0.7;  // Moderate confidence for centroid method

    return result;
}

SubPixelResult1D RefineQuartic1D(const double* signal, size_t size, int32_t index) {
    SubPixelResult1D result;
    result.integerPosition = index;
    result.subpixelPosition = static_cast<double>(index);

    // Need 5 points
    if (index < 2 || static_cast<size_t>(index) >= size - 2) {
        // Fall back to parabolic
        if (index >= 1 && static_cast<size_t>(index) < size - 1) {
            double v0 = signal[index - 1];
            double v1 = signal[index];
            double v2 = signal[index + 1];
            double offset = RefineParabolic1D(v0, v1, v2);
            result.success = true;
            result.offset = offset;
            result.subpixelPosition = index + offset;
            result.peakValue = ParabolicPeakValue(v0, v1, v2, offset);
            result.curvature = ComputeCurvature1D(v0, v1, v2);
            result.confidence = 0.6;
            return result;
        }
        result.success = false;
        result.confidence = 0.0;
        return result;
    }

    // Sample 5 points: y(-2), y(-1), y(0), y(1), y(2)
    double v[5];
    for (int32_t i = 0; i < 5; ++i) {
        v[i] = signal[index - 2 + i];
    }

    // Fit 4th order polynomial: y = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4
    // Using least squares to solve for coefficients
    // For x = -2, -1, 0, 1, 2

    // Design matrix X^T * X and X^T * y
    // Due to symmetry, odd terms vanish when evaluating at symmetric points
    // We can use the even polynomial: y = a0 + a2*x^2 + a4*x^4 for peak finding

    // Simplified approach: use polynomial fit to find derivative zero
    // First derivative: a1 + 2*a2*x + 3*a3*x^2 + 4*a4*x^3 = 0 at peak

    // For a symmetric peak, we can use parabolic on wider window
    // or use the 5-point second derivative for better accuracy

    // Compute second derivative using 5-point stencil
    // f''(0) = (v[-2] - 2v[0] + v[2]) / 4 for 5-point
    double d2 = (-v[0] + 16*v[1] - 30*v[2] + 16*v[3] - v[4]) / 12.0;

    // First derivative using 5-point stencil
    double d1 = (v[0] - 8*v[1] + 8*v[3] - v[4]) / 12.0;

    if (std::abs(d2) < SUBPIXEL_MIN_CURVATURE) {
        result.success = true;
        result.offset = 0.0;
        result.subpixelPosition = static_cast<double>(index);
        result.peakValue = v[2];
        result.curvature = d2;
        result.confidence = 0.3;
        return result;
    }

    // Newton's method: x = -f'/f''
    double offset = -d1 / d2;
    offset = Clamp(offset, -SUBPIXEL_MAX_OFFSET, SUBPIXEL_MAX_OFFSET);

    // Interpolate peak value using Taylor expansion
    double peakValue = v[2] + d1 * offset + 0.5 * d2 * offset * offset;

    result.success = true;
    result.offset = offset;
    result.subpixelPosition = index + offset;
    result.peakValue = peakValue;
    result.curvature = d2;
    result.confidence = ComputeSubPixelConfidence1D(d2, peakValue, 0.0, offset);

    return result;
}

// =============================================================================
// 2D Subpixel Refinement Functions
// =============================================================================

SubPixelResult2D RefineSubPixel2D(const float* data, int32_t width, int32_t height,
                                   int32_t x, int32_t y,
                                   SubPixelMethod2D method) {
    switch (method) {
        case SubPixelMethod2D::Quadratic:
            return RefineQuadratic2D(data, width, height, x, y);

        case SubPixelMethod2D::Taylor:
            return RefineTaylor2D(data, width, height, x, y);

        case SubPixelMethod2D::Centroid:
            return RefineCentroid2D(data, width, height, x, y);

        case SubPixelMethod2D::BiQuadratic:
            // Fall back to quadratic with larger neighborhood
            // (full biquadratic would need 5x5 window)
            return RefineQuadratic2D(data, width, height, x, y);

        case SubPixelMethod2D::Gaussian2D:
            // Use Taylor method as a good approximation for Gaussian surfaces
            return RefineTaylor2D(data, width, height, x, y);

        default:
            return RefineQuadratic2D(data, width, height, x, y);
    }
}

SubPixelResult2D RefineSubPixel2D(const double* data, int32_t width, int32_t height,
                                   int32_t x, int32_t y,
                                   SubPixelMethod2D method) {
    switch (method) {
        case SubPixelMethod2D::Quadratic:
            return RefineQuadratic2D(data, width, height, x, y);

        case SubPixelMethod2D::Taylor:
            return RefineTaylor2D(data, width, height, x, y);

        case SubPixelMethod2D::Centroid:
            return RefineCentroid2D(data, width, height, x, y);

        case SubPixelMethod2D::BiQuadratic:
            return RefineQuadratic2D(data, width, height, x, y);

        case SubPixelMethod2D::Gaussian2D:
            return RefineTaylor2D(data, width, height, x, y);

        default:
            return RefineQuadratic2D(data, width, height, x, y);
    }
}

// =============================================================================
// Edge Subpixel Refinement Functions
// =============================================================================

SubPixelEdgeResult RefineEdgeSubPixel(const double* profile, size_t size,
                                       int32_t edgeIndex,
                                       EdgeSubPixelMethod method) {
    SubPixelEdgeResult result;

    // Check bounds
    if (edgeIndex < 0 || static_cast<size_t>(edgeIndex) >= size) {
        result.success = false;
        result.confidence = 0.0;
        return result;
    }

    switch (method) {
        case EdgeSubPixelMethod::GradientInterp: {
            // Compute gradient at edge point and neighbors
            if (edgeIndex < 1 || static_cast<size_t>(edgeIndex) >= size - 1) {
                result.success = false;
                result.confidence = 0.0;
                return result;
            }

            double g0 = profile[edgeIndex] - profile[edgeIndex - 1];
            double g1 = profile[edgeIndex + 1] - profile[edgeIndex];

            if (std::abs(g1 - g0) > 1e-10) {
                double offset = RefineEdgeGradient(g0, g1, (g0 + g1) * 0.5);
                result.position = edgeIndex + offset;
                result.gradient = std::abs(g0 + (g1 - g0) * offset);
            } else {
                result.position = static_cast<double>(edgeIndex);
                result.gradient = std::abs(g0);
            }

            result.success = true;
            result.amplitude = std::abs(profile[edgeIndex + 1] - profile[edgeIndex - 1]);
            result.confidence = std::min(1.0, result.amplitude / SUBPIXEL_EDGE_MIN_CONTRAST);
            break;
        }

        case EdgeSubPixelMethod::ZeroCrossing:
            return RefineEdgeZeroCrossing(profile, size, edgeIndex);

        case EdgeSubPixelMethod::ParabolicGradient: {
            // Compute gradient profile
            if (edgeIndex < 1 || static_cast<size_t>(edgeIndex) >= size - 1) {
                result.success = false;
                result.confidence = 0.0;
                return result;
            }

            // Create gradient profile
            std::vector<double> gradient(size - 1);
            for (size_t i = 0; i < size - 1; ++i) {
                gradient[i] = std::abs(profile[i + 1] - profile[i]);
            }

            // Find gradient peak near edgeIndex
            int32_t gradIndex = edgeIndex;
            if (gradIndex > 0 && static_cast<size_t>(gradIndex) < gradient.size()) {
                // Check neighbors
                if (gradIndex > 0 && gradient[gradIndex - 1] > gradient[gradIndex]) {
                    gradIndex--;
                }
                if (static_cast<size_t>(gradIndex) < gradient.size() - 1 &&
                    gradient[gradIndex + 1] > gradient[gradIndex]) {
                    gradIndex++;
                }
            }

            return RefineEdgeParabolic(gradient.data(), gradient.size(), gradIndex);
        }

        case EdgeSubPixelMethod::Moment: {
            // First moment (centroid) of gradient magnitude
            if (edgeIndex < 2 || static_cast<size_t>(edgeIndex) >= size - 2) {
                result.success = false;
                result.confidence = 0.0;
                return result;
            }

            double sumPos = 0.0;
            double sumWeight = 0.0;
            int32_t halfWindow = 2;

            for (int32_t i = edgeIndex - halfWindow; i <= edgeIndex + halfWindow; ++i) {
                if (i >= 0 && static_cast<size_t>(i) < size - 1) {
                    double grad = std::abs(profile[i + 1] - profile[i]);
                    sumPos += grad * (i + 0.5);  // Edge is between i and i+1
                    sumWeight += grad;
                }
            }

            if (sumWeight < 1e-10) {
                result.success = false;
                result.confidence = 0.0;
                return result;
            }

            result.success = true;
            result.position = sumPos / sumWeight;
            result.gradient = sumWeight / (2 * halfWindow + 1);
            result.amplitude = std::abs(profile[edgeIndex + halfWindow] - profile[edgeIndex - halfWindow]);
            result.confidence = std::min(1.0, result.amplitude / SUBPIXEL_EDGE_MIN_CONTRAST);
            break;
        }

        default:
            result.success = false;
            result.confidence = 0.0;
            break;
    }

    return result;
}

SubPixelEdgeResult RefineEdgeZeroCrossing(const double* profile, size_t size,
                                           int32_t edgeIndex) {
    SubPixelEdgeResult result;

    // Need at least 4 points for second derivative zero crossing
    if (edgeIndex < 1 || static_cast<size_t>(edgeIndex) >= size - 2) {
        result.success = false;
        result.confidence = 0.0;
        return result;
    }

    // Compute second derivative at edgeIndex and edgeIndex+1
    double d2_left = profile[edgeIndex - 1] - 2.0 * profile[edgeIndex] + profile[edgeIndex + 1];
    double d2_right = profile[edgeIndex] - 2.0 * profile[edgeIndex + 1] + profile[edgeIndex + 2];

    // Check for sign change
    if (d2_left * d2_right > 0) {
        // No zero crossing between these points, check neighboring intervals
        if (edgeIndex >= 2 && static_cast<size_t>(edgeIndex) < size - 1) {
            double d2_far_left = profile[edgeIndex - 2] - 2.0 * profile[edgeIndex - 1] + profile[edgeIndex];
            if (d2_far_left * d2_left < 0) {
                // Zero crossing is between edgeIndex-1 and edgeIndex
                double offset = -d2_left / (d2_left - d2_far_left);
                result.position = edgeIndex - 1 + offset;
            } else {
                // Use the point with smaller second derivative
                result.position = static_cast<double>(edgeIndex);
            }
        } else {
            result.position = static_cast<double>(edgeIndex);
        }
    } else {
        // Linear interpolation to find zero crossing
        double offset = -d2_left / (d2_right - d2_left);
        result.position = edgeIndex + offset;
    }

    // Compute gradient at edge position
    double fracPos = result.position - std::floor(result.position);
    int32_t baseIdx = static_cast<int32_t>(std::floor(result.position));
    if (baseIdx >= 0 && static_cast<size_t>(baseIdx) < size - 1) {
        double g0 = profile[baseIdx + 1] - profile[baseIdx];
        double g1 = (static_cast<size_t>(baseIdx + 1) < size - 1) ?
                    profile[baseIdx + 2] - profile[baseIdx + 1] : g0;
        result.gradient = std::abs(g0 + (g1 - g0) * fracPos);
    } else {
        result.gradient = 0.0;
    }

    result.success = true;
    result.amplitude = std::abs(profile[std::min(static_cast<size_t>(edgeIndex + 1), size - 1)] -
                                 profile[std::max(edgeIndex - 1, 0)]);
    result.confidence = std::min(1.0, result.amplitude / SUBPIXEL_EDGE_MIN_CONTRAST);

    return result;
}

SubPixelEdgeResult RefineEdgeParabolic(const double* gradient, size_t size,
                                        int32_t peakIndex) {
    SubPixelEdgeResult result;

    // Need 3 points for parabolic fit
    if (peakIndex < 1 || static_cast<size_t>(peakIndex) >= size - 1) {
        result.success = false;
        result.confidence = 0.0;
        return result;
    }

    double g0 = gradient[peakIndex - 1];
    double g1 = gradient[peakIndex];
    double g2 = gradient[peakIndex + 1];

    // Parabolic refinement on gradient magnitude
    double offset = RefineParabolic1D(g0, g1, g2);
    double peakGradient = ParabolicPeakValue(g0, g1, g2, offset);

    result.success = true;
    result.position = peakIndex + 0.5 + offset;  // +0.5 because gradient is between pixels
    result.gradient = peakGradient;
    result.amplitude = 0.0;  // Not directly available from gradient
    result.confidence = std::min(1.0, peakGradient / SUBPIXEL_EDGE_MIN_CONTRAST);

    return result;
}

// =============================================================================
// Template Matching Subpixel Refinement
// =============================================================================

SubPixelResult2D RefineMatchSubPixel(const float* response, int32_t width, int32_t height,
                                      int32_t x, int32_t y,
                                      SubPixelMethod2D method) {
    return RefineSubPixel2D(response, width, height, x, y, method);
}

SubPixelResult2D RefineNCCSubPixel(const float* nccResponse, int32_t width, int32_t height,
                                    int32_t x, int32_t y) {
    // For NCC, quadratic fit is typically sufficient
    // NCC response is in [-1, 1], peaks are typically smooth
    SubPixelResult2D result = RefineQuadratic2D(nccResponse, width, height, x, y);

    // Additional validation for NCC
    if (result.success) {
        // NCC peak value should be close to 1.0 for good matches
        if (result.peakValue < 0.5) {
            result.confidence *= 0.5;
        }
    }

    return result;
}

// =============================================================================
// Angle Subpixel Refinement
// =============================================================================

double RefineAngleSubPixel(const double* responses, size_t numAngles,
                           double angleStep, int32_t bestIndex) {
    if (numAngles < 3 || bestIndex < 0 || static_cast<size_t>(bestIndex) >= numAngles) {
        return bestIndex * angleStep;
    }

    // Handle circular nature of angles
    int32_t prevIndex = (bestIndex == 0) ? static_cast<int32_t>(numAngles - 1) : bestIndex - 1;
    int32_t nextIndex = (static_cast<size_t>(bestIndex) == numAngles - 1) ? 0 : bestIndex + 1;

    double v0 = responses[prevIndex];
    double v1 = responses[bestIndex];
    double v2 = responses[nextIndex];

    // Parabolic refinement
    double offset = RefineParabolic1D(v0, v1, v2);

    // Convert to angle
    double refinedAngle = (bestIndex + offset) * angleStep;

    // Normalize to [0, 2*PI)
    while (refinedAngle < 0) refinedAngle += TWO_PI;
    while (refinedAngle >= TWO_PI) refinedAngle -= TWO_PI;

    return refinedAngle;
}

// =============================================================================
// Utility Functions
// =============================================================================

double ComputeSubPixelConfidence1D(double curvature, double peakValue,
                                    double backgroundValue, double offset) {
    double confidence = 1.0;

    // Factor 1: Curvature should be significant (negative for max, positive for min)
    double absCurvature = std::abs(curvature);
    if (absCurvature < 0.1) {
        confidence *= absCurvature / 0.1;
    }

    // Factor 2: Offset should be small (center is most reliable)
    double absOffset = std::abs(offset);
    confidence *= 1.0 - absOffset;  // Linear decrease from 1.0 at center

    // Factor 3: SNR (signal to noise ratio)
    double signalStrength = std::abs(peakValue - backgroundValue);
    if (signalStrength < 5.0) {
        confidence *= signalStrength / 5.0;
    }

    return Clamp(confidence, 0.0, 1.0);
}

double ComputeSubPixelConfidence2D(double curvatureX, double curvatureY,
                                    double curvatureMixed, double peakValue,
                                    double offsetX, double offsetY) {
    double confidence = 1.0;

    // Factor 1: Curvatures should be significant
    double absCurvX = std::abs(curvatureX);
    double absCurvY = std::abs(curvatureY);
    double minCurv = std::min(absCurvX, absCurvY);
    if (minCurv < 0.1) {
        confidence *= minCurv / 0.1;
    }

    // Factor 1.5: Peak value should be significant (higher peaks are more reliable)
    if (peakValue > 0) {
        // Just reference peakValue to avoid unused parameter warning
        // This factor could be used for more sophisticated confidence computation
        (void)peakValue;
    }

    // Factor 2: Curvatures should have same sign (both negative for max)
    if (curvatureX * curvatureY < 0) {
        confidence *= 0.5;  // Mixed signs indicate saddle tendency
    }

    // Factor 3: Mixed curvature should be small relative to main curvatures
    double avgCurv = (absCurvX + absCurvY) * 0.5;
    if (avgCurv > 0 && std::abs(curvatureMixed) > avgCurv * 0.5) {
        confidence *= 0.8;
    }

    // Factor 4: Offset should be small
    double offsetMag = std::sqrt(offsetX * offsetX + offsetY * offsetY);
    confidence *= 1.0 - offsetMag / std::sqrt(2.0);  // Normalize by max possible offset

    // Factor 5: Check Hessian determinant for true extremum
    double det = curvatureX * curvatureY - curvatureMixed * curvatureMixed;
    if (det <= 0) {
        confidence *= 0.3;  // Saddle point or degenerate
    }

    return Clamp(confidence, 0.0, 1.0);
}

} // namespace Qi::Vision::Internal
