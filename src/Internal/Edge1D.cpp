/**
 * @file Edge1D.cpp
 * @brief 1D edge detection implementation
 */

#include <QiVision/Internal/Edge1D.h>
#include <QiVision/Internal/Gaussian.h>

#include <algorithm>
#include <cmath>

namespace Qi::Vision::Internal {

// ============================================================================
// Profile Gradient
// ============================================================================

void ComputeProfileGradient(const double* profile, double* gradient,
                            size_t length, double sigma) {
    if (length < 2) {
        if (length == 1) gradient[0] = 0.0;
        return;
    }

    // Optional Gaussian smoothing
    std::vector<double> smoothed;
    const double* src = profile;

    if (sigma > 0.0) {
        auto kernel = Gaussian::Kernel1D(sigma);
        int32_t halfK = static_cast<int32_t>(kernel.size()) / 2;

        smoothed.resize(length);
        for (size_t i = 0; i < length; ++i) {
            double sum = 0.0;
            for (int32_t k = -halfK; k <= halfK; ++k) {
                int32_t idx = static_cast<int32_t>(i) + k;
                if (idx < 0) idx = 0;
                if (idx >= static_cast<int32_t>(length)) idx = static_cast<int32_t>(length) - 1;
                sum += profile[idx] * kernel[k + halfK];
            }
            smoothed[i] = sum;
        }
        src = smoothed.data();
    }

    // Central difference for interior points
    for (size_t i = 1; i < length - 1; ++i) {
        gradient[i] = (src[i + 1] - src[i - 1]) * 0.5;
    }

    // Forward/backward difference for endpoints
    gradient[0] = src[1] - src[0];
    gradient[length - 1] = src[length - 1] - src[length - 2];
}

void ComputeProfileGradientSmooth(const double* profile, double* gradient,
                                   size_t length, double sigma) {
    if (length < 2) {
        if (length == 1) gradient[0] = 0.0;
        return;
    }

    // Gaussian derivative kernel
    auto kernel = Gaussian::Derivative1D(sigma);
    int32_t halfK = static_cast<int32_t>(kernel.size()) / 2;

    for (size_t i = 0; i < length; ++i) {
        double sum = 0.0;
        for (int32_t k = -halfK; k <= halfK; ++k) {
            int32_t idx = static_cast<int32_t>(i) + k;
            if (idx < 0) idx = 0;
            if (idx >= static_cast<int32_t>(length)) idx = static_cast<int32_t>(length) - 1;
            sum += profile[idx] * kernel[k + halfK];
        }
        gradient[i] = sum;
    }
}

// ============================================================================
// Subpixel Refinement
// ============================================================================

double RefineEdgeSubpixel(const double* gradient, size_t length, double roughPos) {
    int32_t idx = static_cast<int32_t>(std::round(roughPos));

    // Clamp to valid range
    if (idx <= 0) return 0.0;
    if (idx >= static_cast<int32_t>(length) - 1) return static_cast<double>(length) - 1;

    // Parabolic interpolation: fit parabola through 3 points
    double g0 = std::abs(gradient[idx - 1]);
    double g1 = std::abs(gradient[idx]);
    double g2 = std::abs(gradient[idx + 1]);

    // Parabola: g(x) = a*x^2 + b*x + c at x=-1,0,1
    // Maximum at x = -b/(2a)
    double denom = 2.0 * (g0 - 2.0 * g1 + g2);
    if (std::abs(denom) < 1e-10) {
        return static_cast<double>(idx);
    }

    double offset = (g0 - g2) / denom;

    // Clamp offset to [-0.5, 0.5]
    if (offset < -0.5) offset = -0.5;
    if (offset > 0.5) offset = 0.5;

    return static_cast<double>(idx) + offset;
}

double RefineEdgeZeroCrossing(const double* profile, size_t length,
                               double roughPos, double sigma) {
    // Compute second derivative
    std::vector<double> gradient(length);
    std::vector<double> secondDeriv(length);

    ComputeProfileGradientSmooth(profile, gradient.data(), length, sigma);
    ComputeProfileGradient(gradient.data(), secondDeriv.data(), length, 0.0);

    int32_t idx = static_cast<int32_t>(std::round(roughPos));
    if (idx <= 0) idx = 1;
    if (idx >= static_cast<int32_t>(length) - 1) idx = static_cast<int32_t>(length) - 2;

    // Find zero crossing nearest to roughPos
    for (int32_t i = idx; i > 0 && i < static_cast<int32_t>(length) - 1; ) {
        if (secondDeriv[i] * secondDeriv[i + 1] <= 0) {
            // Linear interpolation for zero crossing
            double t = secondDeriv[i] / (secondDeriv[i] - secondDeriv[i + 1]);
            return static_cast<double>(i) + t;
        }
        if (secondDeriv[i] * secondDeriv[i - 1] <= 0) {
            double t = secondDeriv[i - 1] / (secondDeriv[i - 1] - secondDeriv[i]);
            return static_cast<double>(i - 1) + t;
        }
        break;
    }

    return RefineEdgeSubpixel(gradient.data(), length, roughPos);
}

// ============================================================================
// Peak Finding
// ============================================================================

std::vector<size_t> FindGradientPeaks(const double* gradient, size_t length,
                                       double minAmplitude) {
    std::vector<size_t> peaks;

    for (size_t i = 1; i < length - 1; ++i) {
        double absG = std::abs(gradient[i]);

        // Skip if below threshold
        if (absG < minAmplitude) continue;

        // Check if local maximum of |gradient|
        double absLeft = std::abs(gradient[i - 1]);
        double absRight = std::abs(gradient[i + 1]);

        if (absG >= absLeft && absG > absRight) {
            peaks.push_back(i);
        }
    }

    return peaks;
}

// ============================================================================
// Edge Detection
// ============================================================================

std::vector<Edge1DResult> DetectEdges1D(const double* profile, size_t length,
                                         double minAmplitude,
                                         EdgePolarity polarity,  // Both = Any direction
                                         double sigma) {
    std::vector<Edge1DResult> edges;

    if (length < 3) return edges;

    // Compute gradient
    std::vector<double> gradient(length);
    if (sigma > 0.0) {
        ComputeProfileGradientSmooth(profile, gradient.data(), length, sigma);
    } else {
        ComputeProfileGradient(profile, gradient.data(), length, 0.0);
    }

    // Find peaks in |gradient|
    auto peaks = FindGradientPeaks(gradient.data(), length, minAmplitude);

    for (size_t idx : peaks) {
        EdgePolarity edgePolarity = ClassifyPolarity(gradient[idx]);

        // Filter by polarity
        if (!MatchesPolarity(edgePolarity, polarity)) {
            continue;
        }

        // Refine to subpixel
        double subpixelPos = RefineEdgeSubpixel(gradient.data(), length,
                                                 static_cast<double>(idx));

        edges.emplace_back(subpixelPos, std::abs(gradient[idx]), edgePolarity);
    }

    // Sort by position
    std::sort(edges.begin(), edges.end(),
              [](const Edge1DResult& a, const Edge1DResult& b) {
                  return a.position < b.position;
              });

    return edges;
}

Edge1DResult DetectSingleEdge1D(const double* profile, size_t length,
                                 double minAmplitude,
                                 EdgePolarity polarity,
                                 EdgeSelect select,
                                 double sigma,
                                 bool& found) {
    auto edges = DetectEdges1D(profile, length, minAmplitude, polarity, sigma);

    found = !edges.empty();
    if (!found) {
        return Edge1DResult();
    }

    switch (select) {
        case EdgeSelect::First:
            return edges.front();

        case EdgeSelect::Last:
            return edges.back();

        case EdgeSelect::Strongest:
        case EdgeSelect::Best:
            return *std::max_element(edges.begin(), edges.end(),
                                      [](const Edge1DResult& a, const Edge1DResult& b) {
                                          return a.amplitude < b.amplitude;
                                      });

        case EdgeSelect::All:
        default:
            return edges.front();
    }
}

// ============================================================================
// Edge Pair Detection
// ============================================================================

std::vector<EdgePairResult> DetectEdgePairs1D(const double* profile, size_t length,
                                               double minAmplitude,
                                               EdgePairSelect selection,
                                               double sigma) {
    std::vector<EdgePairResult> pairs;

    // Detect all edges
    auto allEdges = DetectEdges1D(profile, length, minAmplitude, EdgePolarity::Both, sigma);

    if (allEdges.size() < 2) return pairs;

    // Find positive and negative edges
    std::vector<Edge1DResult> positive, negative;
    for (const auto& edge : allEdges) {
        if (edge.polarity == EdgePolarity::Positive) {
            positive.push_back(edge);
        } else {
            negative.push_back(edge);
        }
    }

    if (positive.empty() || negative.empty()) return pairs;

    // Create pairs based on selection mode
    switch (selection) {
        case EdgePairSelect::All:
            // All valid pairs (positive followed by negative)
            for (const auto& pos : positive) {
                for (const auto& neg : negative) {
                    if (neg.position > pos.position) {
                        pairs.emplace_back(pos, neg);
                    }
                }
            }
            break;

        case EdgePairSelect::FirstLast:
            // First positive, last negative
            {
                auto& firstPos = positive.front();
                auto& lastNeg = negative.back();
                if (lastNeg.position > firstPos.position) {
                    pairs.emplace_back(firstPos, lastNeg);
                }
            }
            break;

        case EdgePairSelect::BestPair:
            // Best matching pair (highest combined amplitude)
            {
                EdgePairResult best;
                double bestScore = -1.0;
                for (const auto& pos : positive) {
                    for (const auto& neg : negative) {
                        if (neg.position > pos.position) {
                            double score = pos.amplitude + neg.amplitude;
                            if (score > bestScore) {
                                bestScore = score;
                                best = EdgePairResult(pos, neg);
                            }
                        }
                    }
                }
                if (bestScore > 0.0) {
                    pairs.push_back(best);
                }
            }
            break;

        case EdgePairSelect::Closest:
            // Pair with smallest distance
            {
                EdgePairResult closest;
                double minDist = std::numeric_limits<double>::max();
                for (const auto& pos : positive) {
                    for (const auto& neg : negative) {
                        if (neg.position > pos.position) {
                            double dist = neg.position - pos.position;
                            if (dist < minDist) {
                                minDist = dist;
                                closest = EdgePairResult(pos, neg);
                            }
                        }
                    }
                }
                if (minDist < std::numeric_limits<double>::max()) {
                    pairs.push_back(closest);
                }
            }
            break;
    }

    return pairs;
}

EdgePairResult DetectSinglePair1D(const double* profile, size_t length,
                                   double minAmplitude,
                                   EdgePairSelect selection,
                                   double sigma,
                                   bool& found) {
    auto pairs = DetectEdgePairs1D(profile, length, minAmplitude, selection, sigma);

    found = !pairs.empty();
    if (!found) {
        return EdgePairResult();
    }

    return pairs.front();
}

// Explicit template instantiations
template void ExtractProfile<uint8_t>(const uint8_t*, int32_t, int32_t,
                                       double, double, double, double,
                                       std::vector<double>&, size_t, InterpolationMethod);
template void ExtractProfile<float>(const float*, int32_t, int32_t,
                                     double, double, double, double,
                                     std::vector<double>&, size_t, InterpolationMethod);

template void ExtractPerpendicularProfile<uint8_t>(const uint8_t*, int32_t, int32_t,
                                                    double, double, double, double,
                                                    std::vector<double>&, size_t, InterpolationMethod);
template void ExtractPerpendicularProfile<float>(const float*, int32_t, int32_t,
                                                  double, double, double, double,
                                                  std::vector<double>&, size_t, InterpolationMethod);

} // namespace Qi::Vision::Internal
