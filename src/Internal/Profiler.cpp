/**
 * @file Profiler.cpp
 * @brief 1D profile extraction implementation
 */

#include <QiVision/Internal/Profiler.h>
#include <QiVision/Internal/Gaussian.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace Qi::Vision::Internal {

// ============================================================================
// Line Profile Extraction
// ============================================================================

Profile1D ExtractLineProfile(const QImage& image,
                              double x0, double y0,
                              double x1, double y1,
                              size_t numSamples,
                              InterpolationMethod method) {
    if (image.Empty()) {
        return Profile1D();
    }

    // Handle different pixel types
    switch (image.Type()) {
        case PixelType::UInt8: {
            const uint8_t* data = static_cast<const uint8_t*>(image.Data());
            return ExtractLineProfile(data, image.Width(), image.Height(),
                                       x0, y0, x1, y1, numSamples, method);
        }
        case PixelType::UInt16: {
            const uint16_t* data = static_cast<const uint16_t*>(image.Data());
            return ExtractLineProfile(data, image.Width(), image.Height(),
                                       x0, y0, x1, y1, numSamples, method);
        }
        case PixelType::Float32: {
            const float* data = static_cast<const float*>(image.Data());
            return ExtractLineProfile(data, image.Width(), image.Height(),
                                       x0, y0, x1, y1, numSamples, method);
        }
        default:
            return Profile1D();
    }
}

std::vector<Profile1D> ExtractParallelProfiles(
    const QImage& image,
    double x0, double y0,
    double x1, double y1,
    double spacing,
    int32_t numLines,
    size_t numSamples,
    InterpolationMethod method) {

    std::vector<Profile1D> profiles;
    if (image.Empty() || numLines <= 0) {
        return profiles;
    }

    profiles.reserve(numLines);

    // Compute line direction and perpendicular
    double dx = x1 - x0;
    double dy = y1 - y0;
    double length = std::sqrt(dx * dx + dy * dy);
    if (length < 1e-10) {
        return profiles;
    }

    double angle = std::atan2(dy, dx);
    double perpAngle = angle - M_PI / 2.0;
    double perpDx = std::cos(perpAngle);
    double perpDy = std::sin(perpAngle);

    // Extract parallel lines
    double halfExtent = (numLines - 1) * spacing / 2.0;

    for (int32_t i = 0; i < numLines; ++i) {
        double offset = -halfExtent + i * spacing;

        double ox0 = x0 + offset * perpDx;
        double oy0 = y0 + offset * perpDy;
        double ox1 = x1 + offset * perpDx;
        double oy1 = y1 + offset * perpDy;

        profiles.push_back(ExtractLineProfile(image, ox0, oy0, ox1, oy1,
                                               numSamples, method));
    }

    return profiles;
}

// ============================================================================
// Rectangle Profile Extraction
// ============================================================================

Profile1D ExtractRectProfile(const QImage& image, const RectProfileParams& params) {
    if (image.Empty()) {
        return Profile1D();
    }

    switch (image.Type()) {
        case PixelType::UInt8: {
            const uint8_t* data = static_cast<const uint8_t*>(image.Data());
            return ExtractRectProfile(data, image.Width(), image.Height(), params);
        }
        case PixelType::UInt16: {
            const uint16_t* data = static_cast<const uint16_t*>(image.Data());
            return ExtractRectProfile(data, image.Width(), image.Height(), params);
        }
        case PixelType::Float32: {
            const float* data = static_cast<const float*>(image.Data());
            return ExtractRectProfile(data, image.Width(), image.Height(), params);
        }
        default:
            return Profile1D();
    }
}

// ============================================================================
// Arc Profile Extraction
// ============================================================================

Profile1D ExtractArcProfile(const QImage& image, const ArcProfileParams& params) {
    if (image.Empty()) {
        return Profile1D();
    }

    switch (image.Type()) {
        case PixelType::UInt8: {
            const uint8_t* data = static_cast<const uint8_t*>(image.Data());
            return ExtractArcProfile(data, image.Width(), image.Height(), params);
        }
        case PixelType::UInt16: {
            const uint16_t* data = static_cast<const uint16_t*>(image.Data());
            return ExtractArcProfile(data, image.Width(), image.Height(), params);
        }
        case PixelType::Float32: {
            const float* data = static_cast<const float*>(image.Data());
            return ExtractArcProfile(data, image.Width(), image.Height(), params);
        }
        default:
            return Profile1D();
    }
}

// ============================================================================
// Annular Profile Extraction
// ============================================================================

Profile1D ExtractAnnularProfile(const QImage& image, const AnnularProfileParams& params) {
    if (image.Empty()) {
        return Profile1D();
    }

    switch (image.Type()) {
        case PixelType::UInt8: {
            const uint8_t* data = static_cast<const uint8_t*>(image.Data());
            return ExtractAnnularProfile(data, image.Width(), image.Height(), params);
        }
        case PixelType::UInt16: {
            const uint16_t* data = static_cast<const uint16_t*>(image.Data());
            return ExtractAnnularProfile(data, image.Width(), image.Height(), params);
        }
        case PixelType::Float32: {
            const float* data = static_cast<const float*>(image.Data());
            return ExtractAnnularProfile(data, image.Width(), image.Height(), params);
        }
        default:
            return Profile1D();
    }
}

// ============================================================================
// Profile Statistics
// ============================================================================

ProfileStats ComputeProfileStats(const Profile1D& profile) {
    return ComputeProfileStats(profile.data.data(), profile.data.size());
}

ProfileStats ComputeProfileStats(const double* data, size_t length) {
    ProfileStats stats;
    stats.count = length;

    if (length == 0 || data == nullptr) {
        return stats;
    }

    // Find min, max, sum
    stats.min = data[0];
    stats.max = data[0];
    stats.minIdx = 0;
    stats.maxIdx = 0;
    stats.sum = 0;

    for (size_t i = 0; i < length; ++i) {
        double val = data[i];
        stats.sum += val;

        if (val < stats.min) {
            stats.min = val;
            stats.minIdx = i;
        }
        if (val > stats.max) {
            stats.max = val;
            stats.maxIdx = i;
        }
    }

    stats.mean = stats.sum / length;

    // Compute standard deviation
    double sumSq = 0;
    for (size_t i = 0; i < length; ++i) {
        double diff = data[i] - stats.mean;
        sumSq += diff * diff;
    }
    stats.stddev = (length > 1) ? std::sqrt(sumSq / (length - 1)) : 0.0;

    return stats;
}

// ============================================================================
// Profile Operations
// ============================================================================

void NormalizeProfile(Profile1D& profile, ProfileNormalize mode) {
    if (profile.Empty()) return;

    ProfileStats stats = ComputeProfileStats(profile);

    switch (mode) {
        case ProfileNormalize::None:
            break;

        case ProfileNormalize::MinMax: {
            double range = stats.max - stats.min;
            if (range > 1e-10) {
                for (double& v : profile.data) {
                    v = (v - stats.min) / range;
                }
            }
            break;
        }

        case ProfileNormalize::ZScore: {
            if (stats.stddev > 1e-10) {
                for (double& v : profile.data) {
                    v = (v - stats.mean) / stats.stddev;
                }
            }
            break;
        }

        case ProfileNormalize::Sum: {
            if (std::abs(stats.sum) > 1e-10) {
                for (double& v : profile.data) {
                    v /= stats.sum;
                }
            }
            break;
        }
    }
}

void SmoothProfile(Profile1D& profile, double sigma) {
    if (profile.Empty() || sigma <= 0) return;

    // Generate Gaussian kernel
    auto kernel = Gaussian::Kernel1D(sigma);
    int32_t kernelSize = static_cast<int32_t>(kernel.size());
    int32_t halfKernel = kernelSize / 2;

    size_t length = profile.data.size();
    std::vector<double> result(length);

    // Convolve with reflect border handling
    for (size_t i = 0; i < length; ++i) {
        double sum = 0;
        for (int32_t k = 0; k < kernelSize; ++k) {
            int32_t idx = static_cast<int32_t>(i) + k - halfKernel;

            // Reflect border
            if (idx < 0) idx = -idx;
            if (idx >= static_cast<int32_t>(length)) {
                idx = 2 * static_cast<int32_t>(length) - idx - 2;
            }
            idx = std::max(0, std::min(idx, static_cast<int32_t>(length) - 1));

            sum += profile.data[idx] * kernel[k];
        }
        result[i] = sum;
    }

    profile.data = std::move(result);
}

Profile1D ComputeProfileGradient(const Profile1D& profile, double sigma) {
    Profile1D result;
    if (profile.Size() < 2) return result;

    result = profile;  // Copy metadata

    size_t length = profile.data.size();
    result.data.resize(length);

    if (sigma > 0) {
        // Use Gaussian derivative kernel
        auto kernel = Gaussian::Derivative1D(sigma);
        int32_t kernelSize = static_cast<int32_t>(kernel.size());
        int32_t halfKernel = kernelSize / 2;

        for (size_t i = 0; i < length; ++i) {
            double sum = 0;
            for (int32_t k = 0; k < kernelSize; ++k) {
                int32_t idx = static_cast<int32_t>(i) + k - halfKernel;

                // Reflect border
                if (idx < 0) idx = -idx;
                if (idx >= static_cast<int32_t>(length)) {
                    idx = 2 * static_cast<int32_t>(length) - idx - 2;
                }
                idx = std::max(0, std::min(idx, static_cast<int32_t>(length) - 1));

                sum += profile.data[idx] * kernel[k];
            }
            result.data[i] = sum;
        }
    } else {
        // Simple central difference
        result.data[0] = profile.data[1] - profile.data[0];
        for (size_t i = 1; i < length - 1; ++i) {
            result.data[i] = (profile.data[i + 1] - profile.data[i - 1]) / 2.0;
        }
        result.data[length - 1] = profile.data[length - 1] - profile.data[length - 2];
    }

    return result;
}

Profile1D ComputeProfileSecondDerivative(const Profile1D& profile, double sigma) {
    Profile1D result;
    if (profile.Size() < 3) return result;

    result = profile;  // Copy metadata

    size_t length = profile.data.size();
    result.data.resize(length);

    if (sigma > 0) {
        // Use Gaussian second derivative kernel
        auto kernel = Gaussian::SecondDerivative1D(sigma);
        int32_t kernelSize = static_cast<int32_t>(kernel.size());
        int32_t halfKernel = kernelSize / 2;

        for (size_t i = 0; i < length; ++i) {
            double sum = 0;
            for (int32_t k = 0; k < kernelSize; ++k) {
                int32_t idx = static_cast<int32_t>(i) + k - halfKernel;

                // Reflect border
                if (idx < 0) idx = -idx;
                if (idx >= static_cast<int32_t>(length)) {
                    idx = 2 * static_cast<int32_t>(length) - idx - 2;
                }
                idx = std::max(0, std::min(idx, static_cast<int32_t>(length) - 1));

                sum += profile.data[idx] * kernel[k];
            }
            result.data[i] = sum;
        }
    } else {
        // Simple second difference
        result.data[0] = profile.data[2] - 2 * profile.data[1] + profile.data[0];
        for (size_t i = 1; i < length - 1; ++i) {
            result.data[i] = profile.data[i + 1] - 2 * profile.data[i] + profile.data[i - 1];
        }
        result.data[length - 1] = profile.data[length - 1] - 2 * profile.data[length - 2] +
                                   profile.data[length - 3];
    }

    return result;
}

Profile1D ResampleProfile(const Profile1D& profile, size_t newSize) {
    Profile1D result;
    if (profile.Empty() || newSize < 2) return result;

    result.startX = profile.startX;
    result.startY = profile.startY;
    result.endX = profile.endX;
    result.endY = profile.endY;
    result.angle = profile.angle;
    result.data.resize(newSize);
    result.stepSize = profile.Length() / (newSize - 1);

    size_t oldSize = profile.data.size();

    for (size_t i = 0; i < newSize; ++i) {
        double t = static_cast<double>(i) / (newSize - 1);
        double srcIdx = t * (oldSize - 1);

        int32_t idx0 = static_cast<int32_t>(srcIdx);
        int32_t idx1 = idx0 + 1;
        double frac = srcIdx - idx0;

        if (idx1 >= static_cast<int32_t>(oldSize)) {
            result.data[i] = profile.data[oldSize - 1];
        } else {
            result.data[i] = profile.data[idx0] * (1.0 - frac) +
                              profile.data[idx1] * frac;
        }
    }

    return result;
}

Profile1D CombineProfiles(const std::vector<Profile1D>& profiles, ProfileMethod method) {
    Profile1D result;
    if (profiles.empty()) return result;

    // Use first profile as template
    result = profiles[0];
    size_t length = result.data.size();

    if (profiles.size() == 1 || method == ProfileMethod::Single) {
        return result;
    }

    // Check all profiles have same size
    for (const auto& p : profiles) {
        if (p.data.size() != length) {
            // Different sizes - just return first
            return result;
        }
    }

    // Combine based on method
    for (size_t i = 0; i < length; ++i) {
        std::vector<double> values;
        values.reserve(profiles.size());

        for (const auto& p : profiles) {
            values.push_back(p.data[i]);
        }

        switch (method) {
            case ProfileMethod::Average: {
                double sum = 0;
                for (double v : values) sum += v;
                result.data[i] = sum / values.size();
                break;
            }

            case ProfileMethod::Maximum: {
                double maxVal = values[0];
                for (double v : values) if (v > maxVal) maxVal = v;
                result.data[i] = maxVal;
                break;
            }

            case ProfileMethod::Minimum: {
                double minVal = values[0];
                for (double v : values) if (v < minVal) minVal = v;
                result.data[i] = minVal;
                break;
            }

            case ProfileMethod::Median: {
                std::sort(values.begin(), values.end());
                size_t mid = values.size() / 2;
                result.data[i] = (values.size() % 2 == 0)
                    ? (values[mid - 1] + values[mid]) / 2.0
                    : values[mid];
                break;
            }

            default:
                result.data[i] = values[0];
                break;
        }
    }

    return result;
}

// ============================================================================
// Profile Projection
// ============================================================================

Profile1D ProjectRegion(const QImage& image, const Rect2i& rect,
                         bool horizontal, ProfileMethod method) {
    Profile1D result;
    if (image.Empty()) return result;

    // Clamp rect to image bounds
    int32_t x0 = std::max(0, rect.x);
    int32_t y0 = std::max(0, rect.y);
    int32_t x1 = std::min(image.Width(), rect.x + rect.width);
    int32_t y1 = std::min(image.Height(), rect.y + rect.height);

    if (x1 <= x0 || y1 <= y0) return result;

    int32_t w = x1 - x0;
    int32_t h = y1 - y0;

    if (horizontal) {
        // Project horizontally (result has width samples)
        result.data.resize(w);
        result.startX = x0;
        result.startY = (y0 + y1) / 2.0;
        result.endX = x1;
        result.endY = result.startY;
        result.angle = 0;
        result.stepSize = 1.0;

        for (int32_t x = 0; x < w; ++x) {
            std::vector<double> values;
            values.reserve(h);

            for (int32_t y = 0; y < h; ++y) {
                double val = 0;
                switch (image.Type()) {
                    case PixelType::UInt8:
                        val = static_cast<const uint8_t*>(image.Data())
                            [(y0 + y) * image.Width() + (x0 + x)];
                        break;
                    case PixelType::UInt16:
                        val = static_cast<const uint16_t*>(image.Data())
                            [(y0 + y) * image.Width() + (x0 + x)];
                        break;
                    case PixelType::Float32:
                        val = static_cast<const float*>(image.Data())
                            [(y0 + y) * image.Width() + (x0 + x)];
                        break;
                    default:
                        break;
                }
                values.push_back(val);
            }

            // Combine values
            switch (method) {
                case ProfileMethod::Average: {
                    double sum = 0;
                    for (double v : values) sum += v;
                    result.data[x] = sum / values.size();
                    break;
                }
                case ProfileMethod::Maximum: {
                    double maxVal = values[0];
                    for (double v : values) if (v > maxVal) maxVal = v;
                    result.data[x] = maxVal;
                    break;
                }
                case ProfileMethod::Minimum: {
                    double minVal = values[0];
                    for (double v : values) if (v < minVal) minVal = v;
                    result.data[x] = minVal;
                    break;
                }
                case ProfileMethod::Median: {
                    std::sort(values.begin(), values.end());
                    size_t mid = values.size() / 2;
                    result.data[x] = (values.size() % 2 == 0)
                        ? (values[mid - 1] + values[mid]) / 2.0
                        : values[mid];
                    break;
                }
                default:
                    result.data[x] = values[0];
                    break;
            }
        }
    } else {
        // Project vertically (result has height samples)
        result.data.resize(h);
        result.startX = (x0 + x1) / 2.0;
        result.startY = y0;
        result.endX = result.startX;
        result.endY = y1;
        result.angle = M_PI / 2.0;
        result.stepSize = 1.0;

        for (int32_t y = 0; y < h; ++y) {
            std::vector<double> values;
            values.reserve(w);

            for (int32_t x = 0; x < w; ++x) {
                double val = 0;
                switch (image.Type()) {
                    case PixelType::UInt8:
                        val = static_cast<const uint8_t*>(image.Data())
                            [(y0 + y) * image.Width() + (x0 + x)];
                        break;
                    case PixelType::UInt16:
                        val = static_cast<const uint16_t*>(image.Data())
                            [(y0 + y) * image.Width() + (x0 + x)];
                        break;
                    case PixelType::Float32:
                        val = static_cast<const float*>(image.Data())
                            [(y0 + y) * image.Width() + (x0 + x)];
                        break;
                    default:
                        break;
                }
                values.push_back(val);
            }

            // Combine values
            switch (method) {
                case ProfileMethod::Average: {
                    double sum = 0;
                    for (double v : values) sum += v;
                    result.data[y] = sum / values.size();
                    break;
                }
                case ProfileMethod::Maximum: {
                    double maxVal = values[0];
                    for (double v : values) if (v > maxVal) maxVal = v;
                    result.data[y] = maxVal;
                    break;
                }
                case ProfileMethod::Minimum: {
                    double minVal = values[0];
                    for (double v : values) if (v < minVal) minVal = v;
                    result.data[y] = minVal;
                    break;
                }
                case ProfileMethod::Median: {
                    std::sort(values.begin(), values.end());
                    size_t mid = values.size() / 2;
                    result.data[y] = (values.size() % 2 == 0)
                        ? (values[mid - 1] + values[mid]) / 2.0
                        : values[mid];
                    break;
                }
                default:
                    result.data[y] = values[0];
                    break;
            }
        }
    }

    return result;
}

Profile1D ProjectRotatedRect(const QImage& image,
                              Point2d center, double width, double height,
                              double angle, bool alongWidth,
                              ProfileMethod method) {
    Profile1D result;
    if (image.Empty()) return result;

    // Determine profile direction and perpendicular extent
    double profileLen, perpExtent;
    double profileAngle;

    if (alongWidth) {
        profileLen = width;
        perpExtent = height;
        profileAngle = angle;
    } else {
        profileLen = height;
        perpExtent = width;
        profileAngle = angle + M_PI / 2.0;
    }

    // Use RectProfileParams for extraction
    RectProfileParams params;
    params.centerX = center.x;
    params.centerY = center.y;
    params.length = profileLen;
    params.width = perpExtent;
    params.angle = profileAngle;
    params.numLines = static_cast<int32_t>(std::ceil(perpExtent));
    params.method = method;

    return ExtractRectProfile(image, params);
}

// ============================================================================
// Utility Functions
// ============================================================================

void ComputeLineSamples(double x0, double y0, double x1, double y1,
                        size_t numSamples,
                        std::vector<double>& xCoords,
                        std::vector<double>& yCoords) {
    xCoords.resize(numSamples);
    yCoords.resize(numSamples);

    if (numSamples == 0) return;

    if (numSamples == 1) {
        xCoords[0] = (x0 + x1) / 2.0;
        yCoords[0] = (y0 + y1) / 2.0;
        return;
    }

    double dx = (x1 - x0) / (numSamples - 1);
    double dy = (y1 - y0) / (numSamples - 1);

    for (size_t i = 0; i < numSamples; ++i) {
        xCoords[i] = x0 + i * dx;
        yCoords[i] = y0 + i * dy;
    }
}

void ComputeArcSamples(double cx, double cy, double radius,
                       double startAngle, double endAngle,
                       size_t numSamples,
                       std::vector<double>& xCoords,
                       std::vector<double>& yCoords) {
    xCoords.resize(numSamples);
    yCoords.resize(numSamples);

    if (numSamples == 0) return;

    if (numSamples == 1) {
        double midAngle = (startAngle + endAngle) / 2.0;
        xCoords[0] = cx + radius * std::cos(midAngle);
        yCoords[0] = cy + radius * std::sin(midAngle);
        return;
    }

    double angleStep = (endAngle - startAngle) / (numSamples - 1);

    for (size_t i = 0; i < numSamples; ++i) {
        double angle = startAngle + i * angleStep;
        xCoords[i] = cx + radius * std::cos(angle);
        yCoords[i] = cy + radius * std::sin(angle);
    }
}

} // namespace Qi::Vision::Internal
