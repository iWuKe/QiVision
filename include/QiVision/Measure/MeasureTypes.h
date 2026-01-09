#pragma once

/**
 * @file MeasureTypes.h
 * @brief Measure module type definitions
 *
 * Provides:
 * - Edge transition types
 * - Measurement parameters
 * - Edge and pair result structures
 * - Score and quality metrics
 */

#include <QiVision/Core/Types.h>

#include <cstdint>
#include <vector>

namespace Qi::Vision::Measure {

// =============================================================================
// Constants
// =============================================================================

/// Default minimum edge amplitude
constexpr double DEFAULT_MIN_AMPLITUDE = 20.0;

/// Default Gaussian smoothing sigma
constexpr double DEFAULT_SIGMA = 1.0;

/// Default number of perpendicular lines for averaging
constexpr int32_t DEFAULT_NUM_LINES = 10;

/// Default interpolation samples per pixel
constexpr double DEFAULT_SAMPLES_PER_PIXEL = 1.0;

/// Maximum number of edges to return
constexpr int32_t MAX_EDGES = 1000;

// =============================================================================
// Enumerations
// =============================================================================

/**
 * @brief Edge transition type (polarity)
 */
enum class EdgeTransition {
    Positive,       ///< Dark to light (rising edge)
    Negative,       ///< Light to dark (falling edge)
    All             ///< Both transitions
};

/**
 * @brief Edge selection mode
 */
enum class EdgeSelectMode {
    All,            ///< Return all detected edges
    First,          ///< First edge only (along profile direction)
    Last,           ///< Last edge only
    Strongest,      ///< Edge with highest amplitude
    Weakest         ///< Edge with lowest amplitude (above threshold)
};

/**
 * @brief Edge pair selection mode
 */
enum class PairSelectMode {
    All,            ///< All valid pairs
    First,          ///< First pair only
    Last,           ///< Last pair only
    Strongest,      ///< Pair with highest combined amplitude
    Widest,         ///< Pair with largest distance
    Narrowest       ///< Pair with smallest distance
};

/**
 * @brief Interpolation method for profile extraction
 */
enum class ProfileInterpolation {
    Nearest,        ///< Nearest neighbor (fast)
    Bilinear,       ///< Bilinear (default, good balance)
    Bicubic         ///< Bicubic (highest accuracy)
};

/**
 * @brief Score computation method for fuzzy measurement
 */
enum class ScoreMethod {
    Amplitude,      ///< Based on edge amplitude only
    AmplitudeScore, ///< Amplitude normalized by max possible
    Contrast,       ///< Local contrast ratio
    FuzzyScore      ///< Combined fuzzy logic score
};

// =============================================================================
// Parameter Structures
// =============================================================================

/**
 * @brief Common measurement parameters
 */
struct MeasureParams {
    // Edge detection parameters
    double sigma = DEFAULT_SIGMA;                   ///< Gaussian smoothing sigma
    double minAmplitude = DEFAULT_MIN_AMPLITUDE;    ///< Minimum edge amplitude
    EdgeTransition transition = EdgeTransition::All; ///< Edge polarity filter

    // Profile parameters
    int32_t numLines = DEFAULT_NUM_LINES;           ///< Number of perpendicular lines
    double samplesPerPixel = DEFAULT_SAMPLES_PER_PIXEL; ///< Sampling density
    ProfileInterpolation interp = ProfileInterpolation::Bilinear;

    // Selection parameters
    EdgeSelectMode selectMode = EdgeSelectMode::All;
    int32_t maxEdges = MAX_EDGES;                   ///< Maximum edges to return

    // Score parameters
    ScoreMethod scoreMethod = ScoreMethod::Amplitude;

    // Builder pattern for fluent configuration
    MeasureParams& SetSigma(double s) { sigma = s; return *this; }
    MeasureParams& SetMinAmplitude(double a) { minAmplitude = a; return *this; }
    MeasureParams& SetTransition(EdgeTransition t) { transition = t; return *this; }
    MeasureParams& SetNumLines(int32_t n) { numLines = n; return *this; }
    MeasureParams& SetSamplesPerPixel(double s) { samplesPerPixel = s; return *this; }
    MeasureParams& SetInterpolation(ProfileInterpolation i) { interp = i; return *this; }
    MeasureParams& SetSelectMode(EdgeSelectMode m) { selectMode = m; return *this; }
    MeasureParams& SetMaxEdges(int32_t n) { maxEdges = n; return *this; }
};

/**
 * @brief Parameters for edge pair (width) measurement
 */
struct PairParams : public MeasureParams {
    // Pair-specific parameters
    EdgeTransition firstTransition = EdgeTransition::Positive;   ///< First edge polarity
    EdgeTransition secondTransition = EdgeTransition::Negative;  ///< Second edge polarity

    double minWidth = 0.0;          ///< Minimum pair width (pixels)
    double maxWidth = 1e9;          ///< Maximum pair width (pixels)

    PairSelectMode pairSelectMode = PairSelectMode::All;
    int32_t maxPairs = MAX_EDGES;   ///< Maximum pairs to return

    PairParams& SetFirstTransition(EdgeTransition t) { firstTransition = t; return *this; }
    PairParams& SetSecondTransition(EdgeTransition t) { secondTransition = t; return *this; }
    PairParams& SetWidthRange(double minW, double maxW) {
        minWidth = minW; maxWidth = maxW; return *this;
    }
    PairParams& SetPairSelectMode(PairSelectMode m) { pairSelectMode = m; return *this; }
    PairParams& SetMaxPairs(int32_t n) { maxPairs = n; return *this; }
};

/**
 * @brief Fuzzy measurement parameters (extended)
 */
struct FuzzyParams : public MeasureParams {
    // Fuzzy-specific parameters
    double fuzzyThresholdLow = 0.3;     ///< Lower amplitude threshold ratio
    double fuzzyThresholdHigh = 0.8;    ///< Upper amplitude threshold ratio
    double minScore = 0.5;              ///< Minimum score threshold

    bool computeScore = true;           ///< Whether to compute detailed score
    bool useAdaptiveThreshold = false;  ///< Adapt threshold based on local contrast

    FuzzyParams& SetFuzzyThresholds(double low, double high) {
        fuzzyThresholdLow = low; fuzzyThresholdHigh = high; return *this;
    }
    FuzzyParams& SetMinScore(double s) { minScore = s; return *this; }
    FuzzyParams& SetAdaptiveThreshold(bool use) { useAdaptiveThreshold = use; return *this; }
};

// =============================================================================
// Result Structures
// =============================================================================

/**
 * @brief Single edge measurement result
 */
struct EdgeResult {
    // Position in image coordinates (subpixel)
    double row = 0.0;           ///< Y coordinate (row)
    double column = 0.0;        ///< X coordinate (column)

    // Position along profile
    double profilePosition = 0.0;   ///< Position along measurement profile [0, length]

    // Edge properties
    double amplitude = 0.0;     ///< Edge amplitude (gradient magnitude)
    EdgeTransition transition = EdgeTransition::Positive; ///< Edge polarity

    // Quality metrics
    double score = 0.0;         ///< Quality score [0, 1] (for fuzzy)
    double confidence = 0.0;    ///< Detection confidence [0, 1]

    // Optional: edge direction
    double angle = 0.0;         ///< Edge normal angle (radians)

    /// Check if result is valid
    bool IsValid() const { return amplitude > 0 && confidence > 0; }

    /// Get position as Point2d
    Point2d Position() const { return {column, row}; }
};

/**
 * @brief Edge pair (width) measurement result
 */
struct PairResult {
    EdgeResult first;           ///< First edge of pair
    EdgeResult second;          ///< Second edge of pair

    // Pair metrics
    double width = 0.0;         ///< Distance between edges (pixels)
    double centerRow = 0.0;     ///< Center Y coordinate
    double centerColumn = 0.0;  ///< Center X coordinate

    // Quality
    double score = 0.0;         ///< Combined pair score [0, 1]
    double symmetry = 0.0;      ///< Amplitude symmetry [0, 1]

    /// Check if result is valid
    bool IsValid() const {
        return first.IsValid() && second.IsValid() && width > 0;
    }

    /// Get center position
    Point2d Center() const { return {centerColumn, centerRow}; }
};

/**
 * @brief Measurement statistics
 */
struct MeasureStats {
    int32_t numEdgesFound = 0;      ///< Total edges detected
    int32_t numEdgesReturned = 0;   ///< Edges after filtering

    double meanAmplitude = 0.0;     ///< Mean edge amplitude
    double maxAmplitude = 0.0;      ///< Maximum amplitude
    double minAmplitude = 0.0;      ///< Minimum amplitude (of detected)

    double profileContrast = 0.0;   ///< Overall profile contrast
    double signalNoiseRatio = 0.0;  ///< Estimated SNR
};

// =============================================================================
// Conversion Functions
// =============================================================================

/**
 * @brief Convert EdgeTransition to Internal EdgePolarity
 */
inline EdgePolarity ToEdgePolarity(EdgeTransition t) {
    switch (t) {
        case EdgeTransition::Positive: return EdgePolarity::Positive;
        case EdgeTransition::Negative: return EdgePolarity::Negative;
        case EdgeTransition::All:      return EdgePolarity::Both;
    }
    return EdgePolarity::Both;
}

/**
 * @brief Convert Internal EdgePolarity to EdgeTransition
 */
inline EdgeTransition FromEdgePolarity(EdgePolarity p) {
    switch (p) {
        case EdgePolarity::Positive: return EdgeTransition::Positive;
        case EdgePolarity::Negative: return EdgeTransition::Negative;
        case EdgePolarity::Both:     return EdgeTransition::All;
    }
    return EdgeTransition::All;
}

} // namespace Qi::Vision::Measure
