#pragma once

/**
 * @file MeasureHandle.h
 * @brief Measurement region handles (Rectangle, Arc, Concentric)
 *
 * Provides:
 * - Pre-computed sampling geometry for efficient repeated measurements
 * - Rectangle, Arc, and Concentric Circle handles
 *
 * Design:
 * - Handles are lightweight and can be copied
 * - Sampling points are computed once at creation
 * - Thread-safe for concurrent reads (const methods)
 */

#include <QiVision/Core/Types.h>
#include <QiVision/Measure/MeasureTypes.h>
#include <QiVision/Core/Export.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace Qi::Vision::Measure {

// =============================================================================
// Handle Type Enumeration
// =============================================================================

/**
 * @brief Measure handle type enumeration
 */
enum class HandleType {
    Rectangle,      ///< Rectangular (linear) measurement
    Arc,            ///< Arc (curved) measurement
    Concentric      ///< Concentric circles measurement
};

// =============================================================================
// MeasureHandle Base Class
// =============================================================================

/**
 * @brief Base class for measurement handles
 *
 * A handle encapsulates the measurement region geometry and pre-computed
 * sampling information. This allows efficient repeated measurements in
 * the same region.
 */
class QIVISION_API MeasureHandle {
public:
    virtual ~MeasureHandle() = default;

    /// Get handle type
    virtual HandleType Type() const = 0;

    /// Check if handle is valid
    virtual bool IsValid() const = 0;

    /// Get profile length in pixels
    virtual double ProfileLength() const = 0;

    /// Get profile direction angle (radians)
    virtual double ProfileAngle() const = 0;

    /// Get bounding box of measurement region
    virtual Rect2d BoundingBox() const = 0;

    /// Check if point is inside measurement region
    virtual bool Contains(const Point2d& point) const = 0;

    /// Get sampling parameters
    int32_t NumLines() const { return numLines_; }
    double SamplesPerPixel() const { return samplesPerPixel_; }

protected:
    int32_t numLines_ = DEFAULT_NUM_LINES;
    double samplesPerPixel_ = DEFAULT_SAMPLES_PER_PIXEL;
};

// =============================================================================
// MeasureRectangle2
// =============================================================================

/**
 * @brief Rectangular (linear) measurement handle
 *
 * Defines a rotated rectangle for linear edge measurement.
 * Profile is extracted perpendicular to the rectangle's long axis.
 *
 * Geometry (Halcon compatible):
 * - Row, Column: Rectangle center
 * - Phi: Rotation angle of rectangle (radians, perpendicular to profile)
 * - Length1: Half-length along measurement direction (profile half-length)
 * - Length2: Half-length perpendicular to measurement (averaging half-width)
 *
 * Coordinate convention:
 * - Profile direction: angle = phi + PI/2
 * - Profile total length = 2 * Length1
 * - Rectangle total width = 2 * Length2
 *
 * Halcon equivalent: gen_measure_rectangle2
 */
class QIVISION_API MeasureRectangle2 : public MeasureHandle {
public:
    /**
     * @brief Default constructor
     */
    MeasureRectangle2();

    /**
     * @brief Construct rectangle handle (Halcon compatible)
     * @param row Center Y coordinate (Halcon: Row)
     * @param column Center X coordinate (Halcon: Column)
     * @param phi Rectangle rotation angle in radians (Halcon: Phi)
     * @param length1 Half-length along measurement direction (Halcon: Length1)
     * @param length2 Half-length perpendicular to measurement (Halcon: Length2)
     * @param width Image width for interpolation (Halcon: Width), 0 = auto
     * @param height Image height for interpolation (Halcon: Height), 0 = auto
     * @param interpolation Interpolation method (Halcon: Interpolation)
     */
    MeasureRectangle2(double row, double column,
                      double phi, double length1, double length2,
                      int32_t width = 0, int32_t height = 0,
                      const std::string& interpolation = "bilinear");

    /// Create from RotatedRect2d
    static MeasureRectangle2 FromRotatedRect(const RotatedRect2d& rect,
                                              int32_t numLines = DEFAULT_NUM_LINES);

    /// Create from two points (defining profile axis)
    static MeasureRectangle2 FromPoints(const Point2d& p1, const Point2d& p2,
                                         double width,
                                         int32_t numLines = DEFAULT_NUM_LINES);

    // MeasureHandle interface
    HandleType Type() const override { return HandleType::Rectangle; }
    bool IsValid() const override;
    double ProfileLength() const override { return 2.0 * length1_; }
    double ProfileAngle() const override;
    Rect2d BoundingBox() const override;
    bool Contains(const Point2d& point) const override;

    // Rectangle-specific accessors (Halcon compatible names)
    double Row() const { return row_; }
    double Column() const { return column_; }
    double Phi() const { return phi_; }
    double Length1() const { return length1_; }  ///< Half-length along profile
    double Length2() const { return length2_; }  ///< Half-width perpendicular

    // Legacy accessors (for compatibility)
    double CenterRow() const { return row_; }
    double CenterCol() const { return column_; }

    /// Get interpolation method
    const std::string& Interpolation() const { return interpolation_; }

    /// Get the rotated rectangle
    RotatedRect2d ToRotatedRect() const;

    /// Get profile start and end points
    void GetProfileEndpoints(Point2d& start, Point2d& end) const;

    /// Get all sampling lines (for visualization)
    std::vector<Segment2d> GetSamplingLines() const;

    /// Get perpendicular line offsets
    const std::vector<double>& GetLineOffsets() const { return lineOffsets_; }

    /// Translate the measurement region (Halcon: translate_measure)
    void Translate(double deltaRow, double deltaCol);

    /// Set center position
    void SetPosition(double row, double column);

private:
    double row_ = 0.0;           ///< Center row (Halcon: Row)
    double column_ = 0.0;        ///< Center column (Halcon: Column)
    double phi_ = 0.0;           ///< Rotation angle (Halcon: Phi)
    double length1_ = 0.0;       ///< Half-length along profile (Halcon: Length1)
    double length2_ = 0.0;       ///< Half-width perpendicular (Halcon: Length2)
    int32_t imageWidth_ = 0;     ///< Image width for interpolation (Halcon: Width)
    int32_t imageHeight_ = 0;    ///< Image height for interpolation (Halcon: Height)
    std::string interpolation_ = "bilinear";  ///< Interpolation method

    // Pre-computed geometry
    std::vector<double> lineOffsets_;  // Perpendicular offsets for averaging

    void ComputeSamplingGeometry();
};

// =============================================================================
// MeasureArc
// =============================================================================

/**
 * @brief Arc measurement handle
 *
 * Defines a circular arc for curved edge measurement.
 * Profile is extracted along the arc, with optional radial averaging.
 *
 * Geometry (Halcon compatible):
 * - CenterRow, CenterCol: Arc center
 * - Radius: Arc radius
 * - AngleStart: Start angle (radians)
 * - AngleExtent: Arc extent (radians), positive = CCW
 * - AnnulusRadius: Half-width of annular region for averaging
 *
 * Halcon equivalent: gen_measure_arc
 */
class QIVISION_API MeasureArc : public MeasureHandle {
public:
    /**
     * @brief Default constructor
     */
    MeasureArc();

    /**
     * @brief Construct arc handle (Halcon compatible)
     * @param centerRow Center Y coordinate (Halcon: CenterRow)
     * @param centerCol Center X coordinate (Halcon: CenterCol)
     * @param radius Arc radius (Halcon: Radius)
     * @param angleStart Start angle in radians (Halcon: AngleStart)
     * @param angleExtent Arc extent in radians (Halcon: AngleExtent)
     * @param annulusRadius Radial half-width for averaging (Halcon: AnnulusRadius)
     * @param width Image width for interpolation (Halcon: Width), 0 = auto
     * @param height Image height for interpolation (Halcon: Height), 0 = auto
     * @param interpolation Interpolation method (Halcon: Interpolation)
     */
    MeasureArc(double centerRow, double centerCol,
               double radius, double angleStart, double angleExtent,
               double annulusRadius = 0.0,
               int32_t width = 0, int32_t height = 0,
               const std::string& interpolation = "bilinear");

    /// Create from Arc2d
    static MeasureArc FromArc(const Arc2d& arc,
                               double annulusRadius = 0.0,
                               int32_t numLines = DEFAULT_NUM_LINES);

    /// Create from Circle2d (full circle)
    static MeasureArc FromCircle(const Circle2d& circle,
                                  double annulusRadius = 0.0,
                                  int32_t numLines = DEFAULT_NUM_LINES);

    // MeasureHandle interface
    HandleType Type() const override { return HandleType::Arc; }
    bool IsValid() const override;
    double ProfileLength() const override;  // Arc length
    double ProfileAngle() const override { return angleStart_; }  // Tangent at start
    Rect2d BoundingBox() const override;
    bool Contains(const Point2d& point) const override;

    // Arc-specific accessors
    double CenterRow() const { return centerRow_; }
    double CenterCol() const { return centerCol_; }
    double Radius() const { return radius_; }
    double AngleStart() const { return angleStart_; }
    double AngleExtent() const { return angleExtent_; }
    double AngleEnd() const { return angleStart_ + angleExtent_; }
    double AnnulusRadius() const { return annulusRadius_; }

    /// Get the arc
    Arc2d ToArc() const;

    /// Get point on arc at parameter t [0, 1]
    Point2d PointAt(double t) const;

    /// Get tangent angle at parameter t [0, 1]
    double TangentAt(double t) const;

    /// Convert profile position to angle
    double ProfilePosToAngle(double pos) const;

    /// Convert angle to profile position
    double AngleToProfilePos(double angle) const;

    /// Get radial offsets for averaging
    const std::vector<double>& GetRadiusOffsets() const { return radiusOffsets_; }

    /// Translate the measurement region (Halcon: translate_measure)
    void Translate(double deltaRow, double deltaCol);

    /// Set center position
    void SetPosition(double centerRow, double centerCol);

private:
    double centerRow_ = 0.0;
    double centerCol_ = 0.0;
    double radius_ = 0.0;
    double angleStart_ = 0.0;
    double angleExtent_ = 0.0;
    double annulusRadius_ = 0.0;

    std::vector<double> radiusOffsets_;  // Radial offsets for averaging

    void ComputeSamplingGeometry();
};

// =============================================================================
// MeasureConcentricCircles
// =============================================================================

/**
 * @brief Concentric circles measurement handle
 *
 * Defines a radial measurement region for multi-ring analysis.
 * Profile is extracted from inner to outer radius.
 *
 * Geometry:
 * - center: Circle center
 * - innerRadius, outerRadius: Radial extent
 * - angle: Direction angle for radial profile
 * - angularWidth: Angular width for averaging
 */
class QIVISION_API MeasureConcentricCircles : public MeasureHandle {
public:
    /**
     * @brief Default constructor
     */
    MeasureConcentricCircles();

    /**
     * @brief Construct concentric circles handle
     * @param centerRow Center Y coordinate
     * @param centerCol Center X coordinate
     * @param innerRadius Inner radius
     * @param outerRadius Outer radius
     * @param angle Direction angle for radial profile (radians)
     * @param angularWidth Angular width for averaging (radians)
     * @param numLines Number of angular lines for averaging
     * @param samplesPerPixel Sampling density along radius
     */
    MeasureConcentricCircles(double centerRow, double centerCol,
                              double innerRadius, double outerRadius,
                              double angle,
                              double angularWidth = 0.1,
                              int32_t numLines = DEFAULT_NUM_LINES,
                              double samplesPerPixel = DEFAULT_SAMPLES_PER_PIXEL);

    // MeasureHandle interface
    HandleType Type() const override { return HandleType::Concentric; }
    bool IsValid() const override;
    double ProfileLength() const override { return outerRadius_ - innerRadius_; }
    double ProfileAngle() const override { return angle_; }
    Rect2d BoundingBox() const override;
    bool Contains(const Point2d& point) const override;

    // Concentric-specific accessors
    double CenterRow() const { return centerRow_; }
    double CenterCol() const { return centerCol_; }
    double InnerRadius() const { return innerRadius_; }
    double OuterRadius() const { return outerRadius_; }
    double Angle() const { return angle_; }
    double AngularWidth() const { return angularWidth_; }

    /// Get point at radius
    Point2d PointAtRadius(double radius) const;

    /// Convert profile position to radius
    double ProfilePosToRadius(double pos) const;

    /// Get angular offsets for averaging
    const std::vector<double>& GetAngleOffsets() const { return angleOffsets_; }

    /// Translate the measurement region (Halcon: translate_measure)
    void Translate(double deltaRow, double deltaCol);

    /// Set center position
    void SetPosition(double centerRow, double centerCol);

private:
    double centerRow_ = 0.0;
    double centerCol_ = 0.0;
    double innerRadius_ = 0.0;
    double outerRadius_ = 0.0;
    double angle_ = 0.0;
    double angularWidth_ = 0.0;

    std::vector<double> angleOffsets_;  // Angular offsets for averaging

    void ComputeSamplingGeometry();
};

// =============================================================================
// Factory Functions (Halcon compatible)
// =============================================================================

/**
 * @brief Create rectangular measurement handle (Halcon: gen_measure_rectangle2)
 *
 * @param row Center Y coordinate
 * @param column Center X coordinate
 * @param phi Rotation angle (radians)
 * @param length1 Half-length along measurement direction
 * @param length2 Half-length perpendicular to measurement
 * @param width Image width for interpolation (0 = auto)
 * @param height Image height for interpolation (0 = auto)
 * @param interpolation Interpolation method: "nearest", "bilinear", "bicubic"
 * @return MeasureRectangle2 handle
 */
QIVISION_API MeasureRectangle2 GenMeasureRectangle2(double row, double column,
                                        double phi, double length1, double length2,
                                        int32_t width = 0, int32_t height = 0,
                                        const std::string& interpolation = "bilinear");

/**
 * @brief Create arc measurement handle (Halcon: gen_measure_arc)
 *
 * @param centerRow Center Y coordinate
 * @param centerCol Center X coordinate
 * @param radius Arc radius
 * @param angleStart Start angle (radians)
 * @param angleExtent Arc extent (radians)
 * @param annulusRadius Radial half-width for averaging
 * @param width Image width for interpolation (0 = auto)
 * @param height Image height for interpolation (0 = auto)
 * @param interpolation Interpolation method
 * @return MeasureArc handle
 */
QIVISION_API MeasureArc GenMeasureArc(double centerRow, double centerCol,
                          double radius, double angleStart, double angleExtent,
                          double annulusRadius = 0.0,
                          int32_t width = 0, int32_t height = 0,
                          const std::string& interpolation = "bilinear");

/**
 * @brief Create concentric circles measurement handle
 */
QIVISION_API MeasureConcentricCircles CreateMeasureConcentric(double centerRow, double centerCol,
                                                   double innerRadius, double outerRadius,
                                                   double angle,
                                                   double angularWidth = 0.1,
                                                   int32_t width = 0, int32_t height = 0,
                                                   const std::string& interpolation = "bilinear");

/**
 * @brief Create a measure handle covering a line segment
 */
QIVISION_API MeasureRectangle2 CreateMeasureFromSegment(const Point2d& p1, const Point2d& p2,
                                            double halfWidth = 5.0);

/**
 * @brief Create a measure handle covering a rotated rectangle
 */
QIVISION_API MeasureRectangle2 CreateMeasureFromRect(const RotatedRect2d& rect);

// =============================================================================
// Handle Manipulation Functions (Halcon compatible)
// =============================================================================

/**
 * @brief Translate a rectangular measurement handle (Halcon: translate_measure)
 * @param handle Measurement handle to translate (modified in place)
 * @param deltaRow Translation in row direction
 * @param deltaCol Translation in column direction
 */
inline void TranslateMeasure(MeasureRectangle2& handle, double deltaRow, double deltaCol) {
    handle.Translate(deltaRow, deltaCol);
}

/**
 * @brief Translate an arc measurement handle (Halcon: translate_measure)
 */
inline void TranslateMeasure(MeasureArc& handle, double deltaRow, double deltaCol) {
    handle.Translate(deltaRow, deltaCol);
}

/**
 * @brief Translate a concentric circles measurement handle (Halcon: translate_measure)
 */
inline void TranslateMeasure(MeasureConcentricCircles& handle, double deltaRow, double deltaCol) {
    handle.Translate(deltaRow, deltaCol);
}

} // namespace Qi::Vision::Measure
