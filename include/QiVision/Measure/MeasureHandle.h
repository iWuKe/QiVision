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

#include <cmath>
#include <cstdint>
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
class MeasureHandle {
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
 * Geometry:
 * - center: Rectangle center (row, column)
 * - phi: Rotation angle of rectangle (radians, perpendicular to profile)
 * - length: Profile length (along measurement direction)
 * - width: Rectangle width (perpendicular extent for averaging)
 *
 * Coordinate convention:
 * - Profile direction: angle = phi + PI/2
 * - Perpendicular lines are spread across 'width'
 */
class MeasureRectangle2 : public MeasureHandle {
public:
    /**
     * @brief Default constructor
     */
    MeasureRectangle2();

    /**
     * @brief Construct rectangle handle
     * @param centerRow Center Y coordinate
     * @param centerCol Center X coordinate
     * @param phi Rectangle rotation angle (radians)
     * @param length Profile length (measurement direction)
     * @param width Rectangle width (averaging direction)
     * @param numLines Number of perpendicular lines for averaging
     * @param samplesPerPixel Sampling density along profile
     */
    MeasureRectangle2(double centerRow, double centerCol,
                      double phi, double length, double width,
                      int32_t numLines = DEFAULT_NUM_LINES,
                      double samplesPerPixel = DEFAULT_SAMPLES_PER_PIXEL);

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
    double ProfileLength() const override { return length_; }
    double ProfileAngle() const override;
    Rect2d BoundingBox() const override;
    bool Contains(const Point2d& point) const override;

    // Rectangle-specific accessors
    double CenterRow() const { return centerRow_; }
    double CenterCol() const { return centerCol_; }
    double Phi() const { return phi_; }
    double Length() const { return length_; }
    double Width() const { return width_; }

    /// Get the rotated rectangle
    RotatedRect2d ToRotatedRect() const;

    /// Get profile start and end points
    void GetProfileEndpoints(Point2d& start, Point2d& end) const;

    /// Get all sampling lines (for visualization)
    std::vector<Segment2d> GetSamplingLines() const;

    /// Get perpendicular line offsets
    const std::vector<double>& GetLineOffsets() const { return lineOffsets_; }

private:
    double centerRow_ = 0.0;
    double centerCol_ = 0.0;
    double phi_ = 0.0;        // Rectangle rotation (perpendicular to profile)
    double length_ = 0.0;     // Profile length
    double width_ = 0.0;      // Rectangle width

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
 * Geometry:
 * - center: Arc center
 * - radius: Arc radius (for single arc) or center radius (for annular)
 * - angleStart: Start angle (radians)
 * - angleExtent: Arc extent (radians), positive = CCW
 * - annulusRadius: Half-width of annular region for averaging
 */
class MeasureArc : public MeasureHandle {
public:
    /**
     * @brief Default constructor
     */
    MeasureArc();

    /**
     * @brief Construct arc handle
     * @param centerRow Center Y coordinate
     * @param centerCol Center X coordinate
     * @param radius Arc radius
     * @param angleStart Start angle (radians)
     * @param angleExtent Arc extent (radians)
     * @param annulusRadius Radial extent for averaging (0 = single line)
     * @param numLines Number of radial lines for averaging
     * @param samplesPerPixel Sampling density along arc
     */
    MeasureArc(double centerRow, double centerCol,
               double radius, double angleStart, double angleExtent,
               double annulusRadius = 0.0,
               int32_t numLines = DEFAULT_NUM_LINES,
               double samplesPerPixel = DEFAULT_SAMPLES_PER_PIXEL);

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
class MeasureConcentricCircles : public MeasureHandle {
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
// Factory Functions
// =============================================================================

/**
 * @brief Create rectangular measurement handle
 */
MeasureRectangle2 CreateMeasureRect(double centerRow, double centerCol,
                                     double phi, double length, double width,
                                     int32_t numLines = DEFAULT_NUM_LINES,
                                     double samplesPerPixel = DEFAULT_SAMPLES_PER_PIXEL);

/**
 * @brief Create arc measurement handle
 */
MeasureArc CreateMeasureArc(double centerRow, double centerCol,
                             double radius, double angleStart, double angleExtent,
                             double annulusRadius = 0.0,
                             int32_t numLines = DEFAULT_NUM_LINES,
                             double samplesPerPixel = DEFAULT_SAMPLES_PER_PIXEL);

/**
 * @brief Create concentric circles measurement handle
 */
MeasureConcentricCircles CreateMeasureConcentric(double centerRow, double centerCol,
                                                   double innerRadius, double outerRadius,
                                                   double angle,
                                                   double angularWidth = 0.1,
                                                   int32_t numLines = DEFAULT_NUM_LINES,
                                                   double samplesPerPixel = DEFAULT_SAMPLES_PER_PIXEL);

/**
 * @brief Create a measure handle covering a line segment
 */
MeasureRectangle2 CreateMeasureFromSegment(const Point2d& p1, const Point2d& p2,
                                            double width = 10.0,
                                            int32_t numLines = DEFAULT_NUM_LINES);

/**
 * @brief Create a measure handle covering a rotated rectangle
 */
MeasureRectangle2 CreateMeasureFromRect(const RotatedRect2d& rect,
                                         int32_t numLines = DEFAULT_NUM_LINES);

} // namespace Qi::Vision::Measure
