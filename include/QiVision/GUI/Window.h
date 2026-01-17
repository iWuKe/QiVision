#pragma once

/**
 * @file Window.h
 * @brief Lightweight GUI window for image display and debugging
 *
 * Cross-platform window implementation:
 * - Linux: X11 (Xlib)
 * - Windows: Win32 GDI
 *
 * Features:
 * - Display images with auto-scaling
 * - Wait for key press
 * - Adjustable window size
 * - Mouse interaction (click, move, scroll)
 * - Zoom and pan
 * - Interactive ROI drawing
 */

#include <QiVision/Core/QImage.h>
#include <QiVision/Core/Types.h>
#include <string>
#include <memory>
#include <functional>
#include <vector>

namespace Qi::Vision::GUI {

/**
 * @brief Scale mode for image display
 */
enum class ScaleMode {
    None,       ///< No scaling (1:1 pixel)
    Fit,        ///< Fit image to window (maintain aspect ratio)
    Fill,       ///< Fill window (may crop)
    Stretch     ///< Stretch to window size (ignore aspect ratio)
};

// =============================================================================
// Mouse Event Types
// =============================================================================

/**
 * @brief Mouse button identifiers
 */
enum class MouseButton {
    None = 0,
    Left = 1,
    Middle = 2,
    Right = 3,
    WheelUp = 4,
    WheelDown = 5
};

/**
 * @brief Mouse event types
 */
enum class MouseEventType {
    Move,           ///< Mouse moved
    ButtonDown,     ///< Button pressed
    ButtonUp,       ///< Button released
    DoubleClick,    ///< Double click
    Wheel           ///< Mouse wheel
};

/**
 * @brief Keyboard modifier flags
 */
enum class KeyModifier : uint32_t {
    None = 0,
    Shift = 1,
    Ctrl = 2,
    Alt = 4
};

inline KeyModifier operator|(KeyModifier a, KeyModifier b) {
    return static_cast<KeyModifier>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

inline bool operator&(KeyModifier a, KeyModifier b) {
    return (static_cast<uint32_t>(a) & static_cast<uint32_t>(b)) != 0;
}

/**
 * @brief Mouse event data
 */
struct MouseEvent {
    MouseEventType type;        ///< Event type
    MouseButton button;         ///< Button involved
    int32_t x;                  ///< Window X coordinate
    int32_t y;                  ///< Window Y coordinate
    double imageX;              ///< Image X coordinate (after zoom/pan)
    double imageY;              ///< Image Y coordinate (after zoom/pan)
    int32_t wheelDelta;         ///< Wheel delta (positive = up)
    KeyModifier modifiers;      ///< Active modifiers (Shift, Ctrl, Alt)
};

/**
 * @brief Callback function types
 */
using MouseCallback = std::function<void(const MouseEvent& event)>;
using KeyCallback = std::function<void(int32_t keyCode, KeyModifier modifiers)>;

// =============================================================================
// ROI Drawing Types
// =============================================================================

/**
 * @brief ROI (Region of Interest) types for interactive drawing
 */
enum class ROIType {
    Rectangle,      ///< Axis-aligned rectangle
    RotatedRect,    ///< Rotated rectangle
    Circle,         ///< Circle
    Ellipse,        ///< Ellipse
    Line,           ///< Line segment
    Polygon,        ///< Polygon (multiple points)
    Point           ///< Single point
};

/**
 * @brief ROI drawing result
 */
struct ROIResult {
    ROIType type;
    bool valid = false;             ///< True if ROI was completed (not cancelled)

    // Rectangle/RotatedRect
    double row1 = 0, col1 = 0;      ///< Top-left or center
    double row2 = 0, col2 = 0;      ///< Bottom-right or size
    double angle = 0;               ///< Rotation angle (radians)

    // Circle/Ellipse
    double centerRow = 0, centerCol = 0;
    double radius = 0;              ///< Circle radius
    double radiusRow = 0, radiusCol = 0;  ///< Ellipse radii

    // Line
    double startRow = 0, startCol = 0;
    double endRow = 0, endCol = 0;

    // Polygon/Point
    std::vector<Point2d> points;
};

/**
 * @brief Lightweight window for image display
 *
 * Example usage:
 * @code
 * Window win("Debug", 800, 600);
 * win.DispImage(image);
 * win.WaitKey();  // Wait for any key
 *
 * // Or with timeout
 * while (win.WaitKey(30) != 'q') {
 *     win.DispImage(processedImage);
 * }
 * @endcode
 */
class Window {
public:
    /**
     * @brief Create a window
     * @param title Window title
     * @param width Initial window width (0 = auto from first image)
     * @param height Initial window height (0 = auto from first image)
     */
    Window(const std::string& title = "QiVision", int32_t width = 0, int32_t height = 0);

    /**
     * @brief Destructor - closes window
     */
    ~Window();

    // Non-copyable
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    // Movable
    Window(Window&& other) noexcept;
    Window& operator=(Window&& other) noexcept;

    /**
     * @brief Display an image in the window
     * @param image Image to display (grayscale or RGB)
     * @param scaleMode How to scale the image
     */
    void DispImage(const QImage& image, ScaleMode scaleMode = ScaleMode::Fit);

    /**
     * @brief Wait for a key press
     * @param timeoutMs Timeout in milliseconds (0 = wait forever, -1 = no wait)
     * @return Key code pressed, or -1 if timeout/closed
     */
    int32_t WaitKey(int32_t timeoutMs = 0);

    /**
     * @brief Check if window is still open
     */
    bool IsOpen() const;

    /**
     * @brief Close the window
     */
    void Close();

    /**
     * @brief Set window title
     */
    void SetTitle(const std::string& title);

    /**
     * @brief Resize window
     */
    void Resize(int32_t width, int32_t height);

    /**
     * @brief Get current window size
     */
    void GetSize(int32_t& width, int32_t& height) const;

    /**
     * @brief Move window to position
     */
    void Move(int32_t x, int32_t y);

    /**
     * @brief Enable/disable auto-resize mode
     * @param enable If true, window resizes to fit each image (with max size limit)
     * @param maxWidth Maximum window width (0 = screen width)
     * @param maxHeight Maximum window height (0 = screen height)
     */
    void SetAutoResize(bool enable, int32_t maxWidth = 0, int32_t maxHeight = 0);

    /**
     * @brief Check if auto-resize is enabled
     */
    bool IsAutoResize() const;

    // =========================================================================
    // Mouse Interaction
    // =========================================================================

    /**
     * @brief Set mouse event callback
     * @param callback Function called on mouse events (nullptr to disable)
     */
    void SetMouseCallback(MouseCallback callback);

    /**
     * @brief Set key event callback
     * @param callback Function called on key events (nullptr to disable)
     */
    void SetKeyCallback(KeyCallback callback);

    /**
     * @brief Get current mouse position in window coordinates
     * @param[out] x Window X coordinate
     * @param[out] y Window Y coordinate
     * @return True if mouse is over window
     */
    bool GetMousePosition(int32_t& x, int32_t& y) const;

    /**
     * @brief Get current mouse position in image coordinates
     * @param[out] imageX Image X coordinate
     * @param[out] imageY Image Y coordinate
     * @return True if mouse is over image area
     */
    bool GetMouseImagePosition(double& imageX, double& imageY) const;

    // =========================================================================
    // Zoom and Pan
    // =========================================================================

    /**
     * @brief Enable/disable zoom and pan interaction
     * @param enable If true, scroll wheel zooms and drag pans
     *
     * When enabled:
     * - Mouse wheel: zoom in/out (centered on cursor)
     * - Left button drag: pan image
     * - Double-click: reset zoom to fit
     * - Right-click: reset zoom to 1:1
     */
    void EnableZoomPan(bool enable);

    /**
     * @brief Check if zoom/pan is enabled
     */
    bool IsZoomPanEnabled() const;

    /**
     * @brief Get current zoom level
     * @return Zoom factor (1.0 = 100%, 2.0 = 200%, etc.)
     */
    double GetZoomLevel() const;

    /**
     * @brief Set zoom level
     * @param zoom Zoom factor (clamped to [0.1, 100.0])
     * @param centerOnMouse If true, zoom centers on current mouse position
     */
    void SetZoomLevel(double zoom, bool centerOnMouse = false);

    /**
     * @brief Get current pan offset in image coordinates
     * @param[out] offsetX X offset
     * @param[out] offsetY Y offset
     */
    void GetPanOffset(double& offsetX, double& offsetY) const;

    /**
     * @brief Set pan offset in image coordinates
     * @param offsetX X offset
     * @param offsetY Y offset
     */
    void SetPanOffset(double offsetX, double offsetY);

    /**
     * @brief Reset zoom to fit image in window
     */
    void ResetZoom();

    /**
     * @brief Zoom to show specific image region
     * @param row1 Top row
     * @param col1 Left column
     * @param row2 Bottom row
     * @param col2 Right column
     */
    void ZoomToRegion(double row1, double col1, double row2, double col2);

    // =========================================================================
    // Coordinate Conversion
    // =========================================================================

    /**
     * @brief Convert window coordinates to image coordinates
     * @param windowX Window X coordinate
     * @param windowY Window Y coordinate
     * @param[out] imageX Image X coordinate
     * @param[out] imageY Image Y coordinate
     * @return True if point is within image bounds
     */
    bool WindowToImage(int32_t windowX, int32_t windowY,
                       double& imageX, double& imageY) const;

    /**
     * @brief Convert image coordinates to window coordinates
     * @param imageX Image X coordinate
     * @param imageY Image Y coordinate
     * @param[out] windowX Window X coordinate
     * @param[out] windowY Window Y coordinate
     * @return True if point is visible in window
     */
    bool ImageToWindow(double imageX, double imageY,
                       int32_t& windowX, int32_t& windowY) const;

    // =========================================================================
    // Interactive ROI Drawing
    // =========================================================================

    /**
     * @brief Interactively draw a rectangle ROI
     * @return ROI result (check valid flag)
     *
     * Usage:
     * - Click and drag to draw rectangle
     * - Release to confirm
     * - Press ESC to cancel
     */
    ROIResult DrawRectangle();

    /**
     * @brief Interactively draw a circle ROI
     * @return ROI result
     *
     * Usage:
     * - Click center, drag to set radius
     * - Release to confirm
     * - Press ESC to cancel
     */
    ROIResult DrawCircle();

    /**
     * @brief Interactively draw a line ROI
     * @return ROI result
     */
    ROIResult DrawLine();

    /**
     * @brief Interactively draw a polygon ROI
     * @return ROI result
     *
     * Usage:
     * - Click to add points
     * - Double-click or press Enter to close polygon
     * - Press ESC to cancel
     * - Press Backspace to remove last point
     */
    ROIResult DrawPolygon();

    /**
     * @brief Interactively select a point
     * @return ROI result with single point
     */
    ROIResult DrawPoint();

    /**
     * @brief Generic ROI drawing
     * @param type Type of ROI to draw
     * @return ROI result
     */
    ROIResult DrawROI(ROIType type);

    // =========================================================================
    // Static convenience functions
    // =========================================================================

    /**
     * @brief Quick display - create window, show image, wait for key
     * @param image Image to display
     * @param title Window title
     * @return Key code pressed
     */
    static int32_t ShowImage(const QImage& image, const std::string& title = "QiVision");

    /**
     * @brief Quick display with timeout
     */
    static int32_t ShowImage(const QImage& image, const std::string& title, int32_t timeoutMs);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// =============================================================================
// Convenience functions (Halcon-style)
// =============================================================================

/**
 * @brief Display image in a window (creates window if needed)
 * @param image Image to display
 * @param windowName Window name (creates new if not exists)
 */
void DispImage(const QImage& image, const std::string& windowName = "QiVision");

/**
 * @brief Wait for key press on any window
 * @param timeoutMs Timeout in milliseconds (0 = forever)
 * @return Key code, or -1 on timeout
 */
int32_t WaitKey(int32_t timeoutMs = 0);

/**
 * @brief Close a named window
 */
void CloseWindow(const std::string& windowName);

/**
 * @brief Close all windows
 */
void CloseAllWindows();

} // namespace Qi::Vision::GUI
