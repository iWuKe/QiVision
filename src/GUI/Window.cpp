/**
 * @file Window.cpp
 * @brief Cross-platform window implementation
 *
 * Supported platforms:
 * - Linux: X11 (Xlib)
 * - Windows: Win32 GDI
 * - macOS: Cocoa (AppKit) - requires .mm compilation
 * - Android: Stub (GUI requires Java layer)
 * - iOS: Stub (GUI requires Swift/ObjC layer)
 */

#include <QiVision/GUI/Window.h>

#include <algorithm>
#include <unordered_map>
#include <mutex>
#include <cstring>
#include <sstream>

// =============================================================================
// Platform detection
// =============================================================================

#if defined(_WIN32) || defined(_WIN64)
    #define QIVISION_PLATFORM_WINDOWS
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>

#elif defined(__APPLE__)
    #include <TargetConditionals.h>
    #if TARGET_OS_IPHONE || TARGET_IPHONE_SIMULATOR
        #define QIVISION_PLATFORM_IOS
        // iOS: GUI requires Swift/ObjC layer, use stub
    #else
        #define QIVISION_PLATFORM_MACOS
        #ifdef QIVISION_HAS_COCOA
            // macOS with Cocoa support (requires .mm compilation)
            #import <Cocoa/Cocoa.h>
        #endif
    #endif

#elif defined(__ANDROID__)
    #define QIVISION_PLATFORM_ANDROID
    // Android: GUI requires Java layer (Activity/View), use stub
    // Native code can use ANativeWindow but cannot create windows

#elif defined(__linux__)
    #define QIVISION_PLATFORM_LINUX
    #ifdef QIVISION_HAS_X11
        #include <X11/Xlib.h>
        #include <X11/Xutil.h>
        #include <X11/keysym.h>
        #include <unistd.h>
        #include <sys/time.h>
        // X11 defines 'None' as a macro, which conflicts with ScaleMode::None
        #ifdef None
            #undef None
        #endif
    #endif
#endif

namespace Qi::Vision::GUI {

// =============================================================================
// Global window manager for convenience functions
// =============================================================================

static std::unordered_map<std::string, std::unique_ptr<Window>> g_windows;
static std::mutex g_windowsMutex;

// =============================================================================
// Platform-specific implementation
// =============================================================================

#if defined(QIVISION_PLATFORM_LINUX) && defined(QIVISION_HAS_X11)

class Window::Impl {
public:
    Display* display_ = nullptr;
    ::Window window_ = 0;
    GC gc_ = nullptr;
    XImage* ximage_ = nullptr;
    Atom wmDeleteMessage_ = 0;

    int32_t width_ = 0;
    int32_t height_ = 0;
    bool isOpen_ = false;
    std::string title_;

    // Image buffer for display
    std::vector<uint8_t> buffer_;
    int32_t imageWidth_ = 0;
    int32_t imageHeight_ = 0;

    // Original image dimensions (for coordinate conversion)
    int32_t srcImageWidth_ = 0;
    int32_t srcImageHeight_ = 0;

    // Auto-resize settings
    bool autoResize_ = false;
    int32_t maxWidth_ = 0;
    int32_t maxHeight_ = 0;
    int32_t screenWidth_ = 0;
    int32_t screenHeight_ = 0;

    // Resizable setting
    bool resizable_ = true;

    // Mouse interaction
    MouseCallback mouseCallback_;
    KeyCallback keyCallback_;
    int32_t mouseX_ = 0;
    int32_t mouseY_ = 0;
    bool mouseInWindow_ = false;

    // Zoom and pan
    bool zoomPanEnabled_ = false;
    double zoomLevel_ = 1.0;
    double panOffsetX_ = 0.0;
    double panOffsetY_ = 0.0;
    bool isPanning_ = false;
    int32_t panStartX_ = 0;
    int32_t panStartY_ = 0;
    double panStartOffsetX_ = 0.0;
    double panStartOffsetY_ = 0.0;

    // Display offset (for centering)
    int32_t displayOffsetX_ = 0;
    int32_t displayOffsetY_ = 0;
    double displayScale_ = 1.0;

    // Pixel info display
    bool pixelInfoEnabled_ = false;
    std::string baseTitle_;

    // Store the original image for redraw
    QImage currentImage_;
    ScaleMode currentScaleMode_ = ScaleMode::Fit;

    Impl(const std::string& title, int32_t width, int32_t height)
        : width_(width), height_(height), title_(title), baseTitle_(title) {

        display_ = XOpenDisplay(nullptr);
        if (!display_) {
            return;
        }

        int screen = DefaultScreen(display_);

        // Get screen size for auto-resize limits
        screenWidth_ = DisplayWidth(display_, screen);
        screenHeight_ = DisplayHeight(display_, screen);

        // If size not specified, use reasonable default
        if (width_ <= 0) width_ = 800;
        if (height_ <= 0) height_ = 600;

        window_ = XCreateSimpleWindow(
            display_,
            RootWindow(display_, screen),
            100, 100,
            width_, height_,
            1,
            BlackPixel(display_, screen),
            WhitePixel(display_, screen)
        );

        if (!window_) {
            XCloseDisplay(display_);
            display_ = nullptr;
            return;
        }

        // Set window title
        XStoreName(display_, window_, title_.c_str());

        // Select input events (including mouse events)
        XSelectInput(display_, window_,
                     ExposureMask | KeyPressMask | StructureNotifyMask |
                     ButtonPressMask | ButtonReleaseMask | PointerMotionMask |
                     EnterWindowMask | LeaveWindowMask);

        // Handle window close button
        wmDeleteMessage_ = XInternAtom(display_, "WM_DELETE_WINDOW", False);
        XSetWMProtocols(display_, window_, &wmDeleteMessage_, 1);

        // Create graphics context
        gc_ = XCreateGC(display_, window_, 0, nullptr);

        // Show window
        XMapWindow(display_, window_);
        // Use XSync to ensure window is mapped before returning
        XSync(display_, False);

        isOpen_ = true;
    }

    ~Impl() {
        Close();
    }

    void Close() {
        if (ximage_) {
            // Don't destroy data, we own it
            ximage_->data = nullptr;
            XDestroyImage(ximage_);
            ximage_ = nullptr;
        }
        if (gc_) {
            XFreeGC(display_, gc_);
            gc_ = nullptr;
        }
        if (window_) {
            XDestroyWindow(display_, window_);
            window_ = 0;
        }
        if (display_) {
            XCloseDisplay(display_);
            display_ = nullptr;
        }
        isOpen_ = false;
    }

    void Show(const QImage& image, ScaleMode scaleMode) {
        if (!isOpen_ || !display_) return;

        // Store for redraw
        currentImage_ = image.Clone();
        currentScaleMode_ = scaleMode;

        int32_t srcWidth = image.Width();
        int32_t srcHeight = image.Height();
        srcImageWidth_ = srcWidth;
        srcImageHeight_ = srcHeight;

        // Auto-resize window to fit image
        if (autoResize_) {
            int32_t maxW = (maxWidth_ > 0) ? maxWidth_ : (screenWidth_ - 100);
            int32_t maxH = (maxHeight_ > 0) ? maxHeight_ : (screenHeight_ - 100);

            int32_t newWidth = srcWidth;
            int32_t newHeight = srcHeight;

            // Scale down if image is larger than max size
            if (newWidth > maxW || newHeight > maxH) {
                double scaleW = static_cast<double>(maxW) / srcWidth;
                double scaleH = static_cast<double>(maxH) / srcHeight;
                double scale = std::min(scaleW, scaleH);
                newWidth = static_cast<int32_t>(srcWidth * scale);
                newHeight = static_cast<int32_t>(srcHeight * scale);
            }

            // Resize window if size changed
            if (newWidth != width_ || newHeight != height_) {
                width_ = newWidth;
                height_ = newHeight;
                XResizeWindow(display_, window_, width_, height_);
                // Use XSync to ensure window resize completes before drawing
                XSync(display_, False);
            }
        }

        int32_t dstWidth = width_;
        int32_t dstHeight = height_;

        // Calculate display size based on scale mode
        double scaleX = 1.0, scaleY = 1.0;

        switch (scaleMode) {
            case ScaleMode::None:
                dstWidth = srcWidth;
                dstHeight = srcHeight;
                break;

            case ScaleMode::Fit: {
                double scaleW = static_cast<double>(width_) / srcWidth;
                double scaleH = static_cast<double>(height_) / srcHeight;
                double scale = std::min(scaleW, scaleH);
                dstWidth = static_cast<int32_t>(srcWidth * scale);
                dstHeight = static_cast<int32_t>(srcHeight * scale);
                scaleX = scaleY = scale;
                break;
            }

            case ScaleMode::Fill: {
                double scaleW = static_cast<double>(width_) / srcWidth;
                double scaleH = static_cast<double>(height_) / srcHeight;
                double scale = std::max(scaleW, scaleH);
                scaleX = scaleY = scale;
                dstWidth = width_;
                dstHeight = height_;
                break;
            }

            case ScaleMode::Stretch:
                scaleX = static_cast<double>(width_) / srcWidth;
                scaleY = static_cast<double>(height_) / srcHeight;
                dstWidth = width_;
                dstHeight = height_;
                break;
        }

        // Apply zoom if enabled
        if (zoomPanEnabled_) {
            scaleX *= zoomLevel_;
            scaleY *= zoomLevel_;
            dstWidth = static_cast<int32_t>(srcWidth * scaleX);
            dstHeight = static_cast<int32_t>(srcHeight * scaleY);
        }

        imageWidth_ = dstWidth;
        imageHeight_ = dstHeight;
        displayScale_ = scaleX;  // Assuming uniform scaling

        const uint8_t* srcData = static_cast<const uint8_t*>(image.Data());
        size_t srcStride = image.Stride();
        int channels = image.Channels();

        // Create/update XImage
        if (ximage_) {
            ximage_->data = nullptr;
            XDestroyImage(ximage_);
            ximage_ = nullptr;
        }

        int screen = DefaultScreen(display_);
        Visual* visual = DefaultVisual(display_, screen);
        int depth = DefaultDepth(display_, screen);

        // First create XImage to get the proper bytes_per_line
        ximage_ = XCreateImage(
            display_, visual, depth, ZPixmap, 0,
            nullptr,  // Don't set data yet
            dstWidth, dstHeight, 32, 0  // Let X11 decide bytes_per_line
        );

        if (!ximage_) return;

        // Use X11's calculated bytes_per_line for proper alignment
        int xStride = ximage_->bytes_per_line;
        buffer_.resize(xStride * dstHeight);
        std::memset(buffer_.data(), 0, buffer_.size());

        // Scale with area averaging (box filter) for downscaling
        // This prevents thin lines from disappearing when shrinking
        bool useAreaAverage = (scaleX < 1.0 || scaleY < 1.0);

        for (int32_t dy = 0; dy < dstHeight; ++dy) {
            for (int32_t dx = 0; dx < dstWidth; ++dx) {
                uint8_t* dst = &buffer_[dy * xStride + dx * 4];

                if (useAreaAverage) {
                    // Calculate the source region that maps to this destination pixel
                    double srcX0 = dx / scaleX;
                    double srcY0 = dy / scaleY;
                    double srcX1 = (dx + 1) / scaleX;
                    double srcY1 = (dy + 1) / scaleY;

                    int32_t ix0 = static_cast<int32_t>(srcX0);
                    int32_t iy0 = static_cast<int32_t>(srcY0);
                    int32_t ix1 = std::min(static_cast<int32_t>(srcX1) + 1, srcWidth);
                    int32_t iy1 = std::min(static_cast<int32_t>(srcY1) + 1, srcHeight);

                    // Average all pixels in the source region
                    double sumR = 0, sumG = 0, sumB = 0;
                    int count = 0;

                    for (int32_t sy = iy0; sy < iy1; ++sy) {
                        for (int32_t sx = ix0; sx < ix1; ++sx) {
                            if (channels == 1) {
                                uint8_t gray = srcData[sy * srcStride + sx];
                                sumR += gray;
                                sumG += gray;
                                sumB += gray;
                            } else if (channels == 3) {
                                const uint8_t* src = &srcData[sy * srcStride + sx * 3];
                                sumR += src[0];
                                sumG += src[1];
                                sumB += src[2];
                            }
                            count++;
                        }
                    }

                    if (count > 0) {
                        dst[0] = static_cast<uint8_t>(sumB / count);  // B
                        dst[1] = static_cast<uint8_t>(sumG / count);  // G
                        dst[2] = static_cast<uint8_t>(sumR / count);  // R
                        dst[3] = 255;
                    }
                } else {
                    // Nearest neighbor for upscaling
                    int32_t sx = static_cast<int32_t>(dx / scaleX);
                    int32_t sy = static_cast<int32_t>(dy / scaleY);
                    if (sx >= srcWidth) sx = srcWidth - 1;
                    if (sy >= srcHeight) sy = srcHeight - 1;

                    if (channels == 1) {
                        uint8_t gray = srcData[sy * srcStride + sx];
                        dst[0] = gray;  // B
                        dst[1] = gray;  // G
                        dst[2] = gray;  // R
                        dst[3] = 255;   // A
                    } else if (channels == 3) {
                        const uint8_t* src = &srcData[sy * srcStride + sx * 3];
                        dst[0] = src[2];  // B
                        dst[1] = src[1];  // G
                        dst[2] = src[0];  // R
                        dst[3] = 255;     // A
                    }
                }
            }
        }

        ximage_->data = reinterpret_cast<char*>(buffer_.data());

        // Clear window and draw image centered
        // Use XClearArea with exposures=False and sync to avoid flicker
        XClearWindow(display_, window_);
        XSync(display_, False);  // Ensure clear completes before drawing

        int offsetX = (width_ - dstWidth) / 2;
        int offsetY = (height_ - dstHeight) / 2;

        // Apply pan offset if zoom/pan enabled
        if (zoomPanEnabled_) {
            offsetX = static_cast<int>(offsetX - panOffsetX_ * displayScale_);
            offsetY = static_cast<int>(offsetY - panOffsetY_ * displayScale_);
        }

        displayOffsetX_ = offsetX;
        displayOffsetY_ = offsetY;

        // Clip to window bounds
        int srcX = 0, srcY = 0;
        int drawX = offsetX, drawY = offsetY;
        int drawWidth = dstWidth, drawHeight = dstHeight;

        if (drawX < 0) {
            srcX = -drawX;
            drawWidth += drawX;
            drawX = 0;
        }
        if (drawY < 0) {
            srcY = -drawY;
            drawHeight += drawY;
            drawY = 0;
        }
        if (drawX + drawWidth > width_) {
            drawWidth = width_ - drawX;
        }
        if (drawY + drawHeight > height_) {
            drawHeight = height_ - drawY;
        }

        if (drawWidth > 0 && drawHeight > 0) {
            XPutImage(display_, window_, gc_, ximage_,
                      srcX, srcY, drawX, drawY, drawWidth, drawHeight);
        }
        // Use XSync instead of XFlush to ensure the image is actually displayed
        // XFlush only sends requests to server, XSync waits for completion
        XSync(display_, False);
    }

    // Helper to get modifiers from X11 state
    KeyModifier GetModifiers(unsigned int state) {
        KeyModifier mods = KeyModifier::None;
        if (state & ShiftMask) mods = mods | KeyModifier::Shift;
        if (state & ControlMask) mods = mods | KeyModifier::Ctrl;
        if (state & Mod1Mask) mods = mods | KeyModifier::Alt;
        return mods;
    }

    // Convert window coords to image coords
    bool WindowToImageCoords(int32_t wx, int32_t wy, double& ix, double& iy) {
        if (displayScale_ <= 0 || srcImageWidth_ <= 0 || srcImageHeight_ <= 0) {
            return false;
        }
        ix = (wx - displayOffsetX_) / displayScale_ + panOffsetX_;
        iy = (wy - displayOffsetY_) / displayScale_ + panOffsetY_;
        return ix >= 0 && ix < srcImageWidth_ && iy >= 0 && iy < srcImageHeight_;
    }

    // Convert image coords to window coords
    bool ImageToWindowCoords(double ix, double iy, int32_t& wx, int32_t& wy) {
        if (displayScale_ <= 0) return false;
        wx = static_cast<int32_t>(ix * displayScale_ + displayOffsetX_);
        wy = static_cast<int32_t>(iy * displayScale_ + displayOffsetY_);
        return wx >= 0 && wx < width_ && wy >= 0 && wy < height_;
    }

    // Create mouse event from X11 event
    MouseEvent CreateMouseEvent(MouseEventType type, int x, int y,
                                MouseButton button, unsigned int state, int wheelDelta = 0) {
        MouseEvent evt;
        evt.type = type;
        evt.button = button;
        evt.x = x;
        evt.y = y;
        evt.wheelDelta = wheelDelta;
        evt.modifiers = GetModifiers(state);
        WindowToImageCoords(x, y, evt.imageX, evt.imageY);
        return evt;
    }

    void HandleMouseEvent(const XEvent& event) {
        MouseEvent mouseEvt;

        switch (event.type) {
            case MotionNotify:
                mouseX_ = event.xmotion.x;
                mouseY_ = event.xmotion.y;

                // Handle panning
                if (isPanning_ && zoomPanEnabled_) {
                    double dx = (mouseX_ - panStartX_) / displayScale_;
                    double dy = (mouseY_ - panStartY_) / displayScale_;
                    panOffsetX_ = panStartOffsetX_ - dx;
                    panOffsetY_ = panStartOffsetY_ - dy;
                    if (currentImage_.Width() > 0) {
                        Show(currentImage_, currentScaleMode_);
                    }
                }

                if (mouseCallback_) {
                    mouseEvt = CreateMouseEvent(MouseEventType::Move,
                        mouseX_, mouseY_, MouseButton::None, event.xmotion.state);
                    mouseCallback_(mouseEvt);
                }

                // Update pixel info in title
                if (pixelInfoEnabled_ && currentImage_.Width() > 0) {
                    double imgX, imgY;
                    if (WindowToImageCoords(mouseX_, mouseY_, imgX, imgY)) {
                        int ix = static_cast<int>(imgX);
                        int iy = static_cast<int>(imgY);
                        if (ix >= 0 && ix < currentImage_.Width() &&
                            iy >= 0 && iy < currentImage_.Height()) {
                            std::ostringstream oss;
                            oss << baseTitle_ << " - (" << ix << ", " << iy << ")";
                            if (currentImage_.Channels() == 1) {
                                int val = currentImage_.At(ix, iy);
                                oss << " = " << val;
                            } else if (currentImage_.Channels() >= 3) {
                                const uint8_t* row = static_cast<const uint8_t*>(currentImage_.RowPtr(iy));
                                const uint8_t* p = row + ix * currentImage_.Channels();
                                oss << " = (" << (int)p[0] << "," << (int)p[1] << "," << (int)p[2] << ")";
                            }
                            XStoreName(display_, window_, oss.str().c_str());
                            XFlush(display_);
                        }
                    }
                }
                break;

            case ButtonPress: {
                MouseButton btn = MouseButton::None;
                int wheelDelta = 0;

                switch (event.xbutton.button) {
                    case Button1: btn = MouseButton::Left; break;
                    case Button2: btn = MouseButton::Middle; break;
                    case Button3: btn = MouseButton::Right; break;
                    case Button4: btn = MouseButton::WheelUp; wheelDelta = 1; break;
                    case Button5: btn = MouseButton::WheelDown; wheelDelta = -1; break;
                }

                // Handle zoom/pan
                if (zoomPanEnabled_) {
                    if (btn == MouseButton::Left) {
                        isPanning_ = true;
                        panStartX_ = event.xbutton.x;
                        panStartY_ = event.xbutton.y;
                        panStartOffsetX_ = panOffsetX_;
                        panStartOffsetY_ = panOffsetY_;
                    } else if (btn == MouseButton::WheelUp || btn == MouseButton::WheelDown) {
                        // Zoom centered on mouse cursor
                        // Get image coords under mouse before zoom
                        double ix = 0, iy = 0;
                        WindowToImageCoords(event.xbutton.x, event.xbutton.y, ix, iy);

                        double oldZoom = zoomLevel_;
                        if (btn == MouseButton::WheelUp) {
                            zoomLevel_ *= 1.2;
                        } else {
                            zoomLevel_ /= 1.2;
                        }
                        zoomLevel_ = std::max(0.1, std::min(100.0, zoomLevel_));

                        // Calculate base scale (without zoom)
                        double baseScale = displayScale_ / oldZoom;
                        double newScale = baseScale * zoomLevel_;

                        // Adjust panOffset so that (ix, iy) stays under mouse cursor
                        // After zoom: windowX = ix * newScale + newDisplayOffsetX
                        // newDisplayOffsetX = (width_ - srcImageWidth_ * newScale) / 2 - panOffsetX * newScale
                        // We want: event.xbutton.x = ix * newScale + (width_ - srcImageWidth_ * newScale) / 2 - panOffsetX * newScale
                        // Solve for panOffsetX:
                        double newDstWidth = srcImageWidth_ * newScale;
                        double baseOffsetX = (width_ - newDstWidth) / 2.0;
                        panOffsetX_ = (ix * newScale + baseOffsetX - event.xbutton.x) / newScale;

                        double newDstHeight = srcImageHeight_ * newScale;
                        double baseOffsetY = (height_ - newDstHeight) / 2.0;
                        panOffsetY_ = (iy * newScale + baseOffsetY - event.xbutton.y) / newScale;

                        if (currentImage_.Width() > 0) {
                            Show(currentImage_, currentScaleMode_);
                        }
                    } else if (btn == MouseButton::Right) {
                        // Reset to 1:1
                        zoomLevel_ = 1.0;
                        panOffsetX_ = 0;
                        panOffsetY_ = 0;
                        if (currentImage_.Width() > 0) {
                            Show(currentImage_, currentScaleMode_);
                        }
                    }
                }

                if (mouseCallback_) {
                    MouseEventType type = (wheelDelta != 0) ? MouseEventType::Wheel : MouseEventType::ButtonDown;
                    mouseEvt = CreateMouseEvent(type, event.xbutton.x, event.xbutton.y,
                        btn, event.xbutton.state, wheelDelta);
                    mouseCallback_(mouseEvt);
                }
                break;
            }

            case ButtonRelease: {
                MouseButton btn = MouseButton::None;
                switch (event.xbutton.button) {
                    case Button1: btn = MouseButton::Left; break;
                    case Button2: btn = MouseButton::Middle; break;
                    case Button3: btn = MouseButton::Right; break;
                }

                if (btn == MouseButton::Left) {
                    isPanning_ = false;
                }

                if (mouseCallback_) {
                    mouseEvt = CreateMouseEvent(MouseEventType::ButtonUp,
                        event.xbutton.x, event.xbutton.y, btn, event.xbutton.state);
                    mouseCallback_(mouseEvt);
                }
                break;
            }

            case EnterNotify:
                mouseInWindow_ = true;
                mouseX_ = event.xcrossing.x;
                mouseY_ = event.xcrossing.y;
                break;

            case LeaveNotify:
                mouseInWindow_ = false;
                break;
        }
    }

    int32_t WaitKey(int32_t timeoutMs) {
        if (!isOpen_ || !display_) return -1;

        // Calculate end time
        struct timeval startTime, currentTime;
        gettimeofday(&startTime, nullptr);

        while (isOpen_) {
            // Check for events
            while (XPending(display_)) {
                XEvent event;
                XNextEvent(display_, &event);

                switch (event.type) {
                    case Expose:
                        // Redraw
                        if (currentImage_.Width() > 0) {
                            Show(currentImage_, currentScaleMode_);
                        } else if (ximage_) {
                            XPutImage(display_, window_, gc_, ximage_,
                                      0, 0, displayOffsetX_, displayOffsetY_,
                                      imageWidth_, imageHeight_);
                            XSync(display_, False);
                        }
                        break;

                    case KeyPress: {
                        KeySym keysym = XLookupKeysym(&event.xkey, 0);
                        KeyModifier mods = GetModifiers(event.xkey.state);

                        // Call key callback if set
                        if (keyCallback_) {
                            keyCallback_(static_cast<int32_t>(keysym & 0xFFFF), mods);
                        }

                        // Handle double-click reset (not a key, but using 'f' for fit)
                        if (zoomPanEnabled_ && (keysym == XK_f || keysym == XK_F)) {
                            zoomLevel_ = 1.0;
                            panOffsetX_ = 0;
                            panOffsetY_ = 0;
                            if (currentImage_.Width() > 0) {
                                Show(currentImage_, currentScaleMode_);
                            }
                        }

                        // Convert to ASCII if possible
                        if (keysym >= XK_space && keysym <= XK_asciitilde) {
                            return static_cast<int32_t>(keysym);
                        }
                        // Return special keys as-is
                        return static_cast<int32_t>(keysym & 0xFFFF);
                    }

                    case ConfigureNotify:
                        if (width_ != event.xconfigure.width ||
                            height_ != event.xconfigure.height) {
                            width_ = event.xconfigure.width;
                            height_ = event.xconfigure.height;
                            // Redraw with new size
                            if (currentImage_.Width() > 0) {
                                Show(currentImage_, currentScaleMode_);
                            }
                        }
                        break;

                    case ClientMessage:
                        if (static_cast<Atom>(event.xclient.data.l[0]) == wmDeleteMessage_) {
                            isOpen_ = false;
                            return -1;
                        }
                        break;

                    case MotionNotify:
                    case ButtonPress:
                    case ButtonRelease:
                    case EnterNotify:
                    case LeaveNotify:
                        HandleMouseEvent(event);
                        break;
                }
            }

            // Check timeout
            if (timeoutMs > 0) {
                gettimeofday(&currentTime, nullptr);
                long elapsedMs = (currentTime.tv_sec - startTime.tv_sec) * 1000 +
                                 (currentTime.tv_usec - startTime.tv_usec) / 1000;
                if (elapsedMs >= timeoutMs) {
                    return -1;
                }
            } else if (timeoutMs < 0) {
                // No wait mode
                return -1;
            }

            // Small sleep to avoid busy waiting
            usleep(10000);  // 10ms
        }

        return -1;
    }

    void SetTitle(const std::string& title) {
        title_ = title;
        if (display_ && window_) {
            XStoreName(display_, window_, title_.c_str());
            XFlush(display_);
        }
    }

    void Resize(int32_t width, int32_t height) {
        width_ = width;
        height_ = height;
        if (display_ && window_) {
            XResizeWindow(display_, window_, width, height);
            XFlush(display_);
        }
    }

    void Move(int32_t x, int32_t y) {
        if (display_ && window_) {
            XMoveWindow(display_, window_, x, y);
            XFlush(display_);
        }
    }

    void SetAutoResize(bool enable, int32_t maxWidth, int32_t maxHeight) {
        autoResize_ = enable;
        maxWidth_ = maxWidth;
        maxHeight_ = maxHeight;
    }

    bool IsAutoResize() const {
        return autoResize_;
    }

    void SetResizable(bool resizable) {
        resizable_ = resizable;
        if (display_ && window_) {
            XSizeHints hints;
            hints.flags = PMinSize | PMaxSize;
            if (resizable) {
                // Allow any size
                hints.min_width = 1;
                hints.min_height = 1;
                hints.max_width = screenWidth_;
                hints.max_height = screenHeight_;
            } else {
                // Fix size to current dimensions
                hints.min_width = width_;
                hints.min_height = height_;
                hints.max_width = width_;
                hints.max_height = height_;
            }
            XSetWMNormalHints(display_, window_, &hints);
            XFlush(display_);
        }
    }

    bool IsResizable() const {
        return resizable_;
    }

    // Mouse interaction setters/getters
    void SetMouseCallback(MouseCallback cb) { mouseCallback_ = std::move(cb); }
    void SetKeyCallback(KeyCallback cb) { keyCallback_ = std::move(cb); }

    bool GetMousePosition(int32_t& x, int32_t& y) const {
        x = mouseX_;
        y = mouseY_;
        return mouseInWindow_;
    }

    bool GetMouseImagePosition(double& ix, double& iy) const {
        if (!mouseInWindow_) return false;
        double imgX = 0, imgY = 0;
        // Use const_cast for the const method
        bool inImage = const_cast<Impl*>(this)->WindowToImageCoords(mouseX_, mouseY_, imgX, imgY);
        ix = imgX;
        iy = imgY;
        return inImage;
    }

    // Pixel info display
    void EnablePixelInfo(bool enable) {
        pixelInfoEnabled_ = enable;
        if (!enable && display_ && window_) {
            XStoreName(display_, window_, baseTitle_.c_str());
            XFlush(display_);
        }
    }
    bool IsPixelInfoEnabled() const { return pixelInfoEnabled_; }

    // Zoom and pan
    void EnableZoomPan(bool enable) {
        zoomPanEnabled_ = enable;
        if (!enable) {
            zoomLevel_ = 1.0;
            panOffsetX_ = 0;
            panOffsetY_ = 0;
            isPanning_ = false;
        }
    }

    bool IsZoomPanEnabled() const { return zoomPanEnabled_; }
    double GetZoomLevel() const { return zoomLevel_; }

    void SetZoomLevel(double zoom, bool centerOnMouse) {
        double oldZoom = zoomLevel_;
        zoomLevel_ = std::max(0.1, std::min(100.0, zoom));

        if (centerOnMouse && mouseInWindow_) {
            double ix = 0, iy = 0;
            WindowToImageCoords(mouseX_, mouseY_, ix, iy);
            double newScale = displayScale_ * zoomLevel_ / oldZoom;
            panOffsetX_ = ix - (mouseX_ - displayOffsetX_) / newScale;
            panOffsetY_ = iy - (mouseY_ - displayOffsetY_) / newScale;
        }

        if (currentImage_.Width() > 0) {
            Show(currentImage_, currentScaleMode_);
        }
    }

    void GetPanOffset(double& ox, double& oy) const {
        ox = panOffsetX_;
        oy = panOffsetY_;
    }

    void SetPanOffset(double ox, double oy) {
        panOffsetX_ = ox;
        panOffsetY_ = oy;
        if (currentImage_.Width() > 0) {
            Show(currentImage_, currentScaleMode_);
        }
    }

    void ResetZoom() {
        zoomLevel_ = 1.0;
        panOffsetX_ = 0;
        panOffsetY_ = 0;
        if (currentImage_.Width() > 0) {
            Show(currentImage_, currentScaleMode_);
        }
    }

    void ZoomToRegion(double r1, double c1, double r2, double c2) {
        if (r2 <= r1 || c2 <= c1) return;

        double regionWidth = c2 - c1;
        double regionHeight = r2 - r1;

        double scaleX = width_ / regionWidth;
        double scaleY = height_ / regionHeight;
        zoomLevel_ = std::min(scaleX, scaleY);
        zoomLevel_ = std::max(0.1, std::min(100.0, zoomLevel_));

        panOffsetX_ = c1 + regionWidth / 2 - width_ / (2 * displayScale_ * zoomLevel_);
        panOffsetY_ = r1 + regionHeight / 2 - height_ / (2 * displayScale_ * zoomLevel_);

        if (currentImage_.Width() > 0) {
            Show(currentImage_, currentScaleMode_);
        }
    }

    bool WindowToImage(int32_t wx, int32_t wy, double& ix, double& iy) const {
        return const_cast<Impl*>(this)->WindowToImageCoords(wx, wy, ix, iy);
    }

    bool ImageToWindow(double ix, double iy, int32_t& wx, int32_t& wy) const {
        if (displayScale_ <= 0) return false;
        wx = static_cast<int32_t>((ix - panOffsetX_) * displayScale_ + displayOffsetX_);
        wy = static_cast<int32_t>((iy - panOffsetY_) * displayScale_ + displayOffsetY_);
        return wx >= 0 && wx < width_ && wy >= 0 && wy < height_;
    }

    // Draw XOR rectangle overlay on window (for ROI visualization)
    void DrawXORRect(int x1, int y1, int x2, int y2) {
        if (!display_ || !window_ || !gc_) return;

        // Use XOR mode for rubber-band drawing
        XSetFunction(display_, gc_, GXxor);
        XSetForeground(display_, gc_, 0xFFFFFF);  // White XOR
        XSetLineAttributes(display_, gc_, 2, LineSolid, CapButt, JoinMiter);

        // Normalize coordinates
        int rx = std::min(x1, x2);
        int ry = std::min(y1, y2);
        int rw = std::abs(x2 - x1);
        int rh = std::abs(y2 - y1);

        XDrawRectangle(display_, window_, gc_, rx, ry, rw, rh);
        XFlush(display_);

        // Restore normal drawing mode
        XSetFunction(display_, gc_, GXcopy);
    }

    // Draw XOR circle overlay
    void DrawXORCircle(int cx, int cy, int radius) {
        if (!display_ || !window_ || !gc_) return;

        XSetFunction(display_, gc_, GXxor);
        XSetForeground(display_, gc_, 0xFFFFFF);
        XSetLineAttributes(display_, gc_, 2, LineSolid, CapButt, JoinMiter);

        XDrawArc(display_, window_, gc_, cx - radius, cy - radius,
                 radius * 2, radius * 2, 0, 360 * 64);
        XFlush(display_);

        XSetFunction(display_, gc_, GXcopy);
    }

    // Draw XOR line overlay
    void DrawXORLine(int x1, int y1, int x2, int y2) {
        if (!display_ || !window_ || !gc_) return;

        XSetFunction(display_, gc_, GXxor);
        XSetForeground(display_, gc_, 0xFFFFFF);
        XSetLineAttributes(display_, gc_, 2, LineSolid, CapButt, JoinMiter);

        XDrawLine(display_, window_, gc_, x1, y1, x2, y2);
        XFlush(display_);

        XSetFunction(display_, gc_, GXcopy);
    }

    // ROI drawing with real-time feedback
    ROIResult DrawROI(ROIType type) {
        ROIResult result;
        result.type = type;
        result.valid = false;

        if (!isOpen_ || !display_) return result;

        bool drawing = false;
        int32_t startWinX = 0, startWinY = 0;  // Window coords of start point
        int32_t lastWinX = 0, lastWinY = 0;    // Last drawn position (for XOR erase)
        double startImgX = 0, startImgY = 0;   // Image coords of start point
        std::vector<Point2d> polygonPoints;
        bool hasLastDraw = false;

        while (isOpen_) {
            while (XPending(display_)) {
                XEvent event;
                XNextEvent(display_, &event);

                switch (event.type) {
                    case ButtonPress:
                        if (event.xbutton.button == Button1) {
                            double ix = 0, iy = 0;
                            WindowToImageCoords(event.xbutton.x, event.xbutton.y, ix, iy);

                            if (type == ROIType::Point) {
                                result.points.push_back(Point2d{ix, iy});
                                result.valid = true;
                                return result;
                            } else if (type == ROIType::Polygon) {
                                polygonPoints.push_back(Point2d{ix, iy});
                                // TODO: Draw polygon points
                            } else {
                                drawing = true;
                                startWinX = event.xbutton.x;
                                startWinY = event.xbutton.y;
                                startImgX = ix;
                                startImgY = iy;
                                lastWinX = startWinX;
                                lastWinY = startWinY;
                                hasLastDraw = false;
                            }
                        } else if (event.xbutton.button == Button3) {
                            // Right click cancels - erase any drawn shape first
                            if (hasLastDraw) {
                                if (type == ROIType::Rectangle || type == ROIType::RotatedRect) {
                                    DrawXORRect(startWinX, startWinY, lastWinX, lastWinY);
                                } else if (type == ROIType::Circle || type == ROIType::Ellipse) {
                                    int radius = static_cast<int>(std::sqrt(
                                        (lastWinX - startWinX) * (lastWinX - startWinX) +
                                        (lastWinY - startWinY) * (lastWinY - startWinY)));
                                    DrawXORCircle(startWinX, startWinY, radius);
                                } else if (type == ROIType::Line) {
                                    DrawXORLine(startWinX, startWinY, lastWinX, lastWinY);
                                }
                            }
                            return result;
                        }
                        break;

                    case MotionNotify:
                        if (drawing) {
                            int curX = event.xmotion.x;
                            int curY = event.xmotion.y;

                            // Erase previous shape (XOR again = erase)
                            if (hasLastDraw) {
                                if (type == ROIType::Rectangle || type == ROIType::RotatedRect) {
                                    DrawXORRect(startWinX, startWinY, lastWinX, lastWinY);
                                } else if (type == ROIType::Circle || type == ROIType::Ellipse) {
                                    int radius = static_cast<int>(std::sqrt(
                                        (lastWinX - startWinX) * (lastWinX - startWinX) +
                                        (lastWinY - startWinY) * (lastWinY - startWinY)));
                                    DrawXORCircle(startWinX, startWinY, radius);
                                } else if (type == ROIType::Line) {
                                    DrawXORLine(startWinX, startWinY, lastWinX, lastWinY);
                                }
                            }

                            // Draw new shape
                            if (type == ROIType::Rectangle || type == ROIType::RotatedRect) {
                                DrawXORRect(startWinX, startWinY, curX, curY);
                            } else if (type == ROIType::Circle || type == ROIType::Ellipse) {
                                int radius = static_cast<int>(std::sqrt(
                                    (curX - startWinX) * (curX - startWinX) +
                                    (curY - startWinY) * (curY - startWinY)));
                                DrawXORCircle(startWinX, startWinY, radius);
                            } else if (type == ROIType::Line) {
                                DrawXORLine(startWinX, startWinY, curX, curY);
                            }

                            lastWinX = curX;
                            lastWinY = curY;
                            hasLastDraw = true;
                        }
                        break;

                    case ButtonRelease:
                        if (event.xbutton.button == Button1 && drawing) {
                            // Erase final XOR shape
                            if (hasLastDraw) {
                                if (type == ROIType::Rectangle || type == ROIType::RotatedRect) {
                                    DrawXORRect(startWinX, startWinY, lastWinX, lastWinY);
                                } else if (type == ROIType::Circle || type == ROIType::Ellipse) {
                                    int radius = static_cast<int>(std::sqrt(
                                        (lastWinX - startWinX) * (lastWinX - startWinX) +
                                        (lastWinY - startWinY) * (lastWinY - startWinY)));
                                    DrawXORCircle(startWinX, startWinY, radius);
                                } else if (type == ROIType::Line) {
                                    DrawXORLine(startWinX, startWinY, lastWinX, lastWinY);
                                }
                            }

                            double ix = 0, iy = 0;
                            WindowToImageCoords(event.xbutton.x, event.xbutton.y, ix, iy);

                            switch (type) {
                                case ROIType::Rectangle:
                                case ROIType::RotatedRect:
                                    result.row1 = std::min(startImgY, iy);
                                    result.col1 = std::min(startImgX, ix);
                                    result.row2 = std::max(startImgY, iy);
                                    result.col2 = std::max(startImgX, ix);
                                    result.valid = true;
                                    return result;

                                case ROIType::Circle:
                                case ROIType::Ellipse:
                                    result.centerRow = startImgY;
                                    result.centerCol = startImgX;
                                    result.radius = std::sqrt((ix - startImgX) * (ix - startImgX) +
                                                             (iy - startImgY) * (iy - startImgY));
                                    result.valid = true;
                                    return result;

                                case ROIType::Line:
                                    result.startRow = startImgY;
                                    result.startCol = startImgX;
                                    result.endRow = iy;
                                    result.endCol = ix;
                                    result.valid = true;
                                    return result;

                                default:
                                    break;
                            }
                        }
                        break;

                    case KeyPress: {
                        KeySym keysym = XLookupKeysym(&event.xkey, 0);
                        if (keysym == XK_Escape) {
                            // Erase any drawn shape before returning
                            if (hasLastDraw) {
                                if (type == ROIType::Rectangle || type == ROIType::RotatedRect) {
                                    DrawXORRect(startWinX, startWinY, lastWinX, lastWinY);
                                } else if (type == ROIType::Circle || type == ROIType::Ellipse) {
                                    int radius = static_cast<int>(std::sqrt(
                                        (lastWinX - startWinX) * (lastWinX - startWinX) +
                                        (lastWinY - startWinY) * (lastWinY - startWinY)));
                                    DrawXORCircle(startWinX, startWinY, radius);
                                } else if (type == ROIType::Line) {
                                    DrawXORLine(startWinX, startWinY, lastWinX, lastWinY);
                                }
                            }
                            return result; // Cancel
                        }
                        if (keysym == XK_Return && type == ROIType::Polygon && polygonPoints.size() >= 3) {
                            result.points = polygonPoints;
                            result.valid = true;
                            return result;
                        }
                        if (keysym == XK_BackSpace && type == ROIType::Polygon && !polygonPoints.empty()) {
                            polygonPoints.pop_back();
                        }
                        break;
                    }

                    case ClientMessage:
                        if (static_cast<Atom>(event.xclient.data.l[0]) == wmDeleteMessage_) {
                            isOpen_ = false;
                            return result;
                        }
                        break;

                    case Expose:
                        // Redraw image on expose
                        if (currentImage_.Width() > 0) {
                            Show(currentImage_, currentScaleMode_);
                            // Redraw current ROI shape if drawing
                            if (drawing && hasLastDraw) {
                                if (type == ROIType::Rectangle || type == ROIType::RotatedRect) {
                                    DrawXORRect(startWinX, startWinY, lastWinX, lastWinY);
                                } else if (type == ROIType::Circle || type == ROIType::Ellipse) {
                                    int radius = static_cast<int>(std::sqrt(
                                        (lastWinX - startWinX) * (lastWinX - startWinX) +
                                        (lastWinY - startWinY) * (lastWinY - startWinY)));
                                    DrawXORCircle(startWinX, startWinY, radius);
                                } else if (type == ROIType::Line) {
                                    DrawXORLine(startWinX, startWinY, lastWinX, lastWinY);
                                }
                            }
                        }
                        break;

                    default:
                        break;
                }
            }
            usleep(10000);
        }

        return result;
    }
};

#endif // QIVISION_PLATFORM_LINUX && QIVISION_HAS_X11

#ifdef QIVISION_PLATFORM_WINDOWS

class Window::Impl {
public:
    HWND hwnd_ = nullptr;
    HDC hdc_ = nullptr;
    HBITMAP hbitmap_ = nullptr;
    HDC memDC_ = nullptr;

    int32_t width_ = 0;
    int32_t height_ = 0;
    bool isOpen_ = false;
    std::string title_;

    std::vector<uint8_t> buffer_;
    int32_t imageWidth_ = 0;
    int32_t imageHeight_ = 0;

    // Original image dimensions
    int32_t srcImageWidth_ = 0;
    int32_t srcImageHeight_ = 0;

    // Auto-resize settings
    bool autoResize_ = false;
    int32_t maxWidth_ = 0;
    int32_t maxHeight_ = 0;
    int32_t screenWidth_ = 0;
    int32_t screenHeight_ = 0;

    // Resizable setting
    bool resizable_ = true;

    // Mouse interaction
    MouseCallback mouseCallback_;
    KeyCallback keyCallback_;
    int32_t mouseX_ = 0;
    int32_t mouseY_ = 0;
    bool mouseInWindow_ = false;

    // Zoom and pan
    bool zoomPanEnabled_ = false;
    double zoomLevel_ = 1.0;
    double panOffsetX_ = 0.0;
    double panOffsetY_ = 0.0;
    bool isPanning_ = false;
    int32_t panStartX_ = 0;
    int32_t panStartY_ = 0;
    double panStartOffsetX_ = 0.0;
    double panStartOffsetY_ = 0.0;

    // Display offset (for centering)
    int32_t displayOffsetX_ = 0;
    int32_t displayOffsetY_ = 0;
    double displayScale_ = 1.0;

    // Pixel info display
    bool pixelInfoEnabled_ = false;
    std::string baseTitle_;

    // Store the original image for redraw
    QImage currentImage_;
    ScaleMode currentScaleMode_ = ScaleMode::Fit;

    static LRESULT CALLBACK WindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
        Impl* impl = reinterpret_cast<Impl*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));

        switch (msg) {
            case WM_DESTROY:
                if (impl) impl->isOpen_ = false;
                return 0;

            case WM_SIZE:
                if (impl) {
                    impl->width_ = LOWORD(lParam);
                    impl->height_ = HIWORD(lParam);
                    // Redraw with new size
                    if (impl->currentImage_.Width() > 0) {
                        impl->Show(impl->currentImage_, impl->currentScaleMode_);
                    }
                }
                return 0;

            case WM_PAINT: {
                PAINTSTRUCT ps;
                HDC hdc = BeginPaint(hwnd, &ps);
                if (impl && impl->memDC_) {
                    // Clear background
                    RECT rect;
                    GetClientRect(hwnd, &rect);
                    FillRect(hdc, &rect, (HBRUSH)GetStockObject(BLACK_BRUSH));

                    // Draw image
                    BitBlt(hdc, impl->displayOffsetX_, impl->displayOffsetY_,
                           impl->imageWidth_, impl->imageHeight_,
                           impl->memDC_, 0, 0, SRCCOPY);
                }
                EndPaint(hwnd, &ps);
                return 0;
            }

            case WM_MOUSEMOVE:
                if (impl) {
                    impl->mouseX_ = LOWORD(lParam);
                    impl->mouseY_ = HIWORD(lParam);
                    impl->mouseInWindow_ = true;

                    // Handle panning
                    if (impl->isPanning_ && impl->zoomPanEnabled_) {
                        double dx = (impl->mouseX_ - impl->panStartX_) / impl->displayScale_;
                        double dy = (impl->mouseY_ - impl->panStartY_) / impl->displayScale_;
                        impl->panOffsetX_ = impl->panStartOffsetX_ - dx;
                        impl->panOffsetY_ = impl->panStartOffsetY_ - dy;
                        if (impl->currentImage_.Width() > 0) {
                            impl->Show(impl->currentImage_, impl->currentScaleMode_);
                        }
                    }

                    if (impl->mouseCallback_) {
                        MouseEvent evt = impl->CreateMouseEvent(MouseEventType::Move,
                            impl->mouseX_, impl->mouseY_, MouseButton::None, wParam);
                        impl->mouseCallback_(evt);
                    }
                }
                return 0;

            case WM_LBUTTONDOWN:
            case WM_RBUTTONDOWN:
            case WM_MBUTTONDOWN:
                if (impl) {
                    int x = LOWORD(lParam);
                    int y = HIWORD(lParam);
                    MouseButton btn = (msg == WM_LBUTTONDOWN) ? MouseButton::Left :
                                      (msg == WM_RBUTTONDOWN) ? MouseButton::Right : MouseButton::Middle;

                    if (impl->zoomPanEnabled_) {
                        if (btn == MouseButton::Left) {
                            impl->isPanning_ = true;
                            impl->panStartX_ = x;
                            impl->panStartY_ = y;
                            impl->panStartOffsetX_ = impl->panOffsetX_;
                            impl->panStartOffsetY_ = impl->panOffsetY_;
                            SetCapture(hwnd);
                        } else if (btn == MouseButton::Right) {
                            impl->zoomLevel_ = 1.0;
                            impl->panOffsetX_ = 0;
                            impl->panOffsetY_ = 0;
                            if (impl->currentImage_.Width() > 0) {
                                impl->Show(impl->currentImage_, impl->currentScaleMode_);
                            }
                        }
                    }

                    if (impl->mouseCallback_) {
                        MouseEvent evt = impl->CreateMouseEvent(MouseEventType::ButtonDown, x, y, btn, wParam);
                        impl->mouseCallback_(evt);
                    }
                }
                return 0;

            case WM_LBUTTONUP:
            case WM_RBUTTONUP:
            case WM_MBUTTONUP:
                if (impl) {
                    int x = LOWORD(lParam);
                    int y = HIWORD(lParam);
                    MouseButton btn = (msg == WM_LBUTTONUP) ? MouseButton::Left :
                                      (msg == WM_RBUTTONUP) ? MouseButton::Right : MouseButton::Middle;

                    if (btn == MouseButton::Left) {
                        impl->isPanning_ = false;
                        ReleaseCapture();
                    }

                    if (impl->mouseCallback_) {
                        MouseEvent evt = impl->CreateMouseEvent(MouseEventType::ButtonUp, x, y, btn, wParam);
                        impl->mouseCallback_(evt);
                    }
                }
                return 0;

            case WM_MOUSEWHEEL:
                if (impl && impl->zoomPanEnabled_) {
                    int delta = GET_WHEEL_DELTA_WPARAM(wParam);
                    POINT pt = { LOWORD(lParam), HIWORD(lParam) };
                    ScreenToClient(hwnd, &pt);

                    double ix, iy;
                    impl->WindowToImageCoords(pt.x, pt.y, ix, iy);

                    double oldZoom = impl->zoomLevel_;
                    if (delta > 0) {
                        impl->zoomLevel_ *= 1.2;
                    } else {
                        impl->zoomLevel_ /= 1.2;
                    }
                    impl->zoomLevel_ = std::max(0.1, std::min(100.0, impl->zoomLevel_));

                    double newScale = impl->displayScale_ * impl->zoomLevel_ / oldZoom;
                    impl->panOffsetX_ = ix - (pt.x - impl->displayOffsetX_) / newScale;
                    impl->panOffsetY_ = iy - (pt.y - impl->displayOffsetY_) / newScale;

                    if (impl->currentImage_.Width() > 0) {
                        impl->Show(impl->currentImage_, impl->currentScaleMode_);
                    }

                    if (impl->mouseCallback_) {
                        MouseButton btn = (delta > 0) ? MouseButton::WheelUp : MouseButton::WheelDown;
                        MouseEvent evt = impl->CreateMouseEvent(MouseEventType::Wheel, pt.x, pt.y, btn, wParam, delta);
                        impl->mouseCallback_(evt);
                    }
                }
                return 0;

            case WM_MOUSELEAVE:
                if (impl) impl->mouseInWindow_ = false;
                return 0;
        }
        return DefWindowProc(hwnd, msg, wParam, lParam);
    }

    // Helper to get modifiers from Windows state
    KeyModifier GetModifiers(WPARAM wParam) {
        KeyModifier mods = KeyModifier::None;
        if (wParam & MK_SHIFT) mods = mods | KeyModifier::Shift;
        if (wParam & MK_CONTROL) mods = mods | KeyModifier::Ctrl;
        if (GetKeyState(VK_MENU) & 0x8000) mods = mods | KeyModifier::Alt;
        return mods;
    }

    // Convert window coords to image coords
    bool WindowToImageCoords(int32_t wx, int32_t wy, double& ix, double& iy) {
        if (displayScale_ <= 0 || srcImageWidth_ <= 0 || srcImageHeight_ <= 0) {
            return false;
        }
        ix = (wx - displayOffsetX_) / displayScale_ + panOffsetX_;
        iy = (wy - displayOffsetY_) / displayScale_ + panOffsetY_;
        return ix >= 0 && ix < srcImageWidth_ && iy >= 0 && iy < srcImageHeight_;
    }

    // Create mouse event
    MouseEvent CreateMouseEvent(MouseEventType type, int x, int y,
                                MouseButton button, WPARAM wParam, int wheelDelta = 0) {
        MouseEvent evt;
        evt.type = type;
        evt.button = button;
        evt.x = x;
        evt.y = y;
        evt.wheelDelta = wheelDelta;
        evt.modifiers = GetModifiers(wParam);
        WindowToImageCoords(x, y, evt.imageX, evt.imageY);
        return evt;
    }

    Impl(const std::string& title, int32_t width, int32_t height)
        : width_(width), height_(height), title_(title) {

        // Get screen size for auto-resize limits
        screenWidth_ = GetSystemMetrics(SM_CXSCREEN);
        screenHeight_ = GetSystemMetrics(SM_CYSCREEN);

        if (width_ <= 0) width_ = 800;
        if (height_ <= 0) height_ = 600;

        // Register window class
        static bool classRegistered = false;
        static const char* className = "QiVisionWindow";

        if (!classRegistered) {
            WNDCLASSA wc = {};
            wc.lpfnWndProc = WindowProc;
            wc.hInstance = GetModuleHandle(nullptr);
            wc.lpszClassName = className;
            wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
            wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
            RegisterClassA(&wc);
            classRegistered = true;
        }

        // Adjust window size to account for borders
        RECT rect = {0, 0, width_, height_};
        AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);

        hwnd_ = CreateWindowA(
            className, title_.c_str(),
            WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT, CW_USEDEFAULT,
            rect.right - rect.left, rect.bottom - rect.top,
            nullptr, nullptr, GetModuleHandle(nullptr), nullptr
        );

        if (!hwnd_) return;

        SetWindowLongPtr(hwnd_, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(this));

        hdc_ = GetDC(hwnd_);
        memDC_ = CreateCompatibleDC(hdc_);

        ShowWindow(hwnd_, SW_SHOW);
        UpdateWindow(hwnd_);

        isOpen_ = true;
    }

    ~Impl() {
        Close();
    }

    void Close() {
        if (hbitmap_) {
            DeleteObject(hbitmap_);
            hbitmap_ = nullptr;
        }
        if (memDC_) {
            DeleteDC(memDC_);
            memDC_ = nullptr;
        }
        if (hdc_ && hwnd_) {
            ReleaseDC(hwnd_, hdc_);
            hdc_ = nullptr;
        }
        if (hwnd_) {
            DestroyWindow(hwnd_);
            hwnd_ = nullptr;
        }
        isOpen_ = false;
    }

    void Show(const QImage& image, ScaleMode scaleMode) {
        if (!isOpen_ || !hwnd_) return;

        // Store for redraw
        currentImage_ = image.Clone();
        currentScaleMode_ = scaleMode;

        int32_t srcWidth = image.Width();
        int32_t srcHeight = image.Height();
        srcImageWidth_ = srcWidth;
        srcImageHeight_ = srcHeight;

        // Auto-resize window to fit image
        if (autoResize_) {
            int32_t maxW = (maxWidth_ > 0) ? maxWidth_ : (screenWidth_ - 100);
            int32_t maxH = (maxHeight_ > 0) ? maxHeight_ : (screenHeight_ - 100);

            int32_t newWidth = srcWidth;
            int32_t newHeight = srcHeight;

            // Scale down if image is larger than max size
            if (newWidth > maxW || newHeight > maxH) {
                double scaleW = static_cast<double>(maxW) / srcWidth;
                double scaleH = static_cast<double>(maxH) / srcHeight;
                double scale = std::min(scaleW, scaleH);
                newWidth = static_cast<int32_t>(srcWidth * scale);
                newHeight = static_cast<int32_t>(srcHeight * scale);
            }

            // Resize window if size changed
            if (newWidth != width_ || newHeight != height_) {
                width_ = newWidth;
                height_ = newHeight;
                RECT rect = {0, 0, width_, height_};
                AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);
                SetWindowPos(hwnd_, nullptr, 0, 0,
                             rect.right - rect.left, rect.bottom - rect.top,
                             SWP_NOMOVE | SWP_NOZORDER);
            }
        }

        int32_t dstWidth = width_;
        int32_t dstHeight = height_;

        double scaleX = 1.0, scaleY = 1.0;

        switch (scaleMode) {
            case ScaleMode::None:
                dstWidth = srcWidth;
                dstHeight = srcHeight;
                break;

            case ScaleMode::Fit: {
                double scaleW = static_cast<double>(width_) / srcWidth;
                double scaleH = static_cast<double>(height_) / srcHeight;
                double scale = std::min(scaleW, scaleH);
                dstWidth = static_cast<int32_t>(srcWidth * scale);
                dstHeight = static_cast<int32_t>(srcHeight * scale);
                scaleX = scaleY = scale;
                break;
            }

            case ScaleMode::Fill: {
                double scaleW = static_cast<double>(width_) / srcWidth;
                double scaleH = static_cast<double>(height_) / srcHeight;
                double scale = std::max(scaleW, scaleH);
                scaleX = scaleY = scale;
                dstWidth = width_;
                dstHeight = height_;
                break;
            }

            case ScaleMode::Stretch:
                scaleX = static_cast<double>(width_) / srcWidth;
                scaleY = static_cast<double>(height_) / srcHeight;
                dstWidth = width_;
                dstHeight = height_;
                break;
        }

        // Apply zoom if enabled
        if (zoomPanEnabled_) {
            scaleX *= zoomLevel_;
            scaleY *= zoomLevel_;
            dstWidth = static_cast<int32_t>(srcWidth * scaleX);
            dstHeight = static_cast<int32_t>(srcHeight * scaleY);
        }

        imageWidth_ = dstWidth;
        imageHeight_ = dstHeight;
        displayScale_ = scaleX;

        // Calculate display offset
        displayOffsetX_ = (width_ - dstWidth) / 2;
        displayOffsetY_ = (height_ - dstHeight) / 2;

        if (zoomPanEnabled_) {
            displayOffsetX_ = static_cast<int>(displayOffsetX_ - panOffsetX_ * displayScale_);
            displayOffsetY_ = static_cast<int>(displayOffsetY_ - panOffsetY_ * displayScale_);
        }

        // Create DIB
        BITMAPINFO bmi = {};
        bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bmi.bmiHeader.biWidth = dstWidth;
        bmi.bmiHeader.biHeight = -dstHeight;  // Top-down
        bmi.bmiHeader.biPlanes = 1;
        bmi.bmiHeader.biBitCount = 32;
        bmi.bmiHeader.biCompression = BI_RGB;

        void* bits = nullptr;
        HBITMAP newBitmap = CreateDIBSection(hdc_, &bmi, DIB_RGB_COLORS, &bits, nullptr, 0);
        if (!newBitmap) return;

        // Scale and convert to BGRA
        const uint8_t* srcData = static_cast<const uint8_t*>(image.Data());
        size_t srcStride = image.Stride();
        int channels = image.Channels();
        uint8_t* dstData = static_cast<uint8_t*>(bits);

        for (int32_t dy = 0; dy < dstHeight; ++dy) {
            int32_t sy = static_cast<int32_t>(dy / scaleY);
            if (sy >= srcHeight) sy = srcHeight - 1;

            for (int32_t dx = 0; dx < dstWidth; ++dx) {
                int32_t sx = static_cast<int32_t>(dx / scaleX);
                if (sx >= srcWidth) sx = srcWidth - 1;

                uint8_t* dst = &dstData[(dy * dstWidth + dx) * 4];

                if (channels == 1) {
                    uint8_t gray = srcData[sy * srcStride + sx];
                    dst[0] = gray;  // B
                    dst[1] = gray;  // G
                    dst[2] = gray;  // R
                    dst[3] = 255;   // A
                } else if (channels == 3) {
                    const uint8_t* src = &srcData[sy * srcStride + sx * 3];
                    dst[0] = src[2];  // B
                    dst[1] = src[1];  // G
                    dst[2] = src[0];  // R
                    dst[3] = 255;     // A
                }
            }
        }

        // Replace bitmap
        if (hbitmap_) DeleteObject(hbitmap_);
        hbitmap_ = newBitmap;
        SelectObject(memDC_, hbitmap_);

        // Redraw
        InvalidateRect(hwnd_, nullptr, TRUE);
        UpdateWindow(hwnd_);
    }

    int32_t WaitKey(int32_t timeoutMs) {
        if (!isOpen_ || !hwnd_) return -1;

        DWORD startTime = GetTickCount();
        MSG msg;

        while (isOpen_) {
            // Process messages
            while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
                if (msg.message == WM_QUIT) {
                    isOpen_ = false;
                    return -1;
                }

                if (msg.message == WM_KEYDOWN && msg.hwnd == hwnd_) {
                    // Convert virtual key to character
                    int key = static_cast<int>(msg.wParam);
                    if (key >= 'A' && key <= 'Z') {
                        // Check if shift is pressed
                        if (!(GetKeyState(VK_SHIFT) & 0x8000)) {
                            key = key - 'A' + 'a';  // Convert to lowercase
                        }
                    }
                    return key;
                }

                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }

            // Check timeout
            if (timeoutMs > 0) {
                DWORD elapsed = GetTickCount() - startTime;
                if (elapsed >= static_cast<DWORD>(timeoutMs)) {
                    return -1;
                }
            } else if (timeoutMs < 0) {
                return -1;
            }

            Sleep(10);  // Avoid busy waiting
        }

        return -1;
    }

    void SetTitle(const std::string& title) {
        title_ = title;
        if (hwnd_) {
            SetWindowTextA(hwnd_, title_.c_str());
        }
    }

    void Resize(int32_t width, int32_t height) {
        width_ = width;
        height_ = height;
        if (hwnd_) {
            RECT rect = {0, 0, width, height};
            AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);
            SetWindowPos(hwnd_, nullptr, 0, 0,
                         rect.right - rect.left, rect.bottom - rect.top,
                         SWP_NOMOVE | SWP_NOZORDER);
        }
    }

    void Move(int32_t x, int32_t y) {
        if (hwnd_) {
            SetWindowPos(hwnd_, nullptr, x, y, 0, 0, SWP_NOSIZE | SWP_NOZORDER);
        }
    }

    void SetAutoResize(bool enable, int32_t maxWidth, int32_t maxHeight) {
        autoResize_ = enable;
        maxWidth_ = maxWidth;
        maxHeight_ = maxHeight;
    }

    bool IsAutoResize() const {
        return autoResize_;
    }

    void SetResizable(bool resizable) {
        resizable_ = resizable;
        if (hwnd_) {
            LONG style = GetWindowLong(hwnd_, GWL_STYLE);
            if (resizable) {
                style |= WS_SIZEBOX | WS_MAXIMIZEBOX;
            } else {
                style &= ~(WS_SIZEBOX | WS_MAXIMIZEBOX);
            }
            SetWindowLong(hwnd_, GWL_STYLE, style);
            // Force redraw of window frame
            SetWindowPos(hwnd_, nullptr, 0, 0, 0, 0,
                         SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED);
        }
    }

    bool IsResizable() const {
        return resizable_;
    }

    // Mouse interaction setters/getters
    void SetMouseCallback(MouseCallback cb) { mouseCallback_ = std::move(cb); }
    void SetKeyCallback(KeyCallback cb) { keyCallback_ = std::move(cb); }

    bool GetMousePosition(int32_t& x, int32_t& y) const {
        x = mouseX_;
        y = mouseY_;
        return mouseInWindow_;
    }

    bool GetMouseImagePosition(double& ix, double& iy) const {
        if (!mouseInWindow_) return false;
        double imgX, imgY;
        bool inImage = const_cast<Impl*>(this)->WindowToImageCoords(mouseX_, mouseY_, imgX, imgY);
        ix = imgX;
        iy = imgY;
        return inImage;
    }

    // Pixel info display
    void EnablePixelInfo(bool enable) {
        pixelInfoEnabled_ = enable;
        if (!enable && hwnd_) {
            SetWindowTextA(hwnd_, baseTitle_.c_str());
        }
    }
    bool IsPixelInfoEnabled() const { return pixelInfoEnabled_; }

    // Zoom and pan
    void EnableZoomPan(bool enable) {
        zoomPanEnabled_ = enable;
        if (!enable) {
            zoomLevel_ = 1.0;
            panOffsetX_ = 0;
            panOffsetY_ = 0;
            isPanning_ = false;
        }
    }

    bool IsZoomPanEnabled() const { return zoomPanEnabled_; }
    double GetZoomLevel() const { return zoomLevel_; }

    void SetZoomLevel(double zoom, bool centerOnMouse) {
        double oldZoom = zoomLevel_;
        zoomLevel_ = std::max(0.1, std::min(100.0, zoom));

        if (centerOnMouse && mouseInWindow_) {
            double ix = 0, iy = 0;
            WindowToImageCoords(mouseX_, mouseY_, ix, iy);
            double newScale = displayScale_ * zoomLevel_ / oldZoom;
            panOffsetX_ = ix - (mouseX_ - displayOffsetX_) / newScale;
            panOffsetY_ = iy - (mouseY_ - displayOffsetY_) / newScale;
        }

        if (currentImage_.Width() > 0) {
            Show(currentImage_, currentScaleMode_);
        }
    }

    void GetPanOffset(double& ox, double& oy) const {
        ox = panOffsetX_;
        oy = panOffsetY_;
    }

    void SetPanOffset(double ox, double oy) {
        panOffsetX_ = ox;
        panOffsetY_ = oy;
        if (currentImage_.Width() > 0) {
            Show(currentImage_, currentScaleMode_);
        }
    }

    void ResetZoom() {
        zoomLevel_ = 1.0;
        panOffsetX_ = 0;
        panOffsetY_ = 0;
        if (currentImage_.Width() > 0) {
            Show(currentImage_, currentScaleMode_);
        }
    }

    void ZoomToRegion(double r1, double c1, double r2, double c2) {
        if (r2 <= r1 || c2 <= c1) return;

        double regionWidth = c2 - c1;
        double regionHeight = r2 - r1;

        double scaleX = width_ / regionWidth;
        double scaleY = height_ / regionHeight;
        zoomLevel_ = std::min(scaleX, scaleY);
        zoomLevel_ = std::max(0.1, std::min(100.0, zoomLevel_));

        panOffsetX_ = c1 + regionWidth / 2 - width_ / (2 * displayScale_ * zoomLevel_);
        panOffsetY_ = r1 + regionHeight / 2 - height_ / (2 * displayScale_ * zoomLevel_);

        if (currentImage_.Width() > 0) {
            Show(currentImage_, currentScaleMode_);
        }
    }

    bool WindowToImage(int32_t wx, int32_t wy, double& ix, double& iy) const {
        return const_cast<Impl*>(this)->WindowToImageCoords(wx, wy, ix, iy);
    }

    bool ImageToWindow(double ix, double iy, int32_t& wx, int32_t& wy) const {
        if (displayScale_ <= 0) return false;
        wx = static_cast<int32_t>((ix - panOffsetX_) * displayScale_ + displayOffsetX_);
        wy = static_cast<int32_t>((iy - panOffsetY_) * displayScale_ + displayOffsetY_);
        return wx >= 0 && wx < width_ && wy >= 0 && wy < height_;
    }

    // ROI drawing - placeholder (Windows implementation would be similar to X11)
    ROIResult DrawROI(ROIType type) {
        ROIResult result;
        result.type = type;
        result.valid = false;
        // TODO: Implement Windows ROI drawing
        return result;
    }
};

#endif // QIVISION_PLATFORM_WINDOWS

// =============================================================================
// Stub implementation (no GUI available)
// =============================================================================

#if !defined(QIVISION_PLATFORM_WINDOWS) && !(defined(QIVISION_PLATFORM_LINUX) && defined(QIVISION_HAS_X11))

class Window::Impl {
public:
    int32_t width_ = 0;
    int32_t height_ = 0;
    bool isOpen_ = false;
    std::string title_;
    bool autoResize_ = false;

    Impl(const std::string& title, int32_t width, int32_t height)
        : width_(width > 0 ? width : 800)
        , height_(height > 0 ? height : 600)
        , title_(title)
        , isOpen_(true) {
        // Stub: No actual window created
    }

    ~Impl() = default;

    void Close() { isOpen_ = false; }

    void Show(const QImage& /*image*/, ScaleMode /*scaleMode*/) {
        // Stub: No-op
    }

    int32_t WaitKey(int32_t /*timeoutMs*/) {
        // Stub: Return immediately
        return -1;
    }

    void SetTitle(const std::string& title) { title_ = title; }
    void Resize(int32_t width, int32_t height) { width_ = width; height_ = height; }
    void Move(int32_t /*x*/, int32_t /*y*/) {}
    void SetAutoResize(bool enable, int32_t /*maxWidth*/, int32_t /*maxHeight*/) { autoResize_ = enable; }
    bool IsAutoResize() const { return autoResize_; }
    void SetResizable(bool) {}
    bool IsResizable() const { return true; }

    // Stub implementations for mouse interaction
    void SetMouseCallback(MouseCallback) {}
    void SetKeyCallback(KeyCallback) {}
    bool GetMousePosition(int32_t& x, int32_t& y) const { x = y = 0; return false; }
    bool GetMouseImagePosition(double& ix, double& iy) const { ix = iy = 0; return false; }

    // Stub implementations for pixel info
    void EnablePixelInfo(bool) {}
    bool IsPixelInfoEnabled() const { return false; }

    // Stub implementations for zoom/pan
    void EnableZoomPan(bool) {}
    bool IsZoomPanEnabled() const { return false; }
    double GetZoomLevel() const { return 1.0; }
    void SetZoomLevel(double, bool) {}
    void GetPanOffset(double& ox, double& oy) const { ox = oy = 0; }
    void SetPanOffset(double, double) {}
    void ResetZoom() {}
    void ZoomToRegion(double, double, double, double) {}
    bool WindowToImage(int32_t, int32_t, double& ix, double& iy) const { ix = iy = 0; return false; }
    bool ImageToWindow(double, double, int32_t& wx, int32_t& wy) const { wx = wy = 0; return false; }

    // Stub ROI drawing
    ROIResult DrawROI(ROIType type) {
        ROIResult result;
        result.type = type;
        result.valid = false;
        return result;
    }
};

#endif // Stub implementation

// =============================================================================
// Window class implementation
// =============================================================================

Window::Window(const std::string& title, int32_t width, int32_t height)
    : impl_(std::make_unique<Impl>(title, width, height)) {}

Window::~Window() = default;

Window::Window(Window&& other) noexcept = default;
Window& Window::operator=(Window&& other) noexcept = default;

void Window::DispImage(const QImage& image, ScaleMode scaleMode) {
    if (impl_) impl_->Show(image, scaleMode);
}

int32_t Window::WaitKey(int32_t timeoutMs) {
    return impl_ ? impl_->WaitKey(timeoutMs) : -1;
}

bool Window::IsOpen() const {
    return impl_ && impl_->isOpen_;
}

void Window::Close() {
    if (impl_) impl_->Close();
}

void Window::SetTitle(const std::string& title) {
    if (impl_) impl_->SetTitle(title);
}

void Window::Resize(int32_t width, int32_t height) {
    if (impl_) impl_->Resize(width, height);
}

void Window::GetSize(int32_t& width, int32_t& height) const {
    if (impl_) {
        width = impl_->width_;
        height = impl_->height_;
    } else {
        width = height = 0;
    }
}

void Window::Move(int32_t x, int32_t y) {
    if (impl_) impl_->Move(x, y);
}

void Window::SetAutoResize(bool enable, int32_t maxWidth, int32_t maxHeight) {
    if (impl_) impl_->SetAutoResize(enable, maxWidth, maxHeight);
}

bool Window::IsAutoResize() const {
    return impl_ ? impl_->IsAutoResize() : false;
}

void Window::SetResizable(bool resizable) {
    if (impl_) impl_->SetResizable(resizable);
}

bool Window::IsResizable() const {
    return impl_ ? impl_->IsResizable() : true;
}

// Mouse interaction
void Window::SetMouseCallback(MouseCallback callback) {
    if (impl_) impl_->SetMouseCallback(std::move(callback));
}

void Window::SetKeyCallback(KeyCallback callback) {
    if (impl_) impl_->SetKeyCallback(std::move(callback));
}

bool Window::GetMousePosition(int32_t& x, int32_t& y) const {
    return impl_ ? impl_->GetMousePosition(x, y) : false;
}

bool Window::GetMouseImagePosition(double& imageX, double& imageY) const {
    return impl_ ? impl_->GetMouseImagePosition(imageX, imageY) : false;
}

// Pixel info display
void Window::EnablePixelInfo(bool enable) {
    if (impl_) impl_->EnablePixelInfo(enable);
}

bool Window::IsPixelInfoEnabled() const {
    return impl_ ? impl_->IsPixelInfoEnabled() : false;
}

// Zoom and pan
void Window::EnableZoomPan(bool enable) {
    if (impl_) impl_->EnableZoomPan(enable);
}

bool Window::IsZoomPanEnabled() const {
    return impl_ ? impl_->IsZoomPanEnabled() : false;
}

double Window::GetZoomLevel() const {
    return impl_ ? impl_->GetZoomLevel() : 1.0;
}

void Window::SetZoomLevel(double zoom, bool centerOnMouse) {
    if (impl_) impl_->SetZoomLevel(zoom, centerOnMouse);
}

void Window::GetPanOffset(double& offsetX, double& offsetY) const {
    if (impl_) {
        impl_->GetPanOffset(offsetX, offsetY);
    } else {
        offsetX = offsetY = 0;
    }
}

void Window::SetPanOffset(double offsetX, double offsetY) {
    if (impl_) impl_->SetPanOffset(offsetX, offsetY);
}

void Window::ResetZoom() {
    if (impl_) impl_->ResetZoom();
}

void Window::ZoomToRegion(double row1, double col1, double row2, double col2) {
    if (impl_) impl_->ZoomToRegion(row1, col1, row2, col2);
}

// Coordinate conversion
bool Window::WindowToImage(int32_t windowX, int32_t windowY,
                           double& imageX, double& imageY) const {
    return impl_ ? impl_->WindowToImage(windowX, windowY, imageX, imageY) : false;
}

bool Window::ImageToWindow(double imageX, double imageY,
                           int32_t& windowX, int32_t& windowY) const {
    return impl_ ? impl_->ImageToWindow(imageX, imageY, windowX, windowY) : false;
}

// Helper to create invalid ROI result
static ROIResult MakeInvalidROI(ROIType type) {
    ROIResult result;
    result.type = type;
    result.valid = false;
    return result;
}

// ROI drawing
ROIResult Window::DrawRectangle() {
    return impl_ ? impl_->DrawROI(ROIType::Rectangle) : MakeInvalidROI(ROIType::Rectangle);
}

ROIResult Window::DrawCircle() {
    return impl_ ? impl_->DrawROI(ROIType::Circle) : MakeInvalidROI(ROIType::Circle);
}

ROIResult Window::DrawLine() {
    return impl_ ? impl_->DrawROI(ROIType::Line) : MakeInvalidROI(ROIType::Line);
}

ROIResult Window::DrawPolygon() {
    return impl_ ? impl_->DrawROI(ROIType::Polygon) : MakeInvalidROI(ROIType::Polygon);
}

ROIResult Window::DrawPoint() {
    return impl_ ? impl_->DrawROI(ROIType::Point) : MakeInvalidROI(ROIType::Point);
}

ROIResult Window::DrawROI(ROIType type) {
    return impl_ ? impl_->DrawROI(type) : MakeInvalidROI(type);
}

int32_t Window::ShowImage(const QImage& image, const std::string& title) {
    Window win(title, image.Width(), image.Height());
    win.DispImage(image);
    return win.WaitKey();
}

int32_t Window::ShowImage(const QImage& image, const std::string& title, int32_t timeoutMs) {
    Window win(title, image.Width(), image.Height());
    win.DispImage(image);
    return win.WaitKey(timeoutMs);
}

// =============================================================================
// Convenience functions
// =============================================================================

void DispImage(const QImage& image, const std::string& windowName) {
    std::lock_guard<std::mutex> lock(g_windowsMutex);

    auto it = g_windows.find(windowName);
    if (it == g_windows.end()) {
        auto win = std::make_unique<Window>(windowName, image.Width(), image.Height());
        it = g_windows.emplace(windowName, std::move(win)).first;
    }
    it->second->DispImage(image);
}

int32_t WaitKey(int32_t timeoutMs) {
    std::lock_guard<std::mutex> lock(g_windowsMutex);

    // Wait on first available window
    for (auto& [name, win] : g_windows) {
        if (win && win->IsOpen()) {
            return win->WaitKey(timeoutMs);
        }
    }
    return -1;
}

void CloseWindow(const std::string& windowName) {
    std::lock_guard<std::mutex> lock(g_windowsMutex);
    g_windows.erase(windowName);
}

void CloseAllWindows() {
    std::lock_guard<std::mutex> lock(g_windowsMutex);
    g_windows.clear();
}

} // namespace Qi::Vision::GUI
