# API Unification Plan (OpenCV-style)

Goal: expose a single, modern C++ API surface (OpenCV-like), remove Halcon-style handle/string entrances from the public SDK, and keep internal implementation details private.

## Principles
- One public way to do a thing.
- Use enums/structs instead of string parameters.
- Prefer class/namespace-based APIs over free-function handle patterns.
- Internal implementation must not be reachable from public headers.

## Module-by-Module Decisions

### Core / IO
**Keep (public):**
- `Qi::Vision::IO::ReadImage`, `Qi::Vision::IO::WriteImage` (I/O centralized)

**Deprecate/remove:**
- `QImage::FromFile`, `QImage::SaveToFile` (removed)

**Rationale:** avoid duplicated I/O entry points.

---

### Color
**Keep (public):**
- `Qi::Vision::Color::ConvertColorSpace`, `Decompose/Compose`, `AccessChannel`

**Deprecate/remove:**
- `QImage::ToGray` (removed)

**Rationale:** all color conversion in one module.

---

### Display / GUI / Draw
**Keep (public):**
- `Qi::Vision::GUI::Window` (interactive display)
- `Qi::Vision::Draw` primitives (single draw module)

**Deprecate/remove:**
- `Qi::Vision::DispImage`, `Display::Disp*` (duplicate rendering wrappers)
- `Core/Draw.h` compatibility header (removed)

**Rationale:** one interactive display class + one draw namespace.

---

### Measure (Caliper / Metrology)
**Keep (public):**
- `Qi::Vision::Measure::CaliperArray`
- `Qi::Vision::Measure::Metrology` (high-level geometric measurement)

**Deprecate/remove:**
- `GenMeasure*`, `MeasurePos`, `MeasurePairs` (removed)

**Rationale:** single engineering-oriented entry point with batch/fit results.

---

### Morphology / QRegion
**Keep (public):**
- `Qi::Vision::Morphology::*` operators

**Deprecate/remove:**
- `QRegion::Dilate/Erode/Opening/Closing` (duplicate morphology)

**Rationale:** avoid both class methods and free functions for same ops.

---

### Matching
**Keep (public):**
- `NCCModel`, `ShapeModel` classes + function APIs (but no internal impl exposure)

**Deprecate/remove:**
- `Impl()` accessors in public headers

**Rationale:** models can remain handle-style, but implementation stays private.

---

### Internal Headers
**Change:**
- Move `include/QiVision/Internal/*` out of public install/include path
- Remove any direct reference to `Qi::Vision::Internal` types from public headers

**Rationale:** prevent SDK users from depending on internal APIs.

## Parameter Modernization
- Replace string parameters (e.g., "all", "first") with enums or `struct Params`.
- Keep legacy string overloads only during migration; remove in final cleanup.

## Migration Stages
1. Document unified surface and mark deprecated APIs.
2. Update samples and docs to use unified APIs exclusively.
3. Hide/remove `Internal` headers from public includes.
4. Remove deprecated APIs after internal usage is gone.

## Unified Behavior Rules
- Invalid arguments: throw `InvalidArgumentException` with a stable, descriptive message.
- Unsupported types/parameters: throw `UnsupportedException` (never silently coerce).
- Empty inputs:
  - Empty image/region/contour: return empty outputs when a "no-op" makes sense.
  - If the operation requires data (fit, model creation), throw `InvalidArgumentException`.
- Numeric ranges:
  - Angles must be finite; scales > 0; thresholds >= 0; counts >= 1.
  - Reject NaN/Inf for all geometric inputs.
- Output clearing:
  - Functions with output vectors must clear them before filling.
- Consistent coordinate semantics:
  - `row = y`, `col = x`, and angles are in radians (CCW positive).
