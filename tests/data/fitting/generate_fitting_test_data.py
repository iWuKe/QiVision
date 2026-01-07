#!/usr/bin/env python3
"""
Fitting Test Data Generator for QiVision

Generates synthetic test data with known ground truth for testing fitting algorithms:
- Line fitting (LineFit)
- Circle fitting (CircleFit)
- Ellipse fitting (EllipseFit)
- RANSAC with outliers

Precision requirements from CLAUDE.md:
- CircleFit: center/radius < 0.02px (1 sigma)
- LineFit: angle < 0.005 deg (1 sigma)
"""

import json
import math
import os
from datetime import datetime
from typing import List, Tuple, Dict, Any
import random

# Fixed random seed for reproducibility
RANDOM_SEED = 42


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)


def generate_gaussian_noise(sigma: float) -> float:
    """Generate Gaussian noise using Box-Muller transform."""
    if sigma <= 0:
        return 0.0
    u1 = random.random()
    u2 = random.random()
    # Avoid log(0)
    while u1 == 0:
        u1 = random.random()
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return z * sigma


def normalize_line_coefficients(a: float, b: float, c: float) -> Tuple[float, float, float]:
    """
    Normalize line equation ax + by + c = 0 such that a^2 + b^2 = 1.
    Also ensures consistent orientation (a >= 0, or a == 0 and b > 0).
    """
    norm = math.sqrt(a * a + b * b)
    if norm < 1e-12:
        raise ValueError("Invalid line: a and b cannot both be zero")
    a_norm = a / norm
    b_norm = b / norm
    c_norm = c / norm

    # Ensure consistent orientation
    if a_norm < 0 or (abs(a_norm) < 1e-12 and b_norm < 0):
        a_norm = -a_norm
        b_norm = -b_norm
        c_norm = -c_norm

    return a_norm, b_norm, c_norm


def angle_to_line_coefficients(angle_deg: float, offset: float) -> Tuple[float, float, float]:
    """
    Convert angle and offset to line equation ax + by + c = 0.

    The line passes through origin rotated by angle_deg, then shifted by offset
    perpendicular to the line direction.

    angle_deg: angle of line direction (counter-clockwise from x-axis)
    offset: perpendicular distance from origin (positive = left side)
    """
    angle_rad = math.radians(angle_deg)

    # Line direction vector: (cos(angle), sin(angle))
    # Normal vector: (-sin(angle), cos(angle)) - pointing left
    # Line equation: -sin(angle) * x + cos(angle) * y - offset = 0

    a = -math.sin(angle_rad)
    b = math.cos(angle_rad)
    c = -offset

    return normalize_line_coefficients(a, b, c)


def generate_line_points(
    angle_deg: float,
    offset: float,
    num_points: int,
    length: float,
    noise_sigma: float,
    seed_offset: int = 0
) -> Tuple[List[List[float]], Tuple[float, float, float]]:
    """
    Generate points along a line with perpendicular Gaussian noise.

    Returns: (points, (a, b, c)) where ax + by + c = 0, a^2 + b^2 = 1
    """
    set_seed(RANDOM_SEED + seed_offset)

    angle_rad = math.radians(angle_deg)

    # Direction and normal vectors
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)
    nx = -dy  # Normal (perpendicular) vector
    ny = dx

    # Line center point (offset from origin along normal)
    cx = offset * nx
    cy = offset * ny

    points = []
    half_len = length / 2.0

    for i in range(num_points):
        # Parameter t along the line
        if num_points == 1:
            t = 0
        else:
            t = -half_len + (length * i / (num_points - 1))

        # Point on perfect line
        x = cx + t * dx
        y = cy + t * dy

        # Add perpendicular noise
        noise = generate_gaussian_noise(noise_sigma)
        x += noise * nx
        y += noise * ny

        points.append([x, y])

    # Ground truth coefficients
    a, b, c = angle_to_line_coefficients(angle_deg, offset)

    return points, (a, b, c)


def generate_circle_points(
    cx: float,
    cy: float,
    radius: float,
    arc_angle_deg: float,
    start_angle_deg: float,
    num_points: int,
    noise_sigma: float,
    seed_offset: int = 0
) -> List[List[float]]:
    """
    Generate points on a circle/arc with radial Gaussian noise.

    cx, cy: center coordinates
    radius: circle radius
    arc_angle_deg: angular span of the arc (360 for full circle)
    start_angle_deg: starting angle
    num_points: number of points to generate
    noise_sigma: standard deviation of radial noise
    """
    set_seed(RANDOM_SEED + seed_offset)

    points = []
    arc_rad = math.radians(arc_angle_deg)
    start_rad = math.radians(start_angle_deg)

    for i in range(num_points):
        if num_points == 1:
            theta = start_rad
        else:
            theta = start_rad + (arc_rad * i / (num_points - 1))

        # Perfect point on circle
        x = cx + radius * math.cos(theta)
        y = cy + radius * math.sin(theta)

        # Add radial noise
        noise = generate_gaussian_noise(noise_sigma)
        x += noise * math.cos(theta)
        y += noise * math.sin(theta)

        points.append([x, y])

    return points


def generate_ellipse_points(
    cx: float,
    cy: float,
    semi_major: float,
    semi_minor: float,
    rotation_deg: float,
    arc_angle_deg: float,
    start_angle_deg: float,
    num_points: int,
    noise_sigma: float,
    seed_offset: int = 0
) -> List[List[float]]:
    """
    Generate points on an ellipse/arc with normal Gaussian noise.

    cx, cy: center coordinates
    semi_major: semi-major axis length (a)
    semi_minor: semi-minor axis length (b)
    rotation_deg: rotation angle of ellipse (counter-clockwise)
    arc_angle_deg: angular span of the arc (360 for full ellipse)
    start_angle_deg: starting angle (parameter, not geometric)
    num_points: number of points to generate
    noise_sigma: standard deviation of noise perpendicular to ellipse
    """
    set_seed(RANDOM_SEED + seed_offset)

    points = []
    arc_rad = math.radians(arc_angle_deg)
    start_rad = math.radians(start_angle_deg)
    rot_rad = math.radians(rotation_deg)

    cos_rot = math.cos(rot_rad)
    sin_rot = math.sin(rot_rad)

    for i in range(num_points):
        if num_points == 1:
            t = start_rad
        else:
            t = start_rad + (arc_rad * i / (num_points - 1))

        # Point on unrotated ellipse (parametric form)
        x_local = semi_major * math.cos(t)
        y_local = semi_minor * math.sin(t)

        # Compute normal direction at this point (for unrotated ellipse)
        # Gradient of ellipse: (x/a^2, y/b^2)
        nx_local = x_local / (semi_major * semi_major)
        ny_local = y_local / (semi_minor * semi_minor)
        n_norm = math.sqrt(nx_local * nx_local + ny_local * ny_local)
        if n_norm > 1e-12:
            nx_local /= n_norm
            ny_local /= n_norm

        # Add noise along normal direction
        noise = generate_gaussian_noise(noise_sigma)
        x_local += noise * nx_local
        y_local += noise * ny_local

        # Rotate and translate to final position
        x = cx + x_local * cos_rot - y_local * sin_rot
        y = cy + x_local * sin_rot + y_local * cos_rot

        points.append([x, y])

    return points


def add_outliers(
    points: List[List[float]],
    outlier_ratio: float,
    outlier_range: float,
    seed_offset: int = 0
) -> Tuple[List[List[float]], List[int]]:
    """
    Replace some points with random outliers.

    Returns: (modified_points, outlier_indices)
    """
    set_seed(RANDOM_SEED + seed_offset + 10000)

    num_points = len(points)
    num_outliers = int(num_points * outlier_ratio)

    # Compute bounding box center for outlier generation
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # Select random indices for outliers
    outlier_indices = random.sample(range(num_points), num_outliers)
    outlier_indices.sort()

    # Create modified points
    modified_points = [p.copy() for p in points]

    for idx in outlier_indices:
        # Generate outlier far from the geometric shape
        angle = random.random() * 2 * math.pi
        dist = outlier_range * (0.5 + random.random() * 0.5)
        modified_points[idx][0] = center_x + dist * math.cos(angle)
        modified_points[idx][1] = center_y + dist * math.sin(angle)

    return modified_points, outlier_indices


def create_test_case(
    test_type: str,
    ground_truth: Dict[str, Any],
    points: List[List[float]],
    noise_sigma: float,
    description: str,
    outlier_indices: List[int] = None
) -> Dict[str, Any]:
    """Create a test case dictionary."""
    case = {
        "type": test_type,
        "generator": "FittingTestDataGenerator",
        "version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "ground_truth": ground_truth,
        "noise_sigma": noise_sigma,
        "num_points": len(points),
        "points": points,
        "description": description
    }

    if outlier_indices is not None and len(outlier_indices) > 0:
        case["outlier_indices"] = outlier_indices
        case["outlier_ratio"] = len(outlier_indices) / len(points)

    return case


def save_test_case(case: Dict[str, Any], filepath: str) -> None:
    """Save test case to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(case, f, indent=2)
    print(f"  Generated: {filepath}")


def generate_line_tests(output_dir: str) -> int:
    """Generate all line fitting test cases."""
    print("\nGenerating Line Tests...")
    count = 0

    # Test parameters
    angles = [0, 30, 45, 60, 90, 135, 180]
    point_counts = {"short": 10, "medium": 50, "long": 200}
    noise_levels = [0, 1, 2, 5]
    offset = 50.0  # Distance from origin

    for angle in angles:
        for length_name, num_points in point_counts.items():
            for noise in noise_levels:
                # Compute line length based on point count
                length = num_points * 2.0

                seed_offset = count
                points, (a, b, c) = generate_line_points(
                    angle, offset, num_points, length, noise, seed_offset
                )

                # Ground truth
                gt = {
                    "a": a,
                    "b": b,
                    "c": c,
                    "angle_deg": angle,
                    "offset": offset
                }

                desc = f"Line at {angle}deg, {length_name} ({num_points} pts), noise sigma={noise}px"

                case = create_test_case("line", gt, points, noise, desc)

                filename = f"line_angle{angle:03d}_{length_name}_noise{noise}.json"
                save_test_case(case, os.path.join(output_dir, filename))
                count += 1

    return count


def generate_circle_tests(output_dir: str) -> int:
    """Generate all circle fitting test cases."""
    print("\nGenerating Circle Tests...")
    count = 0

    # Test parameters
    radii = {"small": 10, "medium": 50, "large": 200}
    arc_angles = {"full": 360, "half": 180, "quarter": 90}
    point_counts = {"sparse": 8, "medium": 30, "dense": 100}
    noise_levels = [0, 1, 2, 5]

    # Subpixel center offset for precision testing
    cx_base, cy_base = 100.0, 100.0
    subpixel_offsets = [0.0, 0.37, 0.73]  # Test subpixel accuracy

    case_idx = 0
    for rad_name, radius in radii.items():
        for arc_name, arc_angle in arc_angles.items():
            for pts_name, num_points in point_counts.items():
                # Skip invalid combinations (too few points for arc)
                if arc_name == "quarter" and pts_name == "sparse":
                    min_points = 4  # Need at least some points for 90 degree arc
                    if num_points < min_points:
                        continue

                for noise in noise_levels:
                    for offset in subpixel_offsets:
                        cx = cx_base + offset
                        cy = cy_base + offset * 0.5  # Different offset for y

                        seed_offset = case_idx
                        points = generate_circle_points(
                            cx, cy, radius, arc_angle, 0, num_points, noise, seed_offset
                        )

                        gt = {
                            "cx": cx,
                            "cy": cy,
                            "radius": radius,
                            "arc_angle_deg": arc_angle,
                            "start_angle_deg": 0
                        }

                        offset_str = f"{int(offset*100):02d}" if offset > 0 else "00"
                        desc = (f"Circle r={radius}, {arc_name} arc, "
                               f"{pts_name} ({num_points} pts), noise sigma={noise}px, "
                               f"subpixel offset={offset}")

                        case = create_test_case("circle", gt, points, noise, desc)

                        filename = (f"circle_r{radius}_{arc_name}_{pts_name}_"
                                   f"noise{noise}_offset{offset_str}.json")
                        save_test_case(case, os.path.join(output_dir, filename))
                        case_idx += 1

    return case_idx


def generate_ellipse_tests(output_dir: str) -> int:
    """Generate all ellipse fitting test cases."""
    print("\nGenerating Ellipse Tests...")
    count = 0

    # Test parameters
    axis_ratios = [(30, 20), (40, 20), (60, 20)]  # (a, b) semi-major, semi-minor
    rotations = [0, 30, 45, 90]
    point_counts = {"minimal": 6, "medium": 30, "dense": 100}
    noise_levels = [0, 1, 2, 5]

    cx, cy = 100.0, 100.0

    case_idx = 0
    for (a, b) in axis_ratios:
        ratio_str = f"{a}_{b}"
        for rot in rotations:
            for pts_name, num_points in point_counts.items():
                for noise in noise_levels:
                    seed_offset = case_idx
                    points = generate_ellipse_points(
                        cx, cy, a, b, rot, 360, 0, num_points, noise, seed_offset
                    )

                    gt = {
                        "cx": cx,
                        "cy": cy,
                        "semi_major": a,
                        "semi_minor": b,
                        "rotation_deg": rot,
                        "eccentricity": math.sqrt(1 - (b/a)**2)
                    }

                    desc = (f"Ellipse a={a} b={b} (ratio {a/b:.1f}:1), "
                           f"rot={rot}deg, {pts_name} ({num_points} pts), "
                           f"noise sigma={noise}px")

                    case = create_test_case("ellipse", gt, points, noise, desc)

                    filename = f"ellipse_{ratio_str}_rot{rot:03d}_{pts_name}_noise{noise}.json"
                    save_test_case(case, os.path.join(output_dir, filename))
                    case_idx += 1

    return case_idx


def generate_outlier_tests(output_dir: str) -> int:
    """Generate RANSAC test cases with outliers."""
    print("\nGenerating Outlier/RANSAC Tests...")
    count = 0

    outlier_ratios = [0.1, 0.2, 0.3, 0.5]

    # Line with outliers
    for ratio in outlier_ratios:
        seed_offset = count
        points, (a, b, c) = generate_line_points(
            45, 50, 50, 100, 1.0, seed_offset
        )
        points_with_outliers, outlier_indices = add_outliers(
            points, ratio, 100.0, seed_offset
        )

        gt = {
            "type": "line",
            "a": a,
            "b": b,
            "c": c,
            "angle_deg": 45,
            "offset": 50
        }

        desc = f"Line with {int(ratio*100)}% outliers for RANSAC testing"

        case = create_test_case(
            "line_outliers", gt, points_with_outliers, 1.0, desc, outlier_indices
        )

        filename = f"outlier_line_{int(ratio*100):02d}pct.json"
        save_test_case(case, os.path.join(output_dir, filename))
        count += 1

    # Circle with outliers
    for ratio in outlier_ratios:
        seed_offset = count + 100
        cx, cy, r = 100.0, 100.0, 50.0
        points = generate_circle_points(cx, cy, r, 360, 0, 50, 1.0, seed_offset)
        points_with_outliers, outlier_indices = add_outliers(
            points, ratio, 100.0, seed_offset
        )

        gt = {
            "type": "circle",
            "cx": cx,
            "cy": cy,
            "radius": r
        }

        desc = f"Circle with {int(ratio*100)}% outliers for RANSAC testing"

        case = create_test_case(
            "circle_outliers", gt, points_with_outliers, 1.0, desc, outlier_indices
        )

        filename = f"outlier_circle_{int(ratio*100):02d}pct.json"
        save_test_case(case, os.path.join(output_dir, filename))
        count += 1

    # Ellipse with outliers
    for ratio in outlier_ratios:
        seed_offset = count + 200
        cx, cy, a, b = 100.0, 100.0, 50.0, 30.0
        points = generate_ellipse_points(
            cx, cy, a, b, 30, 360, 0, 50, 1.0, seed_offset
        )
        points_with_outliers, outlier_indices = add_outliers(
            points, ratio, 80.0, seed_offset
        )

        gt = {
            "type": "ellipse",
            "cx": cx,
            "cy": cy,
            "semi_major": a,
            "semi_minor": b,
            "rotation_deg": 30
        }

        desc = f"Ellipse with {int(ratio*100)}% outliers for RANSAC testing"

        case = create_test_case(
            "ellipse_outliers", gt, points_with_outliers, 1.0, desc, outlier_indices
        )

        filename = f"outlier_ellipse_{int(ratio*100):02d}pct.json"
        save_test_case(case, os.path.join(output_dir, filename))
        count += 1

    return count


def generate_precision_tests(output_dir: str) -> int:
    """Generate specific test cases for precision verification."""
    print("\nGenerating Precision Tests...")
    count = 0

    # These tests are specifically designed to verify the precision requirements
    # from CLAUDE.md: CircleFit < 0.02px, LineFit < 0.005deg

    # High-precision line tests (many points, low noise)
    for angle in [0, 30, 45, 60, 90]:
        seed_offset = count
        points, (a, b, c) = generate_line_points(
            angle, 50.0, 500, 1000.0, 0.5, seed_offset
        )

        gt = {
            "a": a,
            "b": b,
            "c": c,
            "angle_deg": angle,
            "precision_requirement": "angle_error < 0.005 deg (1 sigma)"
        }

        desc = f"Precision test: Line {angle}deg, 500 pts, noise 0.5px"

        case = create_test_case("line", gt, points, 0.5, desc)
        filename = f"precision_line_angle{angle:03d}.json"
        save_test_case(case, os.path.join(output_dir, filename))
        count += 1

    # High-precision circle tests
    for r in [20, 50, 100]:
        for offset in [0.0, 0.25, 0.5, 0.75]:
            seed_offset = count + 100
            cx = 100.0 + offset
            cy = 100.0 + offset * 0.7
            points = generate_circle_points(
                cx, cy, r, 360, 0, 200, 0.5, seed_offset
            )

            gt = {
                "cx": cx,
                "cy": cy,
                "radius": r,
                "precision_requirement": "center/radius_error < 0.02px (1 sigma)"
            }

            offset_str = f"{int(offset*100):02d}"
            desc = f"Precision test: Circle r={r}, center offset {offset}, 200 pts, noise 0.5px"

            case = create_test_case("circle", gt, points, 0.5, desc)
            filename = f"precision_circle_r{r}_offset{offset_str}.json"
            save_test_case(case, os.path.join(output_dir, filename))
            count += 1

    return count


def main():
    """Main entry point."""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Generate all test data
    total = 0

    total += generate_line_tests(os.path.join(base_dir, "line"))
    total += generate_circle_tests(os.path.join(base_dir, "circle"))
    total += generate_ellipse_tests(os.path.join(base_dir, "ellipse"))
    total += generate_outlier_tests(os.path.join(base_dir, "outlier"))
    total += generate_precision_tests(os.path.join(base_dir, "line"))  # Put precision tests in same dirs

    # Add circle precision tests to circle dir
    print("\nAdding circle precision tests...")

    print(f"\n{'='*60}")
    print(f"Total test cases generated: {total}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    exit(main())
