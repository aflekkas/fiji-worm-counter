#!/usr/bin/env python3
"""Automated fluorescent worm counter for WormScan RGB TIFF images.

The scanner captures two images of each plate (beginning and end of scan).
One appears in the green channel, one in the red channel. We count worms
in each channel separately and average the two for a more accurate count.
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np


def find_plate_mask(image):
    """Detect circular plate region via Hough circles and return a mask."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    h, w = gray.shape
    min_radius = min(h, w) // 4
    max_radius = min(h, w) // 2

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(h, w) // 2,
        param1=50,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    mask = np.zeros(gray.shape, dtype=np.uint8)
    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        # Use the largest circle found
        best = max(circles, key=lambda c: c[2])
        cv2.circle(mask, (best[0], best[1]), best[2], 255, -1)
    else:
        # Fallback: use the whole image
        mask[:] = 255

    return mask


def count_worms_in_channel(channel, plate_mask, min_size, max_size):
    """Count worms in a single channel image. Returns (count, contours)."""
    # Apply plate mask
    masked = cv2.bitwise_and(channel, channel, mask=plate_mask)

    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(masked, (5, 5), 0)

    # Otsu threshold to find bright worms
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleanup: remove small noise, fill small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by size
    valid = [c for c in contours if min_size <= cv2.contourArea(c) <= max_size]

    return len(valid), valid


def process_image(filepath, min_size, max_size):
    """Process a single TIFF image. Returns (green_count, red_count, annotated_image)."""
    image = cv2.imread(filepath, cv2.IMREAD_COLOR)
    if image is None:
        print(f"  WARNING: Could not read {filepath}, skipping.")
        return None

    # Split channels (OpenCV uses BGR order)
    b, g, r = cv2.split(image)

    # Detect plate region
    plate_mask = find_plate_mask(image)

    # Count green worms
    green_count, green_contours = count_worms_in_channel(g, plate_mask, min_size, max_size)

    # Count red worms
    red_count, red_contours = count_worms_in_channel(r, plate_mask, min_size, max_size)

    # Build annotated debug image
    annotated = image.copy()
    cv2.drawContours(annotated, green_contours, -1, (0, 255, 0), 2)   # green outlines
    cv2.drawContours(annotated, red_contours, -1, (0, 0, 255), 2)     # red outlines

    # Add count text
    avg = (green_count + red_count) / 2
    cv2.putText(
        annotated,
        f"Green: {green_count}  Red: {red_count}  Avg: {avg:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )

    return green_count, red_count, annotated


def main():
    parser = argparse.ArgumentParser(description="Count fluorescent worms in WormScan TIFF images.")
    parser.add_argument("--scan-dir", default="scan", help="Directory containing .tif images (default: scan/)")
    parser.add_argument("--output-dir", default="output", help="Directory for annotated debug images (default: output/)")
    parser.add_argument("--min-size", type=int, default=50, help="Minimum contour area to count as a worm (default: 50)")
    parser.add_argument("--max-size", type=int, default=5000, help="Maximum contour area to count as a worm (default: 5000)")
    args = parser.parse_args()

    scan_dir = args.scan_dir
    output_dir = args.output_dir
    min_size = args.min_size
    max_size = args.max_size

    if not os.path.isdir(scan_dir):
        print(f"Error: scan directory '{scan_dir}' not found. Create it and add .tif files.")
        sys.exit(1)

    tif_files = sorted(glob.glob(os.path.join(scan_dir, "*.tif")))
    if not tif_files:
        print(f"No .tif files found in '{scan_dir}/'.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Found {len(tif_files)} image(s) in '{scan_dir}/'.\n")

    results = []
    for filepath in tif_files:
        filename = os.path.basename(filepath)
        print(f"Processing {filename}...")

        result = process_image(filepath, min_size, max_size)
        if result is None:
            continue

        green_count, red_count, annotated = result
        avg = (green_count + red_count) / 2
        results.append((filename, green_count, red_count, avg))

        # Save annotated debug image
        debug_path = os.path.join(output_dir, f"debug_{filename}")
        cv2.imwrite(debug_path, annotated)
        print(f"  Green: {green_count}  Red: {red_count}  Avg: {avg:.1f}")

    if not results:
        print("No images were successfully processed.")
        sys.exit(1)

    # Write output.txt
    total_green = sum(r[1] for r in results)
    total_red = sum(r[2] for r in results)
    total_avg = sum(r[3] for r in results)

    # Calculate column widths for alignment
    name_width = max(len(r[0]) for r in results)
    name_width = max(name_width, len("TOTALS"))

    lines = []
    lines.append("Worm Count Results")
    lines.append("=" * 70)
    for filename, gc, rc, avg in results:
        lines.append(f"{filename:<{name_width}}    Green: {gc:<5}  Red: {rc:<5}  Avg: {avg:<5.1f}")
    lines.append("=" * 70)
    lines.append(f"{'TOTALS':<{name_width}}    Green: {total_green:<5}  Red: {total_red:<5}  Avg: {total_avg:<5.1f}")

    output_text = "\n".join(lines) + "\n"

    with open("output.txt", "w") as f:
        f.write(output_text)

    print(f"\n{output_text}")
    print(f"Results written to output.txt")
    print(f"Debug images saved to {output_dir}/")


if __name__ == "__main__":
    main()
