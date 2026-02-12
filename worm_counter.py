#!/usr/bin/env python3
"""Automated fluorescent worm counter for WormScan RGB TIFF images.

The scanner captures two images of each plate (beginning and end of scan).
One appears in the green channel, one in the red channel. We count worms
in each channel separately and average the two for a more accurate count.
"""

import argparse
import csv
import glob
import os
import re
import sys
from datetime import datetime

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

    # Otsu threshold to find dark worms on bright background
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological cleanup: remove small noise, fill small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by size
    valid = [c for c in contours if min_size <= cv2.contourArea(c) <= max_size]

    return len(valid), valid


def render_channel_image(channel, contours, color, count, label):
    """Render a single channel as a color image with contour outlines and count text."""
    # Convert grayscale channel to BGR for colored drawing
    vis = cv2.cvtColor(channel, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis, contours, -1, color, 2)
    cv2.putText(
        vis,
        f"{label}: {count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )
    return vis


def extract_plate_number(filename):
    """Try to extract a plate number from the filename for subfolder naming."""
    # Match patterns like Plate_01, Plate01, plate-01, etc.
    m = re.search(r'[Pp]late[_\-]?(\d+)', filename)
    if m:
        return m.group(1).zfill(2)
    # Fallback: just use sequential numbering (caller handles this)
    return None


def process_image(filepath, min_size, max_size):
    """Process a single TIFF image. Returns (green_count, red_count, green_vis, red_vis) or None."""
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

    # Render each channel as a separate annotated image
    green_vis = render_channel_image(g, green_contours, (0, 255, 0), green_count, "Green")
    red_vis = render_channel_image(r, red_contours, (0, 0, 255), red_count, "Red")

    return green_count, red_count, green_vis, red_vis


def main():
    parser = argparse.ArgumentParser(description="Count fluorescent worms in WormScan TIFF images.")
    parser.add_argument("--scan-dir", default="scan", help="Directory containing .tif images (default: scan/)")
    parser.add_argument("--output-dir", default="output", help="Base output directory (default: output/)")
    parser.add_argument("--min-size", type=int, default=50, help="Minimum contour area to count as a worm (default: 50)")
    parser.add_argument("--max-size", type=int, default=5000, help="Maximum contour area to count as a worm (default: 5000)")
    args = parser.parse_args()

    scan_dir = args.scan_dir
    output_base = args.output_dir
    min_size = args.min_size
    max_size = args.max_size

    if not os.path.isdir(scan_dir):
        print(f"Error: scan directory '{scan_dir}' not found. Create it and add .tif files.")
        sys.exit(1)

    tif_files = sorted(glob.glob(os.path.join(scan_dir, "*.tif")))
    if not tif_files:
        print(f"No .tif files found in '{scan_dir}/'.")
        sys.exit(1)

    # Create timestamped run directory
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    run_dir = os.path.join(output_base, f"run-{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"Found {len(tif_files)} image(s) in '{scan_dir}/'.\n")

    results = []
    for idx, filepath in enumerate(tif_files, start=1):
        filename = os.path.basename(filepath)
        print(f"Processing {filename}...")

        result = process_image(filepath, min_size, max_size)
        if result is None:
            continue

        green_count, red_count, green_vis, red_vis = result
        avg = (green_count + red_count) / 2
        results.append((filename, green_count, red_count, avg))

        # Determine subfolder name from plate number or sequential index
        plate_num = extract_plate_number(filename)
        if plate_num is None:
            plate_num = str(idx).zfill(2)
        plate_dir = os.path.join(run_dir, plate_num)
        os.makedirs(plate_dir, exist_ok=True)

        # Save compressed JPGs
        cv2.imwrite(os.path.join(plate_dir, "green.jpg"), green_vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        cv2.imwrite(os.path.join(plate_dir, "red.jpg"), red_vis, [cv2.IMWRITE_JPEG_QUALITY, 85])

        print(f"  Green: {green_count}  Red: {red_count}  Avg: {avg:.1f}")

    if not results:
        print("No images were successfully processed.")
        sys.exit(1)

    # Write results.csv inside run directory
    total_green = sum(r[1] for r in results)
    total_red = sum(r[2] for r in results)
    total_avg = sum(r[3] for r in results)

    results_path = os.path.join(run_dir, "results.csv")
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Green", "Red", "Avg"])
        for filename, gc, rc, avg in results:
            writer.writerow([filename, gc, rc, f"{avg:.1f}"])
        writer.writerow(["TOTALS", total_green, total_red, f"{total_avg:.1f}"])

    print(f"\nResults written to {results_path}")
    print(f"Channel images saved to {run_dir}/")


if __name__ == "__main__":
    main()
