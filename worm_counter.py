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


def contrast_stretch(channel, low=None, high=None, mask=None):
    """Clip channel to [low, high] and remap to [0, 255].

    If low/high are not given, auto-detect from the non-zero pixels
    (or masked pixels if a mask is provided).
    """
    if low is None or high is None:
        if mask is not None:
            pixels = channel[mask > 0]
        else:
            pixels = channel[channel > 0]
        if len(pixels) == 0:
            return np.zeros_like(channel)
        if low is None:
            low = float(np.percentile(pixels, 1))
        if high is None:
            high = float(np.percentile(pixels, 99))
    if high <= low:
        return np.zeros_like(channel)
    clipped = np.clip(channel.astype(np.float32), low, high)
    stretched = ((clipped - low) / (high - low) * 255).astype(np.uint8)
    return stretched


def count_worms_in_channel(channel, other_channel, plate_mask, min_size, max_size):
    """Count worms via channel difference with Otsu threshold + shape filtering. Returns (count, contours)."""
    # Average single-worm area for splitting clumps
    AVG_WORM_AREA = 800

    # Raw channel difference: pixels where this channel is brighter than the other
    diff = cv2.subtract(channel, other_channel)

    # Apply plate mask
    masked = cv2.bitwise_and(diff, diff, mask=plate_mask)

    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(masked, (5, 5), 0)

    # Otsu threshold on the difference image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleanup: remove small noise, fill small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by size only
    valid = []
    count = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_size:
            continue

        # Large clumps: estimate worm count by dividing by average area
        if area > max_size:
            count += round(area / AVG_WORM_AREA)
            valid.append(c)
            continue

        valid.append(c)
        count += 1

    return count, valid


def render_combined_image(r, g, green_contours, red_contours,
                          green_count, red_count, plate_mask=None):
    """Render single contrast-stretched RGB image with both green and red contour outlines."""
    # Build contrast-stretched RGB so worms appear as colored shapes on yellow background
    r_s = contrast_stretch(r, mask=plate_mask)
    g_s = contrast_stretch(g, mask=plate_mask)
    b_s = np.zeros_like(r_s)  # no blue channel info
    vis = cv2.merge([b_s, g_s, r_s])  # OpenCV BGR order

    # Draw green contours and labels
    cv2.drawContours(vis, green_contours, -1, (0, 255, 0), 2)
    for i, c in enumerate(green_contours, start=1):
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.putText(vis, f"g{i}", (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Draw red contours and labels
    cv2.drawContours(vis, red_contours, -1, (0, 0, 255), 2)
    for i, c in enumerate(red_contours, start=1):
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.putText(vis, f"r{i}", (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Summary text
    avg = (green_count + red_count) / 2
    cv2.putText(vis, f"Green: {green_count}  Red: {red_count}  Avg: {avg:.0f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
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
    """Process a single TIFF image. Returns (green_count, red_count, combined_vis) or None."""
    image = cv2.imread(filepath, cv2.IMREAD_COLOR)
    if image is None:
        print(f"  WARNING: Could not read {filepath}, skipping.")
        return None

    # Split channels (OpenCV uses BGR order)
    b, g, r = cv2.split(image)

    # Detect plate region
    plate_mask = find_plate_mask(image)

    # Count green worms (G > R means green worm)
    green_count, green_contours = count_worms_in_channel(g, r, plate_mask, min_size, max_size)

    # Count red worms (R > G means red worm)
    red_count, red_contours = count_worms_in_channel(r, g, plate_mask, min_size, max_size)

    # Render single combined image with both contour sets
    vis = render_combined_image(r, g, green_contours, red_contours,
                                green_count, red_count, plate_mask)

    return green_count, red_count, vis


def main():
    parser = argparse.ArgumentParser(description="Count fluorescent worms in WormScan TIFF images.")
    parser.add_argument("--scan-dir", default="scan", help="Directory containing .tif images (default: scan/)")
    parser.add_argument("--output-dir", default="output", help="Base output directory (default: output/)")
    parser.add_argument("--min-size", type=int, default=200, help="Minimum contour area to count as a worm (default: 200)")
    parser.add_argument("--max-size", type=int, default=5000, help="Maximum contour area to count as a worm (default: 5000)")
    args = parser.parse_args()

    scan_dir = args.scan_dir
    output_base = args.output_dir
    min_size = args.min_size
    max_size = args.max_size

    if not os.path.isdir(scan_dir):
        print(f"Error: scan directory '{scan_dir}' not found. Create it and add .tif files.")
        sys.exit(1)

    tif_files = sorted(glob.glob(os.path.join(scan_dir, "*.tif")),
                       key=lambda f: int(re.search(r'(\d+)', os.path.basename(f)).group(1))
                       if re.search(r'(\d+)', os.path.basename(f)) else 0)
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

        green_count, red_count, vis = result
        avg = (green_count + red_count) / 2
        results.append((filename, green_count, red_count, avg))

        # Save single combined image named by plate number
        plate_num = extract_plate_number(filename)
        if plate_num is None:
            plate_num = str(idx).zfill(2)
        cv2.imwrite(os.path.join(run_dir, f"plate_{plate_num}.jpg"), vis,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])

        print(f"  Green: {green_count}  Red: {red_count}  Avg: {avg:.1f}")

    if not results:
        print("No images were successfully processed.")
        sys.exit(1)

    # Write results.csv
    total_green = sum(r[1] for r in results)
    total_red = sum(r[2] for r in results)
    total_avg = sum(r[3] for r in results)

    csv_path = os.path.join(run_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Green", "Red", "Avg"])
        for filename, gc, rc, avg in results:
            writer.writerow([filename, gc, rc, f"{avg:.1f}"])
        writer.writerow(["TOTALS", total_green, total_red, f"{total_avg:.1f}"])

    # Write results.txt (human-readable)
    txt_path = os.path.join(run_dir, "results.txt")
    with open(txt_path, "w") as f:
        for filename, gc, rc, avg in results:
            plate_num = extract_plate_number(filename) or "??"
            f.write(f"Plate {plate_num}:  Green={gc}  Red={rc}  Avg={avg:.1f}\n")
        f.write(f"\nTOTALS:  Green={total_green}  Red={total_red}  Avg={total_avg:.1f}\n")

    # Write copy-paste.txt (just rounded averages, one per line, in plate order)
    paste_path = os.path.join(run_dir, "copy-paste.txt")
    with open(paste_path, "w") as f:
        for _, _, _, avg in results:
            f.write(f"{round(avg)}\n")

    print(f"\nResults written to {csv_path}")
    print(f"Results written to {txt_path}")
    print(f"Copy-paste written to {paste_path}")
    print(f"Channel images saved to {run_dir}/")


if __name__ == "__main__":
    main()
