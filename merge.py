#!/usr/bin/env python3
from __future__ import annotations

"""
3D Bounding Box Generator

This module converts 2D bounding box detections from multiple views into robust
3D bounding boxes through advanced geometric intersection and temporal tracking
algorithms. It processes dual-view detection results and generates optimized
3D object localization for semiconductor inspection.

Key Features:
- Multi-view 2D to 3D bounding box conversion
- Temporal tracking with IoU-based merging
- Adaptive volume-based filtering using K-means clustering
- Cross-view geometric intersection computation
- Robust handling of detection gaps and noise
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from sklearn.cluster import KMeans

from utils import log


# Function to compute the intersection of two 3D bounding boxes
def intersect_bbox(bbox1, bbox2):
    """
    Computes the intersection of two 3D bounding boxes.

    Parameters:
    - bbox1: List representing the first bounding box in the format [xmin, ymin, zmin, xmax, ymax, zmax].
    - bbox2: List representing the second bounding box in the format [xmin, ymin, zmin, xmax, ymax, zmax].

    Returns:
    - A list representing the intersected bounding box [xmin, ymin, zmin, xmax, ymax, zmax]
      if the boxes overlap, or None if there is no overlap.
    """

    xmin1, xmax1, ymin1, ymax1, zmin1, zmax1 = bbox1
    xmin2, xmax2, ymin2, ymax2, zmin2, zmax2 = bbox2

    # Calculate the intersection bounds
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    zmin = max(zmin1, zmin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)
    zmax = min(zmax1, zmax2)

    # Check if there is an intersection (i.e., the bounding boxes overlap)
    if xmin <= xmax and ymin <= ymax and zmin <= zmax:
        return [xmin, xmax, ymin, ymax, zmin, zmax]
    else:
        return None


# Function to compute the intersection of all bounding boxes in two lists, considering orientation differences
def compute_bbox_intersections(S1_bboxes, S2_bboxes, sizeY):
    """
    Computes the intersections of bounding boxes from two different orientations (S1 and S2).

    The bounding boxes from S1 correspond to slices from data1[:,:,i] (x and y from the first and second dimensions),
    and the bounding boxes from S2 correspond to slices from data1[:,i,:] (x and y from the first and third dimensions).

    Parameters:
    - S1_bboxes: A list of bounding boxes for slices S1 in the format [xmin, ymin, zmin, xmax, ymax, zmax].
    - S2_bboxes: A list of bounding boxes for slices S2 in the format [xmin, zmin, ymin, xmax, zmax, ymax].

    Returns:
    - A list of intersected bounding boxes in the format [xmin, ymin, zmin, xmax, ymax, zmax].
    """

    intersections = []

    # Iterate through bounding boxes in S1
    for bbox1 in S1_bboxes:
        # Iterate through bounding boxes in S2
        for bbox2 in S2_bboxes:
            # Swap the ymin-ymax with zmin-zmax for bbox2 to align with S1's orientation

            bbox2_aligned = [
                bbox2[0],  # xmin remains the same
                bbox2[1],  # xmax remains the same
                bbox2[4],  # swap ymin with zmin
                bbox2[5],  # swap ymax with zmax
                bbox2[2],
                bbox2[3],
            ]

            # Compute the intersection of bbox1 and the aligned bbox2
            intersected_bbox = intersect_bbox(bbox1, bbox2_aligned)

            # If there's an intersection, add it to the results
            if intersected_bbox:
                intersections.append(intersected_bbox)
    return intersections


def load_bbox_data(file_path: str, image_dimensions: tuple[int, int], is_normalized: bool) -> np.ndarray:
    """Load bounding box data from a file. Adjusts for image dimensions if data is normalized.
    The bounding box data is expected to be in the format [xmin, ymin, xmax, ymax].


    @param file_path: The path to the text file containing bounding box data.
    @param image_dimensions: Tuple containing the width and height of the image.
    @param is_normalized: Boolean indicating if bbox data is normalized.

    @return: A numpy array of bounding boxes, with each bounding box represented by [xmin, ymin, xmax, ymax].
        Coordinates are rounded to integers.
    """
    width, height = image_dimensions

    n_id = 0
    if "view2" in file_path:
        n_id = 1

    try:
        data = np.loadtxt(file_path, delimiter=" ")
        if data.size == 0:  # Check if there is no data to process
            return np.array([])  # Return an empty array if no data
    except OSError as e:
        log(f"Error reading file {file_path}: {e}", level="error")
        return np.array([])
    except ValueError as e:
        log(f"Malformed data in file {file_path}: {e}", level="warning")
        return np.array([])

    if data.ndim == 1:  # Single bbox case, reshape to ensure it's a 2D array
        data = np.reshape(data, (-1, 5))

    data_bb = data[data[:, 0] == n_id]
    data_bb = data_bb[:, 1:]

    widths = data_bb[:, 2] - data_bb[:, 0]
    data = data_bb[widths < 150]

    # Adjust order from [xmin, ymin, xmax, ymax] to [xmin, xmax, ymin, ymax]
    data = data[:, [0, 2, 1, 3]]

    if is_normalized:
        # Scale normalized coordinates up to the actual image dimensions
        # width, height = image_dimensions
        data[:, [0, 2]] *= width  # Scale xmin and xmax by image width
        data[:, [1, 3]] *= height  # Scale ymin and ymax by image height

    return np.round(data).astype(int)  # Round coordinates and convert to integers


def calculate_intersection_over_union(bbox_a, bbox_b):
    """Calculate the Intersection over Union (IoU) of two bounding boxes.

    @param bbox_a: First bounding box [xmin, xmax, ymin, ymax].
    @param bbox_b: Second bounding box [xmin, xmax, ymin, ymax].

    @return: The IoU score.
    """
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(bbox_a[0], bbox_b[0])
    xB = min(bbox_a[1], bbox_b[1])
    yA = max(bbox_a[2], bbox_b[2])
    yB = min(bbox_a[3], bbox_b[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0  # No overlap

    # Compute the area of both the prediction and ground-truth rectangles
    bbox_a_area = (bbox_a[1] - bbox_a[0]) * (bbox_a[3] - bbox_a[2])
    bbox_b_area = (bbox_b[1] - bbox_b[0]) * (bbox_b[3] - bbox_b[2])

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(bbox_a_area + bbox_b_area - interArea)

    return iou


def batch_iou(bbox, existing_bboxes):
    """Calculate IoU between one bbox and an array of existing bboxes - vectorized.

    @param bbox: Single bounding box [xmin, xmax, ymin, ymax].
    @param existing_bboxes: Array of bboxes [N, 6] where first 4 cols are [xmin, xmax, ymin, ymax].

    @return: Array of IoU scores [N].
    """
    if len(existing_bboxes) == 0:
        return np.array([])

    # Extract coordinates
    xA = np.maximum(bbox[0], existing_bboxes[:, 0])
    xB = np.minimum(bbox[1], existing_bboxes[:, 1])
    yA = np.maximum(bbox[2], existing_bboxes[:, 2])
    yB = np.minimum(bbox[3], existing_bboxes[:, 3])

    # Intersection area
    inter_area = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)

    # Areas
    bbox_area = (bbox[1] - bbox[0]) * (bbox[3] - bbox[2])
    existing_areas = (existing_bboxes[:, 1] - existing_bboxes[:, 0]) * (existing_bboxes[:, 3] - existing_bboxes[:, 2])

    # IoU
    union_area = bbox_area + existing_areas - inter_area
    iou = np.where(union_area > 0, inter_area / union_area, 0)

    return iou


def compute_volumes(bboxes):
    """
    Compute the volume of each 3D bounding box.
    For bboxes in format [xmin, xmax, ymin, ymax, zmin, zmax]
    """
    # Calculate dimensions from min/max coordinates
    widths = bboxes[:, 1] - bboxes[:, 0]  # xmax - xmin
    heights = bboxes[:, 3] - bboxes[:, 2]  # ymax - ymin
    depths = bboxes[:, 5] - bboxes[:, 4]  # zmax - zmin

    return widths * heights * depths


def adaptive_threshold(volumes, n_bins=50, plot=False):
    """
    Compute an adaptive threshold based on histogram analysis.
    Uses K-means clustering with 2 clusters to separate small and large volumes.
    """
    # Reshape volumes for K-means
    volumes_reshaped = volumes.reshape(-1, 1)

    # Apply K-means clustering with 2 clusters
    kmeans = KMeans(n_clusters=2, random_state=0).fit(volumes_reshaped)

    # Get cluster centers
    centers = kmeans.cluster_centers_.flatten()

    # Threshold is the midpoint between the cluster centers
    threshold = (centers[0] + centers[1]) / 2

    if plot:
        # Plot histogram
        plt.figure(figsize=(10, 6))
        hist, bin_edges = np.histogram(volumes, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.bar(bin_centers, hist, width=(bin_edges[1] - bin_edges[0]), alpha=0.7)

        # Highlight threshold
        plt.axvline(x=threshold, color="r", linestyle="--", label=f"Threshold: {threshold:.2f}")

        # Add labels and legend
        plt.xlabel("Volume")
        plt.ylabel("Frequency")
        plt.title("Histogram of Bounding Box Volumes")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("volume_histogram.png")
        plt.show()

    return threshold


def filter_bboxes(bboxes, volumes, threshold):
    """Remove bounding boxes smaller than the threshold."""
    mask = volumes >= threshold
    return bboxes[mask], volumes[mask]


def process_bbox_data_inmemory(
    data: np.ndarray,
    image_dimensions: tuple[int, int],
    is_normalized: bool,
    is_view2: bool = False,
) -> np.ndarray:
    """Process bounding box data from in-memory array.

    @param data: Detection array with shape [N, 5] containing [class_id, x1, y1, x2, y2].
    @param image_dimensions: Tuple containing the width and height of the image.
    @param is_normalized: Boolean indicating if bbox data is normalized.
    @param is_view2: Boolean indicating if this is view2 (uses class_id=1 filter).

    @return: A numpy array of bounding boxes in format [xmin, xmax, ymin, ymax].
    """
    if data.size == 0:
        return np.array([])

    width, height = image_dimensions
    n_id = 1 if is_view2 else 0

    # Ensure 2D array
    if data.ndim == 1:
        data = data.reshape(-1, 5)

    # Filter by class ID
    data_bb = data[data[:, 0] == n_id]
    if data_bb.size == 0:
        return np.array([])

    data_bb = data_bb[:, 1:]  # Remove class_id column

    # Filter by width
    widths = data_bb[:, 2] - data_bb[:, 0]
    data_filtered = data_bb[widths < 150]
    if data_filtered.size == 0:
        return np.array([])

    # Adjust order from [xmin, ymin, xmax, ymax] to [xmin, xmax, ymin, ymax]
    result = data_filtered[:, [0, 2, 1, 3]]

    if is_normalized:
        result[:, [0, 2]] *= width
        result[:, [1, 3]] *= height

    return np.round(result).astype(int)


def merge_2d_bboxes_into_3d_inmemory(
    image_range: range,
    detections: dict[str, np.ndarray],
    image_dimensions: tuple[int, int],
    is_normalized: bool,
    is_view2: bool = False,
) -> np.ndarray:
    """Merge 2D bounding boxes from in-memory detections into 3D bounding boxes.

    This function processes bounding boxes from in-memory detection results and merges
    them into 3D bounding boxes based on spatial overlap (IoU) and temporal proximity.

    @param image_range: A range object defining the indices of images to process.
    @param detections: Dictionary mapping image filename (e.g., "image0") to detection
        arrays with shape [N, 5] containing [class_id, x1, y1, x2, y2].
    @param image_dimensions: Tuple of (width, height) for the images.
    @param is_normalized: Flag indicating if bbox coordinates are normalized.
    @param is_view2: Flag indicating if this is view2 (uses class_id=1 filter).

    @return: An array of merged 3D bounding boxes,
        each represented by [xmin, xmax, ymin, ymax, first_seen, last_seen].
    """
    jump_threshold = 10
    intersection_threshold = 0.2

    max_bboxes = len(image_range) * 20
    combined_3d_bboxes = np.empty((max_bboxes, 6), dtype=np.float64)
    bbox_count = 0

    for i in image_range:
        filename = f"image{i}"

        if filename not in detections:
            continue

        # Process detection data from memory
        bboxes = process_bbox_data_inmemory(
            detections[filename],
            image_dimensions,
            is_normalized,
            is_view2,
        )

        if bboxes.size == 0:
            continue

        if bbox_count == 0:
            num_new = bboxes.shape[0]
            combined_3d_bboxes[:num_new, :4] = bboxes
            combined_3d_bboxes[:num_new, 4] = i
            combined_3d_bboxes[:num_new, 5] = i
            bbox_count = num_new
            continue

        current_bboxes = combined_3d_bboxes[:bbox_count]

        for bbox in bboxes:
            ious = batch_iou(bbox[:4], current_bboxes)
            temporal_mask = np.abs(i - current_bboxes[:, 5]) <= jump_threshold
            valid_mask = (ious > intersection_threshold) & temporal_mask

            if np.any(valid_mask):
                valid_indices = np.where(valid_mask)[0]
                best_idx = valid_indices[np.argmax(ious[valid_indices])]

                combined_3d_bboxes[best_idx, 0] = min(bbox[0], combined_3d_bboxes[best_idx, 0])
                combined_3d_bboxes[best_idx, 1] = max(bbox[1], combined_3d_bboxes[best_idx, 1])
                combined_3d_bboxes[best_idx, 2] = min(bbox[2], combined_3d_bboxes[best_idx, 2])
                combined_3d_bboxes[best_idx, 3] = max(bbox[3], combined_3d_bboxes[best_idx, 3])
                combined_3d_bboxes[best_idx, 5] = i
            else:
                if bbox_count >= max_bboxes:
                    combined_3d_bboxes = np.vstack([combined_3d_bboxes, np.empty((max_bboxes, 6))])
                    max_bboxes *= 2

                combined_3d_bboxes[bbox_count, :4] = bbox
                combined_3d_bboxes[bbox_count, 4] = i
                combined_3d_bboxes[bbox_count, 5] = i
                bbox_count += 1

    result = combined_3d_bboxes[:bbox_count].copy()
    log(f"Generated {bbox_count} 3D bounding boxes.")
    return result


def generate_bb3d_inmemory(
    view1_detections: dict[str, np.ndarray],
    view2_detections: dict[str, np.ndarray],
    view1_num_slices: int,
    view2_num_slices: int,
    image_dimensions: tuple[int, int],
    output_file: str | None = None,
    is_normalized: bool = False,
) -> np.ndarray:
    """Generate 3D bounding boxes from in-memory detection results.

    @param view1_detections: Dictionary of detections for view1.
    @param view2_detections: Dictionary of detections for view2.
    @param view1_num_slices: Number of slices in view1.
    @param view2_num_slices: Number of slices in view2.
    @param image_dimensions: Tuple of (width, height) for the images.
    @param output_file: Optional path for saving 3D bboxes as numpy file.
    @param is_normalized: Whether bbox coordinates are normalized.

    @return: Filtered 3D bounding boxes array.
    """
    image_indices = range(0, view1_num_slices)
    image_indices1 = range(0, view2_num_slices)

    combined_3d_bboxes = merge_2d_bboxes_into_3d_inmemory(
        image_indices, view1_detections, image_dimensions, is_normalized, is_view2=False
    )
    combined_3d_bboxes1 = merge_2d_bboxes_into_3d_inmemory(
        image_indices1, view2_detections, image_dimensions, is_normalized, is_view2=True
    )

    if output_file and combined_3d_bboxes is not None and combined_3d_bboxes.size > 0:
        np.save(output_file, combined_3d_bboxes)
        np.save(output_file[:-4] + "_1.npy", combined_3d_bboxes1)
        log(f"3D bounding boxes with shape {combined_3d_bboxes.shape} saved to {output_file}.", level="debug")

    if combined_3d_bboxes.size == 0:
        log("No 3D bounding boxes were generated.", level="warning")
        return np.empty((0, 6))

    final_3d_bbox = compute_bbox_intersections(combined_3d_bboxes, combined_3d_bboxes1, view1_num_slices)

    if len(final_3d_bbox) < combined_3d_bboxes.shape[0] / 2:
        final_3d_bbox = combined_3d_bboxes.tolist()

    volumes = compute_volumes(np.array(final_3d_bbox))
    threshold = adaptive_threshold(volumes)
    filtered_bboxes, _ = filter_bboxes(np.array(final_3d_bbox), volumes, threshold)

    return filtered_bboxes


def merge_2d_bboxes_into_3d(image_range: range, bbox_dir: str, image_dir: str, is_normalized: bool) -> np.ndarray:
    """Merge 2D bounding boxes from a series of images into 3D bounding boxes.

    This function processes bounding boxes from multiple 2D images and merges them into 3D bounding boxes based on
    spatial overlap (Intersection over Union, IoU) and temporal proximity (within a given jump_threshold).

    @param image_range (range): A range object defining the indices of images to process.
    @param bbox_dir (str): The directory path where bounding box files are stored.
    @param image_dir (str): The directory path where corresponding images are stored.
    @param is_normalized (bool): A flag indicating if the bounding box coordinates are normalized
        relative to image dimensions.

    @return (np.ndarray): An array of merged 3D bounding boxes,
        each represented by [xmin, xmax, ymin, ymax, first_seen, last_seen].
    """
    # Thresholds for deciding when to merge bounding boxes
    jump_threshold = 10  # Temporal jump threshold for merging bounding boxes
    intersection_threshold = 0.2  # IoU threshold for merging

    # Pre-allocate array for combined 3D bounding boxes (avoid repeated vstack)
    max_bboxes = len(image_range) * 20  # Estimate: ~20 bboxes per slice max
    combined_3d_bboxes = np.empty((max_bboxes, 6), dtype=np.float64)
    bbox_count = 0

    # Cache image dimensions - read from first image only
    cached_dims = None
    for i in image_range:
        image_path = f"{image_dir}/image{i}.jpg"
        if os.path.exists(image_path):
            image = imread(image_path)
            cached_dims = (image.shape[1], image.shape[0])  # (width, height)
            break

    if cached_dims is None:
        log("No images found in range. Returning empty array.", level="warning")
        return np.empty((0, 6))

    width, height = cached_dims

    # Process each image in the specified range
    for i in image_range:
        bbox_file_path = f"{bbox_dir}/image{i}.txt"

        if not os.path.exists(bbox_file_path):
            continue

        # Load 2D bounding boxes for the current image (using cached dimensions)
        bboxes = load_bbox_data(bbox_file_path, (width, height), is_normalized)

        # Skip processing if no bounding boxes are found
        if bboxes.size == 0:
            continue

        # If no 3D bounding boxes have been combined yet, initialize with bboxes from first valid image
        if bbox_count == 0:
            num_new = bboxes.shape[0]
            combined_3d_bboxes[:num_new, :4] = bboxes
            combined_3d_bboxes[:num_new, 4] = i  # first_seen
            combined_3d_bboxes[:num_new, 5] = i  # last_seen
            bbox_count = num_new
            continue

        # Attempt to merge current bounding boxes with existing 3D bounding boxes
        current_bboxes = combined_3d_bboxes[:bbox_count]

        for bbox in bboxes:
            # Vectorized IoU calculation against all existing bboxes
            ious = batch_iou(bbox[:4], current_bboxes)

            # Find candidates: IoU > threshold AND within temporal jump threshold
            temporal_mask = np.abs(i - current_bboxes[:, 5]) <= jump_threshold
            valid_mask = (ious > intersection_threshold) & temporal_mask

            if np.any(valid_mask):
                # Find best match (highest IoU among valid candidates)
                valid_indices = np.where(valid_mask)[0]
                best_idx = valid_indices[np.argmax(ious[valid_indices])]

                # Merge: update coordinates and last_seen
                combined_3d_bboxes[best_idx, 0] = min(bbox[0], combined_3d_bboxes[best_idx, 0])
                combined_3d_bboxes[best_idx, 1] = max(bbox[1], combined_3d_bboxes[best_idx, 1])
                combined_3d_bboxes[best_idx, 2] = min(bbox[2], combined_3d_bboxes[best_idx, 2])
                combined_3d_bboxes[best_idx, 3] = max(bbox[3], combined_3d_bboxes[best_idx, 3])
                combined_3d_bboxes[best_idx, 5] = i
            else:
                # Add new 3D bounding box
                if bbox_count >= max_bboxes:
                    # Expand array if needed (rare case)
                    combined_3d_bboxes = np.vstack([combined_3d_bboxes, np.empty((max_bboxes, 6))])
                    max_bboxes *= 2

                combined_3d_bboxes[bbox_count, :4] = bbox
                combined_3d_bboxes[bbox_count, 4] = i
                combined_3d_bboxes[bbox_count, 5] = i
                bbox_count += 1

    # Trim to actual size
    result = combined_3d_bboxes[:bbox_count].copy()
    log(f"Generated {bbox_count} 3D bounding boxes.")

    return result


def generate_bb3d(view1: list, view2: list, output_file: str, is_normalized: bool = False):
    """Process a sequence of images and their associated bounding box files to generate 3D bounding boxes.
    The results are saved to a specified output file.

    @param bbox_dir (str): Directory containing bounding box text files.
    @param image_dir (str): Directory containing images corresponding to the bounding boxes.
    @param start_index (int): The starting index of images to be processed.
    @param end_index (int): The ending index of images to be processed.
    @param output_file (str): Path for saving the resulting 3D bounding boxes as a numpy file.
    @param is_normalized (bool): Indicates whether the bounding box coordinates are normalized (default False).
    """
    # Generate a range of image indices to process

    bbox_dir, image_dir, start_index, end_index = view1
    bbox_dir1, image_dir1, start_index1, end_index1 = view2

    image_indices = range(start_index, end_index + 1)
    image_indices1 = range(start_index1, end_index1 + 1)

    # checking

    # Merge 2D bounding boxes from all specified images into 3D bounding boxes
    combined_3d_bboxes = merge_2d_bboxes_into_3d(image_indices, bbox_dir, image_dir, is_normalized)
    # view2
    combined_3d_bboxes1 = merge_2d_bboxes_into_3d(image_indices1, bbox_dir1, image_dir1, is_normalized)

    # Check if any 3D bounding boxes were generated and save to the output file
    if combined_3d_bboxes is not None and combined_3d_bboxes.size > 0:
        np.save(output_file, combined_3d_bboxes)
        np.save(output_file[:-4] + "_1.npy", combined_3d_bboxes1)
        log(f"3D bounding boxes with shape {combined_3d_bboxes.shape} saved to {output_file}.", level="debug")
    else:
        log("No 3D bounding boxes were generated.", level="warning")

    final_3d_bbox = compute_bbox_intersections(combined_3d_bboxes, combined_3d_bboxes1, end_index)

    if len(final_3d_bbox) < combined_3d_bboxes.shape[0] / 2:
        final_3d_bbox = combined_3d_bboxes.tolist()

    # Compute volumes
    volumes = compute_volumes(np.array(final_3d_bbox))

    # Compute adaptive threshold
    threshold = adaptive_threshold(volumes)

    # Filter out small bounding boxes
    filtered_bboxes, filtered_volumes = filter_bboxes(np.array(final_3d_bbox), volumes, threshold)

    return filtered_bboxes


if __name__ == "__main__":
    if len(sys.argv) != 10:
        print("Usage: generate_bb3d.py <bbox_dir> <image_dir> <start_index> <end_index> <output_file>")
        sys.exit(1)

    # Parse command line arguments
    bbox_directory = sys.argv[1]
    image_directory = sys.argv[2]
    start_index = int(sys.argv[3])
    end_index = int(sys.argv[4])

    bbox_directory1 = sys.argv[5]
    image_directory1 = sys.argv[6]
    start_index1 = int(sys.argv[7])
    end_index1 = int(sys.argv[8])

    output_file_path = sys.argv[9]

    # Process images and bounding boxes
    view1 = [bbox_directory, image_directory, start_index, end_index]
    view2 = [bbox_directory1, image_directory1, start_index1, end_index1]

    generate_bb3d(view1, view2, output_file_path, is_normalized=False)
