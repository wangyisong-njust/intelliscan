#!/usr/bin/env python3
from __future__ import annotations

"""
3D Mask Post-Processing and Metrology Analysis

This module provides comprehensive 3D mask processing and metrology measurement
functions for semiconductor inspection. It handles segmented volume analysis,
defect quantification, and precise dimensional measurements.

Key Features:
- 3D mask cleaning and morphological operations
- Robust mask orientation using structural relationships (hamburger-stack logic)
- Bond line thickness (BLT) measurement
- Pad misalignment detection and quantification
- Void-to-solder ratio analysis
- Solder extrusion measurement
- Pillar dimension extraction
- Defect classification and flagging
- NIfTI format handling

Segmentation Classes:
- Class 1: CuPillar (Copper Pillar) - bottom layer
- Class 2: Solder - middle layer
- Class 3: Void - defects within solder
- Class 4: CuPad (Copper Pad) - top layer (memory die only)

Orientation Strategy:
The module uses a "hamburger-stack" approach to determine correct vertical orientation.
Components should stack from top to bottom as: CuPad → Solder → CuPillar
Key insight: Even with severe solder extrusion, solder will never extend below
the bottom surface of the copper pillar, providing a reliable structural constraint.
"""

import os
from dataclasses import dataclass

import nibabel as nib
import numpy as np
from scipy import ndimage

from utils import log

# ============================================================================
# Configuration Constants
# ============================================================================

PIXEL_SIZE_UM = 0.7  # Pixel size in micrometers
NUM_DECIMALS = 2  # Number of decimal places for rounding measurements
MAKE_CLEAN_DEFAULT = True  # Default value for mask cleaning

# Morphological cleaning parameters
CLEANUP_THRESHOLD = 40  # Max distance from center of mass to keep voxels
BINARY_CLOSING_ITERATIONS = 2  # Iterations for morphological closing

# Defect detection thresholds
VOID_RATIO_THRESHOLD = 0.30  # 30% void-to-solder ratio
PAD_MISALIGNMENT_THRESHOLD = 0.10  # 10% of pillar width
SOLDER_EXTRUSION_THRESHOLD = 0.10  # 10% of pillar width

# Segmentation class labels
CLASS_COPPER_PILLAR = 1
CLASS_SOLDER = 2
CLASS_VOID = 3
CLASS_COPPER_PAD = 4


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class BoundingBox3D:
    """3D bounding box coordinates."""

    x_min: int
    x_max: int
    y_min: int
    y_max: int
    z_min: int
    z_max: int

    @property
    def width(self) -> int:
        """Width along x-axis."""
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        """Height along y-axis."""
        return self.y_max - self.y_min

    @property
    def depth(self) -> int:
        """Depth along z-axis."""
        return self.z_max - self.z_min


@dataclass
class PillarDimensions:
    """Pillar dimensional measurements."""

    width: float  # in pixels
    height: float  # in pixels

    def to_micrometers(self, pixel_size_um: float) -> "PillarDimensions":
        """Convert dimensions to micrometers."""
        return PillarDimensions(
            width=self.width * pixel_size_um,
            height=self.height * pixel_size_um,
        )


@dataclass
class DefectFlags:
    """Defect detection flags."""

    void_ratio_defect: bool = False
    solder_extrusion_defect: bool = False
    pad_misalignment_defect: bool = False


@dataclass
class MetrologyMeasurements:
    """Complete metrology measurement results (in pixels)."""

    is_memory: bool
    blt: float  # Bond line thickness
    pad_misalignment: float
    void_solder_ratio: float
    solder_extrusion_left: float
    solder_extrusion_right: float
    solder_extrusion_front: float
    solder_extrusion_back: float
    pillar_width: float
    pillar_height: float
    defects: DefectFlags

    def to_dict(self, pixel_size_um: float = PIXEL_SIZE_UM, num_decimals: int = NUM_DECIMALS) -> dict:
        """
        Convert measurements to dictionary with physical units (micrometers).

        Args:
            pixel_size_um: Pixel size in micrometers
            num_decimals: Number of decimal places for rounding

        Returns:
            Dictionary with measurements in micrometers
        """

        def to_um(value: float) -> float:
            """Convert pixel value to micrometers and round."""
            return round(float(value) * pixel_size_um, num_decimals)

        return {
            "is_memory": self.is_memory,
            "blt": to_um(self.blt),
            "pad_misalignment": to_um(self.pad_misalignment),
            "void_solder_ratio": round(self.void_solder_ratio, num_decimals),
            "solder_extrusion_left": to_um(self.solder_extrusion_left),
            "solder_extrusion_right": to_um(self.solder_extrusion_right),
            "solder_extrusion_front": to_um(self.solder_extrusion_front),
            "solder_extrusion_back": to_um(self.solder_extrusion_back),
            "empty_connection": -1,  # Placeholder for compatibility
            "pillar_width": to_um(self.pillar_width),
            "pillar_height": to_um(self.pillar_height),
            "void_ratio_defect": self.defects.void_ratio_defect,
            "solder_extrusion_defect": self.defects.solder_extrusion_defect,
            "pad_misalignment_defect": self.defects.pad_misalignment_defect,
        }


# ============================================================================
# Helper Functions
# ============================================================================


def count_nonzero_voxels(mask: np.ndarray) -> int:
    """
    Count the number of non-zero voxels in a 3D mask.

    Args:
        mask: 3D binary mask array

    Returns:
        Number of non-zero voxels
    """
    return int(np.count_nonzero(mask))


def get_bounding_box(mask_3d: np.ndarray) -> tuple[int, BoundingBox3D]:
    """
    Calculate the bounding box of non-zero elements in a 3D mask.

    Args:
        mask_3d: 3D binary mask array

    Returns:
        Tuple of (num_nonzero_voxels, BoundingBox3D)
        Returns (-1, empty bbox) if no non-zero elements found
    """
    coords = np.nonzero(mask_3d)
    num_voxels = len(coords[0])

    if num_voxels == 0:
        return -1, BoundingBox3D(-1, -1, -1, -1, -1, -1)

    x_coords, y_coords, z_coords = coords
    bbox = BoundingBox3D(
        x_min=int(x_coords.min()),
        x_max=int(x_coords.max()),
        y_min=int(y_coords.min()),
        y_max=int(y_coords.max()),
        z_min=int(z_coords.min()),
        z_max=int(z_coords.max()),
    )

    return num_voxels, bbox


def get_center_of_mass(mask_3d: np.ndarray) -> np.ndarray:
    """
    Calculate the center of mass for a 3D mask using median coordinates.

    Args:
        mask_3d: 3D binary mask array

    Returns:
        Array of [x_median, y_median, z_median] coordinates
    """
    coords = np.nonzero(mask_3d)
    return np.array([np.median(coords[0]), np.median(coords[1]), np.median(coords[2])])


def _check_vertical_alignment(
    pillar_bbox: BoundingBox3D, solder_bbox: BoundingBox3D, pad_bbox: BoundingBox3D | None, axis: int
) -> tuple[bool, str]:
    """
    Check if components are properly aligned along a given axis (hamburger stack check).

    The proper vertical alignment should be (from top to bottom):
    - CuPad (if present) at the top
    - Solder in the middle
    - CuPillar at the bottom

    Args:
        pillar_bbox: Bounding box of copper pillar
        solder_bbox: Bounding box of solder
        pad_bbox: Bounding box of copper pad (None if not present)
        axis: Axis index to check (0=x, 1=y, 2=z)

    Returns:
        Tuple of (is_correctly_aligned, reason_message)
    """

    # Extract min/max coordinates along the specified axis
    def get_bounds(bbox: BoundingBox3D, axis: int) -> tuple[int, int]:
        if axis == 0:
            return bbox.x_min, bbox.x_max
        elif axis == 1:
            return bbox.y_min, bbox.y_max
        else:  # axis == 2
            return bbox.z_min, bbox.z_max

    pillar_min, pillar_max = get_bounds(pillar_bbox, axis)
    solder_min, solder_max = get_bounds(solder_bbox, axis)

    # Key check: Solder should NOT reach below the pillar's bottom
    # Even with extrusion, solder stays above the pillar's minimum coordinate
    if solder_min < pillar_min:
        return False, f"Solder extends below pillar bottom (solder_min={solder_min} < pillar_min={pillar_min})"

    # Check that solder overlaps with pillar (they should be in contact/overlapping)
    # Solder's bottom should be at or above pillar's top
    if solder_max < pillar_min:
        return False, "Solder is completely separated from pillar (gap detected)"

    # If pad exists, check pad-solder-pillar ordering
    if pad_bbox is not None:
        pad_min, pad_max = get_bounds(pad_bbox, axis)

        # Pad should be above solder (pad's max should be >= solder's max for memory die)
        if pad_max < solder_max:
            return False, f"Pad is below solder (pad_max={pad_max} < solder_max={solder_max})"

        # Pad should not extend below pillar
        if pad_min < pillar_min:
            return False, "Pad extends below pillar bottom"

    return True, "Components properly stacked"


def orient_mask_by_structure(mask_3d: np.ndarray) -> np.ndarray:
    """
    Orient 3D mask based on structural relationships between components (hamburger-stack logic).

    This function uses the physical arrangement of components to determine the correct
    vertical orientation. The expected stack order from top to bottom is:
        CuPad (memory die only) → Solder → CuPillar

    Key insight: Regardless of solder extrusion (horizontal spread), the solder
    will never extend below the bottom surface of the copper pillar. This provides
    a reliable indicator of correct vertical orientation.

    Args:
        mask_3d: Input 3D segmentation mask array

    Returns:
        Correctly oriented 3D mask with vertical axis aligned to y-axis

    Raises:
        ValueError: If required components (pillar or solder) are not found
    """
    # Extract component masks and bounding boxes
    num_pillar, bbox_pillar_orig = get_bounding_box(mask_3d == CLASS_COPPER_PILLAR)
    num_solder, bbox_solder_orig = get_bounding_box(mask_3d == CLASS_SOLDER)

    if num_pillar <= 0 or num_solder <= 0:
        raise ValueError("Cannot orient mask: missing required components (pillar or solder)")

    # Check if pad exists (memory die)
    num_pad, bbox_pad_orig = get_bounding_box(mask_3d == CLASS_COPPER_PAD)
    has_pad = num_pad > 0

    log(f"Checking orientation (has_pad={has_pad})...")

    # Try all three possible orientations
    # Each transpose operation rotates a different axis to become the new y-axis (vertical)
    orientation_configs = [
        {"name": "y-axis (original)", "transpose": None, "axis": 1},
        {"name": "x-axis → y-axis", "transpose": (1, 0, 2), "axis": 0},  # x becomes y, y becomes x
        {"name": "z-axis → y-axis", "transpose": (0, 2, 1), "axis": 2},  # z becomes y, y becomes z
    ]

    for config in orientation_configs:
        # Get bounding boxes for this orientation
        if config["transpose"] is None:
            test_mask = mask_3d
            bbox_pillar = bbox_pillar_orig
            bbox_solder = bbox_solder_orig
            bbox_pad = bbox_pad_orig if has_pad else None
        else:
            test_mask = np.transpose(mask_3d, config["transpose"])
            _, bbox_pillar = get_bounding_box(test_mask == CLASS_COPPER_PILLAR)
            _, bbox_solder = get_bounding_box(test_mask == CLASS_SOLDER)
            if has_pad:
                _, bbox_pad = get_bounding_box(test_mask == CLASS_COPPER_PAD)
            else:
                bbox_pad = None

        # Check vertical alignment along y-axis (axis=1)
        is_aligned, reason = _check_vertical_alignment(bbox_pillar, bbox_solder, bbox_pad, axis=1)

        log(f"  {config['name']}: {reason}")

        if is_aligned:
            log(f"✓ Correct orientation found: {config['name']}")
            if config["transpose"] is not None:
                log(f"  Applying transpose: {config['transpose']}")
                return np.transpose(mask_3d, config["transpose"])
            else:
                log("  No rotation needed")
                return mask_3d

    # If no valid orientation found, issue warning and return original
    log("⚠ Warning: No valid orientation found. Returning original mask.")
    log("  This may indicate unusual data or segmentation errors.")
    return mask_3d


def orient_mask_vertically(mask_3d: np.ndarray) -> np.ndarray:
    """
    DEPRECATED: Rotate 3D mask to align the longest dimension with the y-axis.

    This function is deprecated in favor of orient_mask_by_structure() which uses
    structural relationships between components for more robust orientation.

    This legacy approach uses the longest dimension of the copper pillar to determine
    orientation, which may fail with non-standard geometries.

    Args:
        mask_3d: Input 3D mask array

    Returns:
        Rotated 3D mask with longest dimension aligned to y-axis

    .. deprecated:: 2.0
        Use :func:`orient_mask_by_structure` instead for better reliability.
    """
    # Get bounding box of the copper pillar (assumed to be class 1)
    _, bbox = get_bounding_box(mask_3d == CLASS_COPPER_PILLAR)

    if bbox.x_min == -1:  # No pillar found
        return mask_3d

    # Determine which axis is longest
    dims = {
        "x": bbox.width,
        "y": bbox.height,
        "z": bbox.depth,
    }

    # If y is already the longest, no rotation needed
    if dims["y"] >= max(dims["x"], dims["z"]):
        return mask_3d

    # If x is longest, transpose to make x → y
    if dims["x"] >= dims["z"]:
        log("Rotating: x-axis is longest, transposing to y-axis")
        return np.transpose(mask_3d, (1, 0, 2))

    # If z is longest, transpose to make z → y
    log("Rotating: z-axis is longest, transposing to y-axis")
    return np.transpose(mask_3d, (2, 0, 1))


# ============================================================================
# Mask Cleaning Functions
# ============================================================================


def clean_binary_mask_bounded(
    mask: np.ndarray,
    threshold: int = CLEANUP_THRESHOLD,
) -> np.ndarray:
    """
    Clean a single binary mask using bounded distance filtering.

    Optimized version that only computes distances within the mask's bounding box
    plus padding, rather than the full volume.

    Args:
        mask: 3D binary mask array
        threshold: Maximum distance from center of mass to retain voxels

    Returns:
        Cleaned binary mask
    """
    if not np.any(mask):
        return mask

    # Apply morphological closing to fill small holes and connect nearby components
    mask = ndimage.binary_closing(mask, iterations=BINARY_CLOSING_ITERATIONS)

    # Get tight bounding box of the mask
    coords = np.nonzero(mask)
    if len(coords[0]) == 0:
        return mask

    x_min, x_max = coords[0].min(), coords[0].max() + 1
    y_min, y_max = coords[1].min(), coords[1].max() + 1
    z_min, z_max = coords[2].min(), coords[2].max() + 1

    # Add padding for threshold distance
    pad = threshold + 5
    x_min = max(0, x_min - pad)
    x_max = min(mask.shape[0], x_max + pad)
    y_min = max(0, y_min - pad)
    y_max = min(mask.shape[1], y_max + pad)
    z_min = max(0, z_min - pad)
    z_max = min(mask.shape[2], z_max + pad)

    # Extract crop
    crop = mask[x_min:x_max, y_min:y_max, z_min:z_max]

    # Calculate center of mass within crop
    center_of_mass = ndimage.center_of_mass(crop)

    # Create coordinate grids only for the crop (much smaller)
    x_coords, y_coords, z_coords = np.ogrid[: crop.shape[0], : crop.shape[1], : crop.shape[2]]

    # Calculate distance from each voxel to center of mass
    distances = np.sqrt(
        (x_coords - center_of_mass[0]) ** 2 + (y_coords - center_of_mass[1]) ** 2 + (z_coords - center_of_mass[2]) ** 2
    )

    # Apply threshold within crop
    crop = crop & (distances <= threshold)

    # Place back into result
    result = np.zeros_like(mask)
    result[x_min:x_max, y_min:y_max, z_min:z_max] = crop

    return result


def clean_segmentation_mask(mask_3d: np.ndarray) -> np.ndarray:
    """
    Clean a multi-class 3D segmentation mask.

    Processes each class separately using morphological operations,
    then recombines them into a single mask. Uses bounded distance
    computation for efficiency.

    Args:
        mask_3d: 3D segmentation mask with integer class labels

    Returns:
        Cleaned 3D segmentation mask
    """
    log(f"Cleaning mask with shape: {mask_3d.shape}")
    mask_3d = np.round(mask_3d).astype(np.uint8)

    # Extract and clean each class separately
    mask_pillar = mask_3d == CLASS_COPPER_PILLAR
    mask_solder = mask_3d == CLASS_SOLDER
    mask_void = mask_3d == CLASS_VOID
    mask_pad = mask_3d == CLASS_COPPER_PAD

    log(f"Copper pillar voxels: {count_nonzero_voxels(mask_pillar)}")

    # Clean non-empty masks using bounded computation
    if count_nonzero_voxels(mask_pillar) > 0:
        mask_pillar = clean_binary_mask_bounded(mask_pillar)

    if count_nonzero_voxels(mask_solder) > 0:
        mask_solder = clean_binary_mask_bounded(mask_solder)

    if count_nonzero_voxels(mask_pad) > 0:
        mask_pad = clean_binary_mask_bounded(mask_pad)

    # Note: Void mask is not cleaned to preserve all void detections

    # Recombine masks with priority order (later overrides earlier)
    cleaned_mask = np.zeros_like(mask_3d, dtype=np.uint8)
    cleaned_mask[mask_solder] = CLASS_SOLDER
    cleaned_mask[mask_pillar] = CLASS_COPPER_PILLAR
    cleaned_mask[mask_pad] = CLASS_COPPER_PAD
    cleaned_mask[mask_void] = CLASS_VOID

    return cleaned_mask


# ============================================================================
# Measurement Functions
# ============================================================================


def compute_measurements(mask_3d: np.ndarray) -> MetrologyMeasurements | None:
    """
    Compute all metrology measurements from a 3D segmentation mask.

    Args:
        mask_3d: 3D segmentation mask with class labels:
            - 1: Copper Pillar
            - 2: Solder
            - 3: Void
            - 4: Copper Pad (memory die only)

    Returns:
        MetrologyMeasurements object with all computed values in pixels,
        or None if mask is invalid
    """
    num_classes = int(np.max(mask_3d) + 0.2)
    log(f"Number of segmentation classes found: {num_classes}")

    # Validate number of classes
    if num_classes < 2 or num_classes > 4:
        log(f"Invalid number of classes: {num_classes}. Expected 2-4.")
        return None

    # Determine if this is a memory die (has copper pad class)
    is_memory = num_classes == 4

    # Extract class masks and compute bounding boxes
    num_pillar, bbox_pillar = get_bounding_box(mask_3d == CLASS_COPPER_PILLAR)
    num_solder, bbox_solder = get_bounding_box(mask_3d == CLASS_SOLDER)

    if num_pillar <= 0 or num_solder <= 0:
        log("Error: Missing required classes (pillar or solder)")
        return None

    # Initialize measurements
    void_solder_ratio = 0.0
    pad_misalignment = -1.0
    blt = 0.0

    # Compute void-to-solder ratio if voids are present
    if num_classes >= 3:
        num_void, bbox_void = get_bounding_box(mask_3d == CLASS_VOID)
        if num_void > 0:
            void_solder_ratio = num_void / num_solder
            if void_solder_ratio > 0.50:
                log(f"Warning: High void ratio detected - void: {num_void}, solder: {num_solder}")

    # Memory die specific measurements
    if is_memory:
        num_pad, bbox_pad = get_bounding_box(mask_3d == CLASS_COPPER_PAD)
        if num_pad <= 0:
            log("Warning: Memory die detected but no copper pad found")
            return None

        # Bond line thickness: distance from pillar bottom to pad top
        blt = (bbox_pad.y_max - bbox_pillar.y_min) + 1

        # Pad misalignment: lateral distance between pillar and pad centers
        com_pillar = get_center_of_mass(mask_3d == CLASS_COPPER_PILLAR)
        com_pad = get_center_of_mass(mask_3d == CLASS_COPPER_PAD)
        pad_misalignment = np.sqrt((com_pad[0] - com_pillar[0]) ** 2 + (com_pad[2] - com_pillar[2]) ** 2)

        # Solder extrusion relative to both pillar and pad
        solder_extrusion_left = bbox_solder.x_min - bbox_pillar.x_min
        solder_extrusion_right = bbox_pillar.x_max - bbox_solder.x_max
        solder_extrusion_front = bbox_solder.z_min - bbox_pillar.z_min
        solder_extrusion_back = bbox_pillar.z_max - bbox_solder.z_max

    else:
        # Logic die measurements
        # Bond line thickness: distance from pillar bottom to solder top
        blt = (bbox_solder.y_max - bbox_pillar.y_min) + 1

        # Solder extrusion relative to pillar only
        solder_extrusion_left = bbox_solder.x_min - bbox_pillar.x_min
        solder_extrusion_right = bbox_pillar.x_max - bbox_solder.x_max
        solder_extrusion_front = bbox_solder.z_min - bbox_pillar.z_min
        solder_extrusion_back = bbox_pillar.z_max - bbox_solder.z_max

    # Compute pillar dimensions
    pillar_width = max(bbox_pillar.width, bbox_pillar.depth)
    pillar_height = bbox_pillar.height

    # Detect defects
    defects = DefectFlags()

    if void_solder_ratio > VOID_RATIO_THRESHOLD:
        defects.void_ratio_defect = True
        log(f"Defect detected: Void ratio {void_solder_ratio:.2%} exceeds threshold {VOID_RATIO_THRESHOLD:.2%}")

    max_extrusion = max(
        abs(solder_extrusion_left),
        abs(solder_extrusion_right),
        abs(solder_extrusion_front),
        abs(solder_extrusion_back),
    )
    if max_extrusion > (SOLDER_EXTRUSION_THRESHOLD * pillar_width):
        defects.solder_extrusion_defect = True
        log(f"Defect detected: Solder extrusion {max_extrusion:.1f}px exceeds threshold")

    if is_memory and pad_misalignment > (PAD_MISALIGNMENT_THRESHOLD * pillar_width):
        defects.pad_misalignment_defect = True
        log(f"Defect detected: Pad misalignment {pad_misalignment:.1f}px exceeds threshold")

    return MetrologyMeasurements(
        is_memory=is_memory,
        blt=float(blt),
        pad_misalignment=float(pad_misalignment),
        void_solder_ratio=float(void_solder_ratio),
        solder_extrusion_left=float(solder_extrusion_left),
        solder_extrusion_right=float(solder_extrusion_right),
        solder_extrusion_front=float(solder_extrusion_front),
        solder_extrusion_back=float(solder_extrusion_back),
        pillar_width=float(pillar_width),
        pillar_height=float(pillar_height),
        defects=defects,
    )


# ============================================================================
# I/O Functions
# ============================================================================


def save_nifti(data: np.ndarray, save_path: str, pixel_size: float = PIXEL_SIZE_UM) -> None:
    """
    Save a 3D numpy array as a NIfTI file.

    Args:
        data: 3D uint8 array containing the segmentation mask
        save_path: Output path (without .nii.gz extension)
        pixel_size: Voxel size in micrometers (currently not used in affine)

    Raises:
        AssertionError: If data is not uint8
    """
    assert data.dtype == np.uint8, f"Expected uint8, got {data.dtype}"

    # Create NIfTI image with identity affine (TODO: incorporate pixel_size)
    nifti_image = nib.Nifti1Image(data, affine=None)
    output_file = save_path if save_path.endswith(".nii.gz") else f"{save_path}.nii.gz"
    nib.save(nifti_image, output_file)
    log(f"Saved cleaned mask to: {output_file}")


# ============================================================================
# Main API Function
# ============================================================================


def compute_metrology_info(
    nii_file: str,
    output_path: str | None = None,
    clean_mask: bool = MAKE_CLEAN_DEFAULT,
    pixel_size_um: float = PIXEL_SIZE_UM,
    num_decimals: int = NUM_DECIMALS,
) -> dict:
    """
    Compute comprehensive metrology information from a NIfTI segmentation file.

    This is the main API function for the metrology module. It loads a 3D
    segmentation mask, optionally cleans it, orients it correctly, and
    computes all metrology measurements.

    Args:
        nii_file: Path to input NIfTI file (.nii or .nii.gz)
        output_path: Path to save cleaned mask (if clean_mask=True)
        clean_mask: Whether to apply morphological cleaning
        pixel_size_um: Pixel/voxel size in micrometers for unit conversion
        num_decimals: Decimal places for rounding measurements

    Returns:
        Dictionary containing all metrology measurements in micrometers:
            - is_memory (bool): Whether memory die is present
            - blt (float): Bond line thickness
            - pad_misalignment (float): Pad misalignment
            - void_solder_ratio (float): Void-to-solder ratio (dimensionless)
            - solder_extrusion_left/right/front/back (float): Solder extrusion
            - pillar_width (float): Pillar width
            - pillar_height (float): Pillar height
            - void_ratio_defect (bool): Void defect flag
            - solder_extrusion_defect (bool): Extrusion defect flag
            - pad_misalignment_defect (bool): Misalignment defect flag
            - empty_connection (int): Placeholder (-1)

    Raises:
        FileNotFoundError: If nii_file doesn't exist
        ValueError: If mask is invalid or measurements cannot be computed
    """
    log("\n" + "=" * 60)
    log(f"Processing: {nii_file}")
    log("=" * 60)

    # Load NIfTI file
    if not os.path.exists(nii_file):
        raise FileNotFoundError(f"Input file not found: {nii_file}")

    nifti_data = nib.load(nii_file)
    mask_3d = nifti_data.get_fdata()

    # Clean mask if requested
    if clean_mask:
        log("Applying morphological cleaning...")
        mask_3d = clean_segmentation_mask(mask_3d)

        if output_path:
            save_nifti(mask_3d.astype(np.uint8), output_path, pixel_size_um)

    # Orient mask vertically (pillar along y-axis) using structural relationships
    log("Orienting mask...")
    try:
        mask_3d = orient_mask_by_structure(mask_3d)
    except ValueError as e:
        log(f"Warning: {e}. Falling back to legacy orientation method.")
        mask_3d = orient_mask_vertically(mask_3d)

    # Compute measurements
    log("Computing metrology measurements...")
    measurements = compute_measurements(mask_3d)

    if measurements is None:
        raise ValueError(f"Failed to compute measurements for {nii_file}")

    # Convert to dictionary with physical units
    result = measurements.to_dict(pixel_size_um, num_decimals)

    log("\nMeasurement Summary:")
    log(f"  Type: {'Memory Die' if result['is_memory'] else 'Logic Die'}")
    log(f"  BLT: {result['blt']:.{num_decimals}f} µm")
    log(f"  Void Ratio: {result['void_solder_ratio']:.{num_decimals}%}")
    log(f"  Pillar: {result['pillar_width']:.{num_decimals}f} × {result['pillar_height']:.{num_decimals}f} µm")
    num_defects = sum(
        [result["void_ratio_defect"], result["solder_extrusion_defect"], result["pad_misalignment_defect"]]
    )
    log(f"  Defects: {num_defects} detected")
    log("=" * 60 + "\n")

    return result


def compute_metrology_from_array(
    mask_3d: np.ndarray,
    clean_mask: bool = MAKE_CLEAN_DEFAULT,
    pixel_size_um: float = PIXEL_SIZE_UM,
    num_decimals: int = NUM_DECIMALS,
) -> dict:
    """
    Compute metrology information directly from a numpy array.

    This function bypasses file I/O for use in combined segmentation+metrology
    pipelines where the mask is already in memory.

    Args:
        mask_3d: 3D segmentation mask array with class labels
        clean_mask: Whether to apply morphological cleaning
        pixel_size_um: Pixel/voxel size in micrometers for unit conversion
        num_decimals: Decimal places for rounding measurements

    Returns:
        Dictionary containing all metrology measurements in micrometers
    """
    # Clean mask if requested
    if clean_mask:
        mask_3d = clean_segmentation_mask(mask_3d)

    # Orient mask vertically (pillar along y-axis) using structural relationships
    try:
        mask_3d = orient_mask_by_structure(mask_3d)
    except ValueError:
        mask_3d = orient_mask_vertically(mask_3d)

    # Compute measurements
    measurements = compute_measurements(mask_3d)

    if measurements is None:
        raise ValueError("Failed to compute measurements from array")

    # Convert to dictionary with physical units
    return measurements.to_dict(pixel_size_um, num_decimals)


# ============================================================================
# Test/Demo
# ============================================================================

if __name__ == "__main__":
    import tempfile

    log("3D Metrology Post-Processing Module v2.0")
    log("Testing with test.nii.gz...")

    with tempfile.TemporaryDirectory(prefix="intelliscan-metrology-") as temp_dir:
        try:
            measurements = compute_metrology_info(
                nii_file="test.nii.gz",
                # output_path=os.path.join(temp_dir, "cleaned_mask"),
                clean_mask=True,
            )

            log("\nDetailed Results:")
            log("-" * 60)
            for key, value in measurements.items():
                log(f"{key:30s}: {value}")
            log("-" * 60)

        except FileNotFoundError:
            log("Error: test.nii.gz not found. Please provide a test file.")
        except Exception as e:
            log(f"Error during processing: {e}")
            import traceback

            traceback.print_exc()
