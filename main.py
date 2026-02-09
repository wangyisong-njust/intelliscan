#!/usr/bin/env python3
from __future__ import annotations

"""
3D-IntelliScan Main Pipeline

This module serves as the main entry point for the 3D-IntelliScan semiconductor
metrology and defect detection pipeline. It orchestrates the complete workflow
from 3D NII to 2D conversion, object detection, 3D bounding box generation,
segmentation, and metrology analysis.

Pipeline Workflow:
1. Convert 3D NIfTI volumes to 2D slices (horizontal & vertical views)
2. Run YOLO object detection on 2D slices (single model for both views)
3. Generate 3D bounding boxes from 2D detections
4. Perform 3D semantic segmentation within bounding boxes
5. Compute metrology measurements (BLT, void ratio, defects)
6. Generate PDF reports with analysis

Usage:
    # Process files from files.txt (batch mode)
    python main.py

    # Process a single file
    python main.py /path/to/input.nii

    # Force reprocessing
    python main.py /path/to/input.nii --force

    # Use as module
    from main import process_single_file
    result = process_single_file("/path/to/input.nii", force=False)
"""

import argparse
import traceback
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from detection import run_yolo_detection, run_yolo_detection_inmemory
from merge import generate_bb3d, generate_bb3d_inmemory
from metrology import MAKE_CLEAN_DEFAULT, compute_metrology_from_array, compute_metrology_info
from report import generate_pdf_report
from segmentation import (
    SegmentationConfig,
    SegmentationInference,
    assemble_full_volume,
    expand_bbox,
    segment_bboxes,
)
from utils import PipelineLogbook, PipelineLogger, PipelineMetrics, create_folder_structure, log, nii2jpg


@dataclass
class PipelineConfig:
    """Pipeline configuration settings."""

    output_base: str = "output"
    detection_model: str = "models/detection_model.pt"
    segmentation_model: str = "models/segmentation_model.ckpt"
    use_combined_seg_metrology: bool = False
    use_inmemory_detection: bool = False
    verbose: bool = True
    clean_mask: bool = MAKE_CLEAN_DEFAULT
    use_trt: bool = False
    trt_engine_path: str | None = None
    enable_ai_analysis: bool = False


# Default configuration
DEFAULT_CONFIG = PipelineConfig()


def process_segmentation_and_metrology_combined(
    engine: SegmentationInference,
    volume: np.ndarray,
    bboxes: np.ndarray,
    seg_output_dir: Path,
    metrology_output_dir: Path,
    clean_mask: bool = MAKE_CLEAN_DEFAULT,
) -> tuple[list, pd.DataFrame]:
    """
    Combined segmentation and metrology processing per bbox.

    Processes each bbox: segment -> compute metrology -> save prediction.
    This reduces I/O by computing metrology directly from in-memory predictions
    instead of saving then reloading each prediction file.

    Args:
        engine: Initialized SegmentationInference instance
        volume: Full 3D volume array
        bboxes: Array of bboxes [N, 6] format [x_min, x_max, y_min, y_max, z_min, z_max]
        seg_output_dir: Directory to save per-bbox predictions
        metrology_output_dir: Directory to save metrology CSV
        clean_mask: Whether to apply morphological cleaning to masks

    Returns:
        Tuple of (segmentation_results, metrology_dataframe)
    """
    from segmentation import BBoxSegmentationResult

    results = []
    cfg = engine.config

    # Ensure model is loaded
    engine.load_model()

    metrology_output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize metrology DataFrame
    columns = [
        "filename",
        "is_memory",
        "void_ratio_defect",
        "solder_extrusion_defect",
        "pad_misalignment_defect",
        "bb",
        "BLT",
        "Pad_misalignment",
        "Void_to_solder_ratio",
        "solder_extrusion_copper_pillar",
        "pillar_width",
        "pillar_height",
    ]
    metrology_df = pd.DataFrame(columns=columns)

    for idx, bbox in enumerate(bboxes):
        # Expand bbox with margin
        expanded = expand_bbox(bbox, volume.shape, cfg.margin)

        # Check for valid region size
        region_size = (
            expanded[1] - expanded[0],
            expanded[3] - expanded[2],
            expanded[5] - expanded[4],
        )
        if any(s < 5 for s in region_size):
            log(f"Skipping bbox {idx}: region too small {region_size}", level="warning")
            continue

        log(f"Processing bbox {idx}: shape x={region_size[0]}, y={region_size[1]}, z={region_size[2]}", level="debug")

        # Extract crop
        crop = volume[
            expanded[0] : expanded[1],
            expanded[2] : expanded[3],
            expanded[4] : expanded[5],
        ]

        # Run segmentation inference
        prediction = engine.infer_crop(crop)

        # Compute metrology directly from in-memory prediction
        try:
            measurements = compute_metrology_from_array(
                prediction.copy(),  # copy to avoid mutation
                clean_mask=clean_mask,
            )

            # Create metrology record
            record = {
                "filename": f"pred_{idx}.nii.gz",
                "is_memory": measurements["is_memory"],
                "void_ratio_defect": measurements["void_ratio_defect"],
                "solder_extrusion_defect": measurements["solder_extrusion_defect"],
                "pad_misalignment_defect": measurements["pad_misalignment_defect"],
                "bb": bbox.tolist() if hasattr(bbox, "tolist") else list(bbox),
                "BLT": measurements["blt"],
                "Pad_misalignment": measurements["pad_misalignment"],
                "Void_to_solder_ratio": measurements["void_solder_ratio"],
                "solder_extrusion_copper_pillar": [
                    measurements["solder_extrusion_left"],
                    measurements["solder_extrusion_right"],
                    measurements["solder_extrusion_front"],
                    measurements["solder_extrusion_back"],
                ],
                "pillar_width": measurements["pillar_width"],
                "pillar_height": measurements["pillar_height"],
            }
            metrology_df = pd.concat([metrology_df, pd.DataFrame([record])], ignore_index=True)

        except Exception as e:
            log(f"Error computing metrology for bbox {idx}: {e}", level="error")

        # Store result
        results.append(
            BBoxSegmentationResult(
                bbox_index=idx,
                prediction=prediction,
                original_bbox=bbox,
                expanded_bbox=expanded,
                crop_shape=crop.shape,
            )
        )

    # Save metrology CSV
    metrology_df.to_csv(metrology_output_dir / "metrology.csv", index=False)

    return results, metrology_df


def process_metrology(
    input_folder: Path,
    output_folder: Path,
    clean_out_path: Path,
    clean_mask: bool,
    bb_3d_list: np.ndarray,
) -> pd.DataFrame:
    """
    Process 3D segmented files and compute metrology measurements.

    Iterates through all .nii.gz segmentation files, computes metrology
    measurements (BLT, void ratio, pad misalignment, etc.).

    Args:
        input_folder: Path to folder containing segmented .nii.gz files
        output_folder: Path to save metrology CSV reports
        clean_out_path: Path to save cleaned/processed masks
        clean_mask: Whether to apply morphological cleaning to masks
        bb_3d_list: Array of 3D bounding boxes [num_classes, num_samples]

    Returns:
        DataFrame with metrology results for all samples

    Output Files:
        - metrology.csv: Metrology data for all samples (memory and logic dies)
    """
    segmented_files = sorted(input_folder.rglob("*.nii.gz"))
    num_files = len(segmented_files)
    log(f"Number of segmented files to process: {num_files}")

    # Create output folders if they don't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    if clean_mask:
        clean_out_path.mkdir(parents=True, exist_ok=True)

    # Initialize DataFrame with all expected columns
    columns = [
        "filename",
        "is_memory",
        "void_ratio_defect",
        "solder_extrusion_defect",
        "pad_misalignment_defect",
        "bb",
        "BLT",
        "Pad_misalignment",
        "Void_to_solder_ratio",
        "solder_extrusion_copper_pillar",
        "pillar_width",
        "pillar_height",
    ]

    metrology_df = pd.DataFrame(columns=columns)

    for i, segmented_file in enumerate(segmented_files):
        log(f"Processing file {i + 1}/{num_files}: {segmented_file}", level="debug")
        rel_path = segmented_file.relative_to(input_folder)
        output_path = clean_out_path / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            measurements = compute_metrology_info(
                nii_file=str(segmented_file),
                output_path=str(output_path) if clean_mask else None,
                clean_mask=clean_mask,
            )

            # Create record with all measurements (values already in micrometers)
            record = {
                "filename": str(rel_path),
                "is_memory": measurements["is_memory"],
                "void_ratio_defect": measurements["void_ratio_defect"],
                "solder_extrusion_defect": measurements["solder_extrusion_defect"],
                "pad_misalignment_defect": measurements["pad_misalignment_defect"],
                "bb": bb_3d_list[0, i],
                "BLT": measurements["blt"],
                "Pad_misalignment": measurements["pad_misalignment"],
                "Void_to_solder_ratio": measurements["void_solder_ratio"],
                "solder_extrusion_copper_pillar": [
                    measurements["solder_extrusion_left"],
                    measurements["solder_extrusion_right"],
                    measurements["solder_extrusion_front"],
                    measurements["solder_extrusion_back"],
                ],
                "pillar_width": measurements["pillar_width"],
                "pillar_height": measurements["pillar_height"],
            }

            # Append to DataFrame
            metrology_df = pd.concat([metrology_df, pd.DataFrame([record])], ignore_index=True)

        except Exception as e:
            log(f"Error processing file {segmented_file}: {e}", level="error")
            log(f"Stack trace: {traceback.format_exc()}", level="debug")
            continue

    # Save DataFrame to CSV
    metrology_df.to_csv(output_folder / "metrology.csv", index=False)

    return metrology_df


def process_single_file(
    input_file: str | Path,
    config: PipelineConfig | None = None,
    force: bool = False,
) -> dict:
    """Process a single input file through the pipeline.

    This is the main entry point for processing a single file. It can be called
    directly as a module or via CLI.

    Args:
        input_file: Path to input NIfTI file
        config: Pipeline configuration (uses DEFAULT_CONFIG if None)
        force: Force reprocessing even if already completed

    Returns:
        Dictionary with processing result:
            - status: "completed", "skipped", or "failed"
            - output_dir: Path to output directory
            - reason: Reason for status (if skipped or failed)
            - metrics: Processing metrics (if completed)
    """
    config = config or DEFAULT_CONFIG
    input_file = Path(input_file)

    # Initialize logbook for job tracking
    logbook = PipelineLogbook(config.output_base)

    # Check if should process
    should_run, reason = logbook.should_process(input_file, force=force)
    if not should_run:
        return {"status": "skipped", "input_file": str(input_file), "reason": reason}

    # Get output directory using sample ID extraction
    outfolder = logbook.get_output_dir(input_file)
    outfolder.mkdir(parents=True, exist_ok=True)

    # Mark job as started
    logbook.mark_started(
        input_file,
        outfolder,
        config={
            "use_combined_seg_metrology": config.use_combined_seg_metrology,
            "use_inmemory_detection": config.use_inmemory_detection,
            "clean_mask": config.clean_mask,
        },
    )

    views = ["view1", "view2"]

    # Initialize logger and metrics for this task
    logger = PipelineLogger(log_file=outfolder / "execution.log", verbose=config.verbose)
    metrics = PipelineMetrics(task_id=str(input_file))

    try:
        with metrics.phase("Folder Creation"):
            folder_structure = create_folder_structure(str(outfolder), views)

        # Load volume data once for reuse across pipeline stages
        log(f"Loading volume: {input_file}")
        img = nib.load(input_file)
        data3d = img.get_fdata()
        if data3d.ndim == 4:
            data3d = data3d[..., 3]
        data_max = data3d.max()

        with metrics.phase("NII to JPG Conversion") as phase:
            log("Converting NII to JPG...")
            view1_slices = nii2jpg(data3d, folder_structure["view1"]["input_images"], 0, data_max)
            view2_slices = nii2jpg(data3d, folder_structure["view2"]["input_images"], 1, data_max)
            num_slices = view1_slices + view2_slices
            phase.complete(count=num_slices)

        if config.use_inmemory_detection:
            # In-memory detection pipeline (faster, no intermediate files)
            with metrics.phase("2D Detection Inference") as phase:
                log("Running object detection inference (in-memory)...")

                log("Running detection on view1...")
                view1_detections = run_yolo_detection_inmemory(
                    model_path=config.detection_model,
                    input_folder=folder_structure["view1"]["input_images"],
                )

                log("Running detection on view2...")
                view2_detections = run_yolo_detection_inmemory(
                    model_path=config.detection_model,
                    input_folder=folder_structure["view2"]["input_images"],
                )

                num_slices = len(view1_detections) + len(view2_detections)
                phase.complete(count=num_slices)

            with metrics.phase("3D Bounding Box Generation") as phase:
                log("Generating 3D bounding boxes (in-memory)...")

                # Get image dimensions from first image
                from PIL import Image

                sample_image_path = Path(folder_structure["view1"]["input_images"]) / "image0.jpg"
                with Image.open(sample_image_path) as pil_img:
                    image_dimensions = pil_img.size  # (width, height)

                view1_num_slices = len(view1_detections)
                view2_num_slices = len(view2_detections)

                bb_3d = generate_bb3d_inmemory(
                    view1_detections=view1_detections,
                    view2_detections=view2_detections,
                    view1_num_slices=view1_num_slices,
                    view2_num_slices=view2_num_slices,
                    image_dimensions=image_dimensions,
                    output_file=str(outfolder / "3d_bounding_boxes.npy"),
                    is_normalized=False,
                )

                # Save as single array
                np.save(outfolder / "bb3d.npy", bb_3d)

                # Free detection memory after merging
                del view1_detections, view2_detections

                effective_slices = view1_num_slices + view2_num_slices
                phase.complete(count=effective_slices)

            # Explicit cleanup before loading segmentation model
            import gc

            gc.collect()

        else:
            # File-based detection pipeline (original behavior)
            with metrics.phase("2D Detection Inference") as phase:
                log("Running object detection inference...")

                for view in views:
                    input_folder = folder_structure[view]["input_images"]
                    output_folder = folder_structure[view]["detections"]

                    log(f"Running detection on {view}...")
                    run_yolo_detection(
                        model_path=config.detection_model,
                        input_folder=input_folder,
                        output_folder=output_folder,
                    )

                view1_dir = Path(folder_structure["view1"]["input_images"])
                view2_dir = Path(folder_structure["view2"]["input_images"])
                num_slices = len(list(view1_dir.glob("*.jpg"))) + len(list(view2_dir.glob("*.jpg")))
                phase.complete(count=num_slices)

            with metrics.phase("3D Bounding Box Generation") as phase:
                log("Generating 3D bounding boxes...")
                view1_dir = Path(folder_structure["view1"]["input_images"])
                view2_dir = Path(folder_structure["view2"]["input_images"])
                view1_num_slices = len(list(view1_dir.glob("*.jpg")))
                view2_num_slices = len(list(view2_dir.glob("*.jpg")))
                view1_params = [
                    folder_structure["view1"]["detections"],
                    folder_structure["view1"]["input_images"],
                    0,
                    view1_num_slices,
                ]
                view2_params = [
                    folder_structure["view2"]["detections"],
                    folder_structure["view2"]["input_images"],
                    0,
                    view2_num_slices,
                ]
                bb_3d = generate_bb3d(
                    view1_params, view2_params, str(outfolder / "3d_bounding_boxes.npy"), is_normalized=False
                )
                np.save(outfolder / "bb3d.npy", bb_3d)
                effective_slices = view1_num_slices + view2_num_slices
                phase.complete(count=effective_slices)

        num_bboxes = len(bb_3d) if bb_3d.ndim == 1 else bb_3d.shape[0]
        log(f"3D bounding boxes shape: {bb_3d.shape}")

        seg_output_dir = outfolder / "mmt"
        seg_config = SegmentationConfig(
            use_trt=config.use_trt,
            trt_engine_path=config.trt_engine_path,
        )
        engine = SegmentationInference(config.segmentation_model, seg_config)

        if config.use_combined_seg_metrology:
            # Combined segmentation + metrology in single phase
            with metrics.phase("3D Segmentation + Metrology") as phase:
                log("Running combined 3D segmentation and metrology...")

                results, metrology_df = process_segmentation_and_metrology_combined(
                    engine=engine,
                    volume=data3d,
                    bboxes=bb_3d,
                    seg_output_dir=seg_output_dir,
                    metrology_output_dir=outfolder / "metrology",
                    clean_mask=config.clean_mask,
                )

                # Assemble into full volume and save
                full_segmentation = assemble_full_volume(results, data3d.shape)
                nib.save(
                    nib.Nifti1Image(full_segmentation, np.eye(4)),
                    str(outfolder / "segmentation.nii.gz"),
                )
                log(f"Combined segmentation + metrology completed, saved to {outfolder}")
                phase.complete(count=num_bboxes)

        else:
            # Separate segmentation and metrology phases
            with metrics.phase("3D Segmentation") as phase:
                log("Running 3D segmentation...")

                # Segment all bounding boxes (save nii.gz for metrology + report)
                (seg_output_dir / "img").mkdir(parents=True, exist_ok=True)
                (seg_output_dir / "pred").mkdir(parents=True, exist_ok=True)
                results = segment_bboxes(engine, data3d, bb_3d, save_dir=seg_output_dir)

                # Assemble into full volume and save
                full_segmentation = assemble_full_volume(results, data3d.shape)
                nib.save(
                    nib.Nifti1Image(full_segmentation, np.eye(4)),
                    str(outfolder / "segmentation.nii.gz"),
                )
                log(f"Segmentation completed, saved to {outfolder}")
                phase.complete(count=num_bboxes)

            with metrics.phase("Metrology") as phase:
                log("Computing metrology measurements...")
                metrology_input_path = seg_output_dir / "pred"
                if not metrology_input_path.exists():
                    log(f"Metrology input folder does not exist: {metrology_input_path}", level="error")
                else:
                    mask_files = list(metrology_input_path.rglob("*.nii.gz"))
                    log(f"Found {len(mask_files)} .nii.gz files to process")
                    if len(mask_files) == 0:
                        log("No files found! Check if previous steps generated files.", level="warning")

                metrology_df = process_metrology(
                    input_folder=metrology_input_path,
                    output_folder=outfolder / "metrology",
                    clean_out_path=outfolder / "metrology" / "clean",
                    clean_mask=config.clean_mask,
                    bb_3d_list=bb_3d[np.newaxis, :] if bb_3d.ndim == 2 else bb_3d,
                )
                phase.complete(count=len(results))

        report_path = outfolder / "metrology" / "metrology_report.pdf"
        with metrics.phase("Report Generation"):
            generate_pdf_report(
                str(outfolder / "metrology" / "metrology.csv"),
                str(report_path),
                input_filename=input_file.name,
                enable_ai_analysis=config.enable_ai_analysis,
            )

        # Save metrics for this task
        metrics.write_log(str(outfolder / "timing.log"))
        metrics.save(str(outfolder / "metrics.json"))

        log(f"Successfully processed {input_file}")
        log(f"Report generated at {report_path.parent}")

        # Mark job as completed
        logbook.mark_completed(input_file, metrics.summary())

        return {
            "status": "completed",
            "input_file": str(input_file),
            "output_dir": str(outfolder),
            "metrics": metrics.summary(),
        }

    except Exception as e:
        error_msg = f"{e}\n{traceback.format_exc()}"
        log(f"Error processing {input_file}: {e}", level="error")

        # Still save partial metrics on error
        metrics.write_log(str(outfolder / "timing.log"))
        metrics.save(str(outfolder / "metrics.json"))

        # Mark job as failed
        logbook.mark_failed(input_file, str(e))

        return {
            "status": "failed",
            "input_file": str(input_file),
            "output_dir": str(outfolder),
            "error": error_msg,
        }

    finally:
        logger.close()


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="3D-IntelliScan Pipeline: Detection, Segmentation, and Metrology",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Process all files in files.txt
  python main.py /path/to/input.nii       # Process single file
  python main.py /path/to/input.nii --force  # Force reprocessing
  python main.py --list                   # List all jobs in logbook
        """,
    )
    parser.add_argument("input_file", nargs="?", help="Input NIfTI file to process")
    parser.add_argument("--force", "-f", action="store_true", help="Force reprocessing")
    parser.add_argument("--output", "-o", default="output", help="Output base directory")
    parser.add_argument("--verbose", "-v", action="store_true", default=True, help="Verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode (only essential output)")
    parser.add_argument("--list", action="store_true", help="List all jobs in logbook")
    parser.add_argument("--inmemory", action="store_true", help="Use in-memory detection (faster)")
    parser.add_argument("--trt", action="store_true", help="Use TensorRT backend for segmentation")
    parser.add_argument("--trt-engine", type=str, default=None, help="Path to TensorRT engine file")
    parser.add_argument("--ai-analysis", action="store_true", help="Enable AI analysis in PDF report")

    args = parser.parse_args()

    # Create config from args
    config = PipelineConfig(
        output_base=args.output,
        verbose=not args.quiet,
        use_inmemory_detection=args.inmemory,
        use_trt=args.trt,
        trt_engine_path=args.trt_engine,
        enable_ai_analysis=args.ai_analysis,
    )

    # List jobs mode
    if args.list:
        logbook = PipelineLogbook(config.output_base)
        jobs = logbook.list_jobs()
        if not jobs:
            print("No jobs in logbook")
        else:
            print(f"{'Status':<12} {'Sample':<12} {'Started':<20} {'Input File'}")
            print("-" * 80)
            for job in jobs:
                sample_id = PipelineLogbook.extract_sample_id(job.get("input_path", ""))
                status = job.get("status", "unknown")
                started = job.get("started_at", "")
                input_path = job.get("input_path", "")
                print(f"{status:<12} {sample_id:<12} {started:<20} {input_path}")
        return

    # Determine input files
    if args.input_file:
        # Single file mode
        input_files = [args.input_file]
    else:
        # Batch mode from files.txt
        try:
            with open("files.txt") as f:
                input_files = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        except FileNotFoundError:
            print("Error: No input file specified and files.txt not found")
            print("Usage: python main.py <input_file> or create files.txt")
            return

    print(f"Found {len(input_files)} file(s) to process")

    # Process each file
    for input_file in input_files:
        print(f"\n{'=' * 60}")
        print(f"Processing: {input_file}")
        print(f"{'=' * 60}")

        result = process_single_file(input_file, config=config, force=args.force)

        if result["status"] == "skipped":
            print(f"Skipped: {result['reason']}")
        elif result["status"] == "completed":
            print(f"Completed: output at {result['output_dir']}")
        else:
            print(f"Failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
