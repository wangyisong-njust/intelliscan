#!/usr/bin/env python3
from __future__ import annotations

"""
3D Segmentation Inference Module

Provides inference capabilities using MONAI BasicUNet for 3D semantic segmentation.
Designed for integration with detection + metrology pipelines.
Supports PyTorch and TensorRT backends.

Workflow:
1. Load trained BasicUNet checkpoint or TensorRT engine
2. For each detected 3D bounding box:
   - Expand bbox with margin (handles edge cases)
   - Extract crop from full volume
   - Apply ClipZScoreNormalize preprocessing
   - Run sliding window inference
   - Apply post-processing (void filling)
3. Assemble all predictions into full volume
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import BasicUNet

from utils import log


# =============================================================================
# TensorRT predictor wrapper
# =============================================================================


class TensorRTPredictor:
    """Wraps a TensorRT engine as a callable predictor for sliding_window_inference.

    Accepts PyTorch CUDA tensors and returns PyTorch CUDA tensors.
    Supports two transfer modes via `gpu_direct` parameter:
    - True:    pycuda.autoinit + memcpy_dtod (fast, needs shared CUDA context)
    - False:   manual context + pinned host relay (compatible with limited VRAM)
    - "auto":  try GPU direct first, fall back to host relay if test inference fails
    """

    def __init__(self, engine_path: str | Path, num_classes: int = 5, gpu_direct: bool | str = "auto"):
        import tensorrt as trt
        import pycuda.driver as cuda

        self._trt = trt
        self._cuda = cuda
        self._gpu_direct = None  # set after successful init
        self._ctx = None
        self._h_input = None
        self._h_output = None

        self._engine_path = Path(engine_path)
        log(f"Loading TensorRT engine: {self._engine_path}")

        if gpu_direct == "auto":
            self._init_auto(self._engine_path, num_classes)
        elif gpu_direct:
            self._setup_gpu_direct(self._engine_path, num_classes)
        else:
            self._setup_host_relay(self._engine_path, num_classes)

        mode_str = "GPU direct (dtod)" if self._gpu_direct else "host relay (htod/dtoh)"
        log(f"TensorRT engine loaded successfully, transfer mode: {mode_str}")

    def _load_engine_and_buffers(self, engine_path: Path, num_classes: int):
        """Load TRT engine and allocate I/O device buffers (shared by both modes)."""
        trt = self._trt
        cuda = self._cuda

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.num_classes = num_classes

        self._input_name = None
        self._output_name = None
        self._d_input = None
        self._d_output = None
        self._input_shape = None
        self._output_shape = None

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)
            nbytes = int(np.prod(shape)) * np.dtype(np.float32).itemsize
            if mode == trt.TensorIOMode.INPUT:
                self._input_name = name
                self._input_shape = tuple(shape)
                self._d_input = cuda.mem_alloc(nbytes)
                self.context.set_tensor_address(name, int(self._d_input))
                log(f"  TRT input '{name}': shape={tuple(shape)}")
            else:
                self._output_name = name
                self._output_shape = tuple(shape)
                self._d_output = cuda.mem_alloc(nbytes)
                self.context.set_tensor_address(name, int(self._d_output))
                log(f"  TRT output '{name}': shape={tuple(shape)}")

    def _setup_gpu_direct(self, engine_path: Path, num_classes: int):
        """Initialize GPU direct mode: pycuda.autoinit + dtod transfers."""
        import pycuda.autoinit  # noqa: F401
        self._load_engine_and_buffers(engine_path, num_classes)
        self._gpu_direct = True

    def _setup_host_relay(self, engine_path: Path, num_classes: int):
        """Initialize host relay mode: manual CUDA context + htod/dtoh transfers."""
        cuda = self._cuda
        cuda.init()
        self._ctx = cuda.Device(0).make_context()
        self._load_engine_and_buffers(engine_path, num_classes)
        self._h_input = cuda.pagelocked_empty(
            int(np.prod(self._input_shape)), dtype=np.float32
        )
        self._h_output = cuda.pagelocked_empty(
            int(np.prod(self._output_shape)), dtype=np.float32
        )
        # Pop manual context so PyTorch can use its own; push/pop per call
        self._ctx.pop()
        self._gpu_direct = False

    def _cleanup_resources(self):
        """Release all TRT device resources before re-initialization."""
        if getattr(self, "_d_input", None) is not None:
            self._d_input.free()
            self._d_input = None
        if getattr(self, "_d_output", None) is not None:
            self._d_output.free()
            self._d_output = None
        for attr in ("context", "engine", "stream"):
            if hasattr(self, attr):
                setattr(self, attr, None)
        self._h_input = None
        self._h_output = None
        if self._ctx is not None:
            self._ctx.detach()
            self._ctx = None

    def _init_auto(self, engine_path: Path, num_classes: int):
        """Auto-detect: try GPU direct first, fall back to host relay on failure."""
        log("Auto-detecting TRT transfer mode...")
        try:
            self._setup_gpu_direct(engine_path, num_classes)
            # Validate with a test inference (detect silent failures like all-zero output)
            test_input = torch.ones(self._input_shape, dtype=torch.float32, device="cuda")
            test_output = self._call_gpu_direct(test_input)
            if test_output.abs().sum().item() > 0:
                log("Auto-detect: GPU direct mode validated successfully")
                return
            log("Auto-detect: GPU direct produced all-zero output")
        except Exception as e:
            log(f"Auto-detect: GPU direct failed ({e})")

        # Cleanup GPU direct resources completely before re-initializing
        self._cleanup_resources()
        log("Auto-detect: falling back to host relay mode")
        self._setup_host_relay(engine_path, num_classes)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Run TRT inference on a PyTorch CUDA tensor.

        Args:
            x: Input tensor of shape (B, 1, X, Y, Z) on CUDA

        Returns:
            Output tensor of shape (B, num_classes, X, Y, Z) on CUDA
        """
        if self._gpu_direct:
            return self._call_gpu_direct(x)
        return self._call_host_relay(x)

    def _call_gpu_direct(self, x: torch.Tensor) -> torch.Tensor:
        """GPU-to-GPU transfer via pycuda.autoinit shared context."""
        cuda = self._cuda
        batch_size = x.shape[0]

        outputs = []
        for b in range(batch_size):
            sample = x[b : b + 1].contiguous().float()

            output_tensor = torch.empty(
                self._output_shape, dtype=torch.float32, device=x.device
            )

            input_nbytes = sample.nelement() * sample.element_size()
            output_nbytes = output_tensor.nelement() * output_tensor.element_size()

            cuda.memcpy_dtod_async(
                self._d_input, sample.data_ptr(), input_nbytes, self.stream
            )
            self.context.execute_async_v3(stream_handle=self.stream.handle)
            cuda.memcpy_dtod_async(
                output_tensor.data_ptr(), self._d_output, output_nbytes, self.stream
            )
            self.stream.synchronize()

            outputs.append(output_tensor)

        if batch_size == 1:
            return outputs[0]
        return torch.cat(outputs, dim=0)

    def _call_host_relay(self, x: torch.Tensor) -> torch.Tensor:
        """Host relay via manual CUDA context with push/pop."""
        cuda = self._cuda
        batch_size = x.shape[0]

        outputs = []
        for b in range(batch_size):
            sample = x[b : b + 1].contiguous().float()
            h_in = sample.cpu().numpy().ravel()
            np.copyto(self._h_input, h_in)

            self._ctx.push()

            cuda.memcpy_htod(self._d_input, self._h_input)
            self.context.execute_async_v3(stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self._h_output, self._d_output, self.stream)
            self.stream.synchronize()

            self._ctx.pop()

            output_tensor = torch.from_numpy(
                self._h_output.reshape(self._output_shape).copy()
            ).to(x.device)

            outputs.append(output_tensor)

        if batch_size == 1:
            return outputs[0]
        return torch.cat(outputs, dim=0)

# =============================================================================
# Post-processing utilities
# =============================================================================


def fill_internal_voids(
    prediction: np.ndarray,
    background_class: int = 0,
    void_class: int = 3,
) -> np.ndarray:
    """Fill internal voids (holes) in a 3D segmentation volume.

    Finds all connected regions of background_class that do NOT touch
    the volume boundary and relabels them as void_class.

    Args:
        prediction: 3D integer segmentation array (D, H, W)
        background_class: Label considered as true background
        void_class: Label to assign to internal holes

    Returns:
        Modified prediction array with internal holes relabeled
    """
    # Lazy import - only needed when apply_void_fill is True
    from scipy.ndimage import label as scipy_label

    bg_mask = prediction == background_class
    bg_labels, n_labels = scipy_label(bg_mask)

    # Identify which labels touch the volume boundary (6 faces of cuboid)
    boundary = np.zeros_like(bg_mask, dtype=bool)
    boundary[0, :, :] = True
    boundary[-1, :, :] = True
    boundary[:, 0, :] = True
    boundary[:, -1, :] = True
    boundary[:, :, 0] = True
    boundary[:, :, -1] = True

    external_labels = set(np.unique(bg_labels[boundary & bg_mask]).tolist())
    external_labels.discard(0)

    # Relabel internal (non-boundary-touching) background as void
    for lab in range(1, n_labels + 1):
        if lab not in external_labels:
            prediction[bg_labels == lab] = void_class

    return prediction


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SegmentationConfig:
    """Configuration for segmentation inference pipeline."""

    roi_size: tuple[int, int, int] = (112, 112, 80)
    overlap: float = 0.5
    sw_batch_size: int = 1
    margin: int = 15
    num_classes: int = 5
    apply_normalization: bool = True
    apply_void_fill: bool = False  # Disabled by default for performance
    device: str = "cuda"

    # Normalization parameters
    clip_percentile_low: float = 1.0
    clip_percentile_high: float = 99.0

    # Post-processing parameters
    void_class: int = 3
    background_class: int = 0

    # TensorRT backend
    use_trt: bool = False
    trt_engine_path: str | None = None
    trt_gpu_direct: bool | str = "auto"  # True=dtod, False=host relay, "auto"=try dtod then fallback


# =============================================================================
# Core inference class
# =============================================================================


class SegmentationInference:
    """Segmentation inference engine using MONAI BasicUNet or TensorRT.

    Provides methods for:
    - Single crop inference (PyTorch or TensorRT backend)
    - Multi-bbox batch inference
    - Full volume assembly
    """

    def __init__(
        self,
        model_path: str | Path,
        config: SegmentationConfig | None = None,
    ):
        """Initialize inference engine.

        Args:
            model_path: Path to trained model checkpoint (.ckpt) or TRT engine (.engine)
            config: Configuration object (uses defaults if None)
        """
        self.model_path = Path(model_path)
        self.config = config or SegmentationConfig()
        self.device = torch.device(self.config.device)
        self.model: BasicUNet | None = None
        self._predictor: TensorRTPredictor | BasicUNet | None = None

    def load_model(self) -> None:
        """Load model (PyTorch or TensorRT).

        Supports:
        - TensorRT engine (.engine) when config.use_trt is True
        - Original PyTorch state_dict (.ckpt)
        - Pruned PyTorch checkpoint (dict with 'features' key)
        """
        if self._predictor is not None:
            return  # Already loaded

        if self.config.use_trt:
            engine_path = self.config.trt_engine_path or str(self.model_path)
            self._predictor = TensorRTPredictor(
                engine_path, self.config.num_classes, gpu_direct=self.config.trt_gpu_direct
            )
            log(f"Using TensorRT backend: {engine_path}")
            return

        raw_ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)

        # Detect pruned model format (contains 'features' key)
        if isinstance(raw_ckpt, dict) and "features" in raw_ckpt:
            features = tuple(raw_ckpt["features"])
            self.model = BasicUNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=self.config.num_classes,
                features=features,
            ).to(self.device)
            state_dict = raw_ckpt["state_dict"]
            log(f"Loading pruned model with features={features}")
        else:
            self.model = BasicUNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=self.config.num_classes,
            ).to(self.device)
            state_dict = raw_ckpt
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self._predictor = self.model
        log(f"Loaded segmentation model from: {self.model_path}")

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Apply ClipZScoreNormalize preprocessing (CPU fallback)."""
        cfg = self.config
        flat = data.ravel()
        p_low = np.percentile(flat, cfg.clip_percentile_low)
        p_high = np.percentile(flat, cfg.clip_percentile_high)
        clipped = np.clip(data, p_low, p_high)
        mean, std = clipped.mean(), clipped.std()
        return ((clipped - mean) / (std + 1e-8)).astype(np.float32)

    def normalize_gpu(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply ClipZScoreNormalize on GPU.

        Args:
            tensor: 1-D or N-D float tensor on CUDA

        Returns:
            Normalized tensor (same shape)
        """
        cfg = self.config
        flat = tensor.reshape(-1)
        q_low = cfg.clip_percentile_low / 100.0
        q_high = cfg.clip_percentile_high / 100.0
        p_low = torch.quantile(flat, q_low)
        p_high = torch.quantile(flat, q_high)
        clipped = tensor.clamp(p_low, p_high)
        mean = clipped.mean()
        std = clipped.std()
        return (clipped - mean) / (std + 1e-8)

    def infer_crop(self, crop: np.ndarray) -> np.ndarray:
        """Run inference on a single 3D crop.

        Args:
            crop: 3D numpy array (X, Y, Z) - raw intensity values

        Returns:
            Prediction array (X, Y, Z) with class labels
        """
        if self._predictor is None:
            self.load_model()

        # To GPU tensor first, then normalize on GPU
        tensor = torch.from_numpy(crop.astype(np.float32)).to(self.device)

        if self.config.apply_normalization:
            tensor = self.normalize_gpu(tensor)

        # Shape to [1, 1, X, Y, Z]
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            orig_shape = tensor.shape[2:]
            roi = self.config.roi_size
            fits_in_roi = all(s <= r for s, r in zip(orig_shape, roi))
            if fits_in_roi:
                # Symmetric pad to roi_size (needed for InstanceNorm consistency)
                pad = []
                for dim in reversed(range(3)):
                    diff = roi[dim] - orig_shape[dim]
                    half = diff // 2
                    pad.extend([half, diff - half])
                padded = torch.nn.functional.pad(tensor, pad, mode="constant", value=0.0)
                logits = self._predictor(padded)
                slices = []
                for dim in range(3):
                    diff = roi[dim] - orig_shape[dim]
                    half = diff // 2
                    slices.append(slice(half, half + orig_shape[dim]))
                logits = logits[:, :, slices[0], slices[1], slices[2]]
            else:
                # Oversized: use sliding_window_inference
                logits = sliding_window_inference(
                    inputs=tensor,
                    roi_size=self.config.roi_size,
                    sw_batch_size=self.config.sw_batch_size,
                    predictor=self._predictor,
                    overlap=self.config.overlap,
                    mode="gaussian",
                )

        prediction = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Post-process
        if self.config.apply_void_fill:
            prediction = fill_internal_voids(
                prediction,
                background_class=self.config.background_class,
                void_class=self.config.void_class,
            )

        return prediction


# =============================================================================
# Result dataclass and utility functions
# =============================================================================


@dataclass
class BBoxSegmentationResult:
    """Result from segmenting a single bounding box region."""

    bbox_index: int
    prediction: np.ndarray
    original_bbox: np.ndarray
    expanded_bbox: list[int]
    crop_shape: tuple[int, ...]


def expand_bbox(
    bbox: np.ndarray,
    volume_shape: tuple[int, ...],
    margin: int,
) -> list[int]:
    """Expand bounding box with margin, clamped to volume boundaries.

    Args:
        bbox: [x_min, x_max, y_min, y_max, z_min, z_max]
        volume_shape: Shape of the full volume (X, Y, Z)
        margin: Expansion margin in voxels

    Returns:
        Expanded bbox as [x_min, x_max, y_min, y_max, z_min, z_max]
    """
    return [
        max(int(bbox[0]) - margin, 0),
        min(int(bbox[1]) + margin, volume_shape[0]),
        max(int(bbox[2]) - margin, 0),
        min(int(bbox[3]) + margin, volume_shape[1]),
        max(int(bbox[4]) - margin, 0),
        min(int(bbox[5]) + margin, volume_shape[2]),
    ]


# =============================================================================
# High-level inference functions
# =============================================================================


def segment_bboxes(
    engine: SegmentationInference,
    volume: np.ndarray,
    bboxes: np.ndarray,
    save_dir: Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    batch_size: int = 8,
) -> list[BBoxSegmentationResult]:
    """Segment multiple bounding box regions from a volume.

    Uses batched inference for crops that fit within roi_size to reduce
    per-crop overhead. Falls back to sliding_window_inference for
    crops larger than roi_size.

    Args:
        engine: Initialized SegmentationInference instance
        volume: Full 3D volume array
        bboxes: Array of bboxes [N, 6] format [x_min, x_max, y_min, y_max, z_min, z_max]
        save_dir: Optional directory to save per-bbox crops and predictions.
                  If provided, saves to save_dir/img/ and save_dir/pred/
        progress_callback: Optional callback(current_idx, total) for progress updates
        batch_size: Number of crops to batch together for inference

    Returns:
        List of BBoxSegmentationResult for each bbox
    """
    cfg = engine.config
    engine.load_model()
    device = engine.device
    roi = cfg.roi_size  # (112, 112, 80)

    # Phase 1: extract & normalize all crops, classify into batchable vs oversized
    batch_items = []   # crops that fit in roi_size → batch inference
    large_items = []   # crops larger than roi_size → sliding_window

    for idx, bbox in enumerate(bboxes):
        expanded = expand_bbox(bbox, volume.shape, cfg.margin)
        region_size = (
            expanded[1] - expanded[0],
            expanded[3] - expanded[2],
            expanded[5] - expanded[4],
        )
        if any(s < 5 for s in region_size):
            log(f"Skipping bbox {idx}: region too small {region_size}")
            continue

        crop = volume[
            expanded[0] : expanded[1],
            expanded[2] : expanded[3],
            expanded[4] : expanded[5],
        ]

        # Normalize on GPU
        tensor = torch.from_numpy(crop.astype(np.float32)).to(device)
        if cfg.apply_normalization:
            tensor = engine.normalize_gpu(tensor)

        fits_in_roi = all(s <= r for s, r in zip(region_size, roi))

        item = {
            "idx": idx, "bbox": bbox, "expanded": expanded,
            "crop_shape": crop.shape, "tensor": tensor,
        }

        if fits_in_roi:
            batch_items.append(item)
        else:
            large_items.append(item)

    log(f"Batchable crops: {len(batch_items)}, oversized crops: {len(large_items)}")

    # Phase 2: batched inference for crops that fit in roi_size
    results_dict = {}

    if batch_items:
        # Pad each crop to roi_size and stack into batches
        for start in range(0, len(batch_items), batch_size):
            batch = batch_items[start : start + batch_size]
            padded_tensors = []

            for item in batch:
                t = item["tensor"]  # shape (X, Y, Z)
                # Symmetric pad to roi_size (matches MONAI sliding_window_inference)
                pad = []
                for dim in reversed(range(3)):
                    diff = roi[dim] - t.shape[dim]
                    half = diff // 2
                    pad.extend([half, diff - half])
                padded = torch.nn.functional.pad(t, pad, mode="constant", value=0.0)
                padded_tensors.append(padded.unsqueeze(0).unsqueeze(0))  # [1, 1, X, Y, Z]

            batch_tensor = torch.cat(padded_tensors, dim=0)  # [B, 1, roi...]

            with torch.no_grad():
                batch_logits = engine._predictor(batch_tensor)  # [B, C, roi...]

            for i, item in enumerate(batch):
                logits_i = batch_logits[i : i + 1]  # [1, C, roi...]
                # Symmetric unpad to original crop shape
                s = item["crop_shape"]
                slices = []
                for dim in range(3):
                    diff = roi[dim] - s[dim]
                    half = diff // 2
                    slices.append(slice(half, half + s[dim]))
                logits_cropped = logits_i[:, :, slices[0], slices[1], slices[2]]
                pred = torch.argmax(logits_cropped, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

                if cfg.apply_void_fill:
                    pred = fill_internal_voids(
                        pred, background_class=cfg.background_class,
                        void_class=cfg.void_class,
                    )

                results_dict[item["idx"]] = BBoxSegmentationResult(
                    bbox_index=item["idx"],
                    prediction=pred,
                    original_bbox=item["bbox"],
                    expanded_bbox=item["expanded"],
                    crop_shape=item["crop_shape"],
                )

            if progress_callback:
                progress_callback(min(start + batch_size, len(batch_items)), len(bboxes))

    # Phase 3: sliding_window for oversized crops
    for item in large_items:
        tensor = item["tensor"].unsqueeze(0).unsqueeze(0)  # [1, 1, X, Y, Z]
        with torch.no_grad():
            logits = sliding_window_inference(
                inputs=tensor,
                roi_size=roi,
                sw_batch_size=cfg.sw_batch_size,
                predictor=engine._predictor,
                overlap=cfg.overlap,
                mode="gaussian",
            )
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        if cfg.apply_void_fill:
            pred = fill_internal_voids(
                pred, background_class=cfg.background_class,
                void_class=cfg.void_class,
            )

        results_dict[item["idx"]] = BBoxSegmentationResult(
            bbox_index=item["idx"],
            prediction=pred,
            original_bbox=item["bbox"],
            expanded_bbox=item["expanded"],
            crop_shape=item["crop_shape"],
        )

    # Phase 4: optional save + collect results in original order
    results = []
    for idx in sorted(results_dict.keys()):
        r = results_dict[idx]
        if save_dir is not None:
            crop = volume[
                r.expanded_bbox[0] : r.expanded_bbox[1],
                r.expanded_bbox[2] : r.expanded_bbox[3],
                r.expanded_bbox[4] : r.expanded_bbox[5],
            ]
            img_path = save_dir / "img" / f"img_{idx}.nii.gz"
            pred_path = save_dir / "pred" / f"pred_{idx}.nii.gz"
            nib.save(nib.Nifti1Image(crop.astype(np.float32), np.eye(4)), str(img_path))
            nib.save(nib.Nifti1Image(r.prediction.astype(np.float32), np.eye(4)), str(pred_path))
        results.append(r)

    return results


def assemble_full_volume(
    results: list[BBoxSegmentationResult],
    volume_shape: tuple[int, ...],
) -> np.ndarray:
    """Assemble per-bbox predictions into a full volume.

    Args:
        results: List of BBoxSegmentationResult from segment_bboxes
        volume_shape: Shape of the output volume

    Returns:
        Full volume with predictions placed at their bbox locations
    """
    output = np.zeros(volume_shape, dtype=np.float32)

    for result in results:
        bbox = result.expanded_bbox
        pred = result.prediction

        # Direct placement - prediction shape matches expanded bbox
        output[
            bbox[0] : bbox[0] + pred.shape[0],
            bbox[2] : bbox[2] + pred.shape[1],
            bbox[4] : bbox[4] + pred.shape[2],
        ] = pred

    return output


if __name__ == "__main__":
    log("3D Segmentation Inference Module")
    log("\nUsage:")
    log("  engine = SegmentationInference(model_path, config)")
    log("  results = segment_bboxes(engine, volume, bboxes, save_dir)")
    log("  full_vol = assemble_full_volume(results, volume.shape)")
