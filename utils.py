from __future__ import annotations

"""
Pipeline utility classes and functions for 3D-IntelliScan.

Contains:
- PipelineLogger: Execution logging with verbose flag and file output
- PipelineMetrics: Timing and count metrics collection for pipeline phases
- PhaseRecord: Individual phase timing record
- ProcessingStatus: Track processing status for input files
- create_folder_structure: Setup output folder hierarchy
- nii2jpg: Convert 3D NIfTI to 2D JPEG slices
"""

import hashlib
import json
import logging
import os
import time
from contextlib import contextmanager
from enum import Enum
from pathlib import Path

from PIL import Image


class LogLevel(Enum):
    """Log levels for pipeline logging."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    PHASE = "PHASE"


class PipelineLogger:
    """Centralized logging for pipeline execution (tee-style logging).

    All messages are written to the log file for comprehensive execution tracing.
    Console output is controlled by verbosity level:
    - verbose=True: Show INFO and above on console
    - verbose=False: Show only PHASE, WARN, ERROR on console (essential metrics)

    Example:
        >>> logger = PipelineLogger(log_file="output/execution.log", verbose=True)
        >>> logger.info("Starting pipeline")
        >>> with logger.section("Processing bboxes"):
        ...     logger.info("Processing bbox 0")
        >>> logger.close()
    """

    # Singleton instance for global access
    _instance = None

    # Console output levels by verbosity setting
    # verbose=True: show INFO and above
    # verbose=False: show only essential (PHASE, WARN, ERROR)
    CONSOLE_LEVELS_VERBOSE = {LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARN, LogLevel.ERROR, LogLevel.PHASE}
    CONSOLE_LEVELS_QUIET = {LogLevel.WARN, LogLevel.ERROR, LogLevel.PHASE}

    def __init__(self, log_file: str | Path | None = None, verbose: bool = True):
        """Initialize logger.

        Args:
            log_file: Path to log file. If None, only prints to console when verbose.
            verbose: If True, show INFO-level messages on console.
                     If False, only show PHASE/WARN/ERROR on console.
                     All messages are always written to the log file.
        """
        self.verbose = verbose
        self.log_file = Path(log_file) if log_file else None
        self._file_handle = None
        self._indent_level = 0
        self._start_time = time.time()
        self._console_levels = self.CONSOLE_LEVELS_VERBOSE if verbose else self.CONSOLE_LEVELS_QUIET

        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self._file_handle = open(self.log_file, "w", encoding="utf-8")
            self._write_header()

        # Set as singleton
        PipelineLogger._instance = self

    def _write_header(self):
        """Write log file header."""
        if self._file_handle:
            self._file_handle.write("=" * 70 + "\n")
            self._file_handle.write("3D-IntelliScan Pipeline Execution Log\n")
            self._file_handle.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            self._file_handle.write("=" * 70 + "\n\n")
            self._file_handle.flush()

    def _format_message(self, level: LogLevel, message: str) -> str:
        """Format log message with timestamp, level, and indentation."""
        timestamp = time.strftime("%H:%M:%S")
        indent = "  " * self._indent_level
        return f"[{timestamp}] [{level.value}] {indent}{message}"

    def _log(self, level: LogLevel, message: str):
        """Internal logging method - always writes to file, conditionally to console."""
        formatted = self._format_message(level, message)

        # Always write to file (tee-style: comprehensive log)
        if self._file_handle:
            self._file_handle.write(formatted + "\n")
            self._file_handle.flush()

        # Print to console based on level and verbosity
        if level in self._console_levels:
            print(formatted)

    def debug(self, message: str):
        """Log debug message (file always, console if verbose)."""
        self._log(LogLevel.DEBUG, message)

    def info(self, message: str):
        """Log info message (file always, console if verbose)."""
        self._log(LogLevel.INFO, message)

    def warning(self, message: str):
        """Log warning message (always to file and console)."""
        self._log(LogLevel.WARN, message)

    def error(self, message: str):
        """Log error message (always to file and console)."""
        self._log(LogLevel.ERROR, message)

    def phase(self, message: str):
        """Log phase transition (always to file and console)."""
        self._log(LogLevel.PHASE, message)

    def metric(self, message: str):
        """Log essential metric (same as phase - always shown)."""
        self._log(LogLevel.PHASE, message)

    @contextmanager
    def section(self, name: str):
        """Context manager for indented log sections."""
        self.debug(f">>> {name}")
        self._indent_level += 1
        try:
            yield
        finally:
            self._indent_level -= 1
            self.debug(f"<<< {name}")

    def close(self):
        """Close log file and write footer."""
        if self._file_handle:
            elapsed = time.time() - self._start_time
            self._file_handle.write("\n" + "=" * 70 + "\n")
            self._file_handle.write(f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            self._file_handle.write(f"Total elapsed: {elapsed:.2f} seconds\n")
            self._file_handle.write("=" * 70 + "\n")
            self._file_handle.close()
            self._file_handle = None

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()
        return False

    @classmethod
    def get_instance(cls) -> "PipelineLogger | None":
        """Get the singleton logger instance."""
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (useful for testing or re-initialization)."""
        cls._instance = None


def log(message: str, level: str = "info"):
    """Convenience function for logging. Uses singleton logger if available, else prints.

    Args:
        message: Message to log
        level: Log level (info, debug, warning, error, phase)
    """
    logger = PipelineLogger.get_instance()
    if logger:
        getattr(logger, level, logger.info)(message)
    else:
        print(message)


class PhaseRecord:
    """Record for a single pipeline phase execution."""

    def __init__(self, name: str, start_time: float):
        self.name = name
        self.start_time = start_time
        self.end_time = None
        self.elapsed = None
        self.count = None

    def complete(self, count: int | None = None):
        """Mark this phase as complete with optional item count."""
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        self.count = count

    @property
    def avg_time_per_item(self) -> float | None:
        """Return average time per item if count is set and > 0."""
        if self.count is not None and self.count > 0 and self.elapsed is not None:
            return self.elapsed / self.count
        return None

    def to_dict(self) -> dict:
        """Convert record to dictionary for serialization."""
        return {
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "elapsed_seconds": self.elapsed,
            "count": self.count,
            "avg_seconds_per_item": self.avg_time_per_item,
        }

    def __repr__(self):
        if self.count is not None and self.avg_time_per_item is not None:
            return f"[{self.name}] {self.elapsed:.2f}s (count: {self.count}, avg: {self.avg_time_per_item:.4f}s/item)"
        return f"[{self.name}] {self.elapsed:.2f}s"


class PipelineMetrics:
    """Collects timing and count metrics for pipeline execution.

    Each task can have its own PipelineMetrics instance to track
    per-phase timing and quantities for analysis and reporting.

    Example:
        >>> metrics = PipelineMetrics(task_id="sample_001")
        >>> with metrics.phase("NII to JPG") as p:
        ...     slices = convert_nii_to_jpg(...)
        ...     p.complete(count=len(slices))
        >>> metrics.save("output/timing.json")
    """

    def __init__(self, task_id: str | None = None):
        self.task_id = task_id
        self.created_at = time.time()
        self.phases: list[PhaseRecord] = []
        self._current_phase: PhaseRecord | None = None

    @contextmanager
    def phase(self, name: str):
        """Context manager for timing a pipeline phase.

        Args:
            name: Descriptive name for the phase

        Yields:
            PhaseRecord: Call record.complete(count=N) to register quantity
        """
        record = PhaseRecord(name, time.time())
        self._current_phase = record
        try:
            yield record
        finally:
            # Auto-complete if not already done (allows phases without count)
            if record.end_time is None:
                record.complete()
            self.phases.append(record)
            self._current_phase = None
            log(str(record), level="phase")

    def summary(self) -> dict:
        """Return summary of all phases."""
        return {
            "task_id": self.task_id,
            "created_at": self.created_at,
            "total_elapsed": sum(p.elapsed for p in self.phases if p.elapsed),
            "phases": [p.to_dict() for p in self.phases],
        }

    def save(self, filepath: str):
        """Save metrics to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2)

    def write_log(self, filepath: str):
        """Write human-readable log file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"=== Pipeline Metrics for {self.task_id} ===\n")
            f.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.created_at))}\n")
            f.write("=" * 50 + "\n\n")
            for phase in self.phases:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(phase.start_time))
                f.write(f"{timestamp} - {phase}\n")
            f.write("\n" + "=" * 50 + "\n")
            total = sum(p.elapsed for p in self.phases if p.elapsed)
            f.write(f"Total elapsed: {total:.2f} seconds\n")


def create_folder_structure(base_folder, views):
    """
    Create and organize folder structure for processing pipeline.

    With single detection model, we no longer need per-class detection folders.
    Structure:
        base_folder/
            view1/
                input_images/  (2D slices for detection)
                detections/    (detection results)
            view2/
                input_images/
                detections/

    Args:
        base_folder: Base output folder path
        views: List of view names (e.g., ['view1', 'view2'])

    Returns:
        Dictionary with folder paths for each view
    """
    folders = {}
    os.makedirs(base_folder, exist_ok=True)
    log(f"Created base folder: {base_folder}", level="debug")

    for view in views:
        view_path = os.path.join(base_folder, view)
        folders[view] = {
            "input_images": os.path.join(view_path, "input_images"),
            "detections": os.path.join(view_path, "detections"),
        }

        # Create all folders for this view
        for folder_type, folder_path in folders[view].items():
            os.makedirs(folder_path, exist_ok=True)
            log(f"Created {folder_type} folder for {view}: {folder_path}", level="debug")

    return folders


def save_image(array, data_max, filename):
    """Convert to 8-bit, rotate, flip, and save as JPEG
    Normalize using the pre-computed maximum value and convert to 8-bit

    This function normalizes the input array using the pre-computed maximum value,
    converts it to an 8-bit image, rotates the image 90 degrees, flips it vertically,
    and saves it as a JPEG file.

    @param array The input array.
    @param data_max The maximum value used for normalization.
    @param filename The name of the file to save the image as, including the .jpeg extension.

    @details
    The function performs the following steps:
    - Normalizes the input array by dividing it by the pre-computed maximum value and
      scaling it to the 0-255 range.
    - Converts the normalized array to an 8-bit unsigned integer type.
    - Rotates the image 90 degrees clockwise.
    - Flips the image vertically (top to bottom).
    - Converts the image to RGB mode.
    - Saves the image as a JPEG file with the specified filename.

    @exception IOError Raised if the image cannot be saved to the specified filename.
    """
    array = (array / data_max) * 255

    img = Image.fromarray(array).rotate(90, expand=True).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    img = img.convert("RGB")
    img.save(filename)


def nii2jpg(data, outputdir, view, data_max=None):
    """Convert a 3D numpy array to a series of 2D JPEG images based on specified view.

    This function processes 3D medical imaging data, extracts slices based
    on the specified view (axial or coronal), normalizes each slice,
    and saves them as JPEG files in the given output directory.

    @param data 3D numpy array containing the volume data.
    @param outputdir The directory where the output JPEG images will be saved.
    @param view View index specifying the slicing direction:
        - 0: Axial (XY plane, slices along Z)
        - 1: Coronal (XZ plane, slices along Y)
    @param data_max Optional pre-computed max value for normalization.
        If None, will be computed from data.

    @return Number of slices saved.

    @exception IOError Raised if output images cannot be saved.
    """
    # Compute max for normalization if not provided
    if data_max is None:
        data_max = data.max()

    if data_max == 0:
        raise ValueError("The input data contains only zero values, normalization not possible.")

    # Determine slicing direction based on the view
    if view == 0:  # Axial view (slices along Z-axis)
        slice_dim = 2
        slice_func = lambda i: data[:, :, i]
    elif view == 1:  # Coronal view (slices along Y-axis)
        slice_dim = 1
        slice_func = lambda i: data[:, i, :]
    else:
        raise ValueError(f"Invalid view parameter: {view}. Must be 0 (axial) or 1 (coronal).")

    # Create output directory if it doesn't exist
    os.makedirs(outputdir, exist_ok=True)

    # Process and save each slice
    num_slices = data.shape[slice_dim]
    for i in range(num_slices):
        slice_data = slice_func(i)
        filename = os.path.join(outputdir, f"image{i}.jpg")
        if not os.path.exists(filename):
            save_image(slice_data, data_max, filename)

    return num_slices


# ============================================================================
# Central Pipeline Logbook
# ============================================================================


class PipelineLogbook:
    """Central logbook for tracking all pipeline jobs.

    Maintains a single JSON file that records:
    - All submitted jobs with their status (pending, in_progress, completed, failed)
    - Input file hashes for change detection
    - Output locations for each job
    - Timestamps for job lifecycle

    Uses file locking to support concurrent job submissions.

    Example:
        >>> logbook = PipelineLogbook("output")
        >>> should_run, reason = logbook.should_process("/path/to/input.nii")
        >>> if should_run:
        ...     logbook.mark_started("/path/to/input.nii", output_dir)
        ...     # ... process ...
        ...     logbook.mark_completed("/path/to/input.nii")
    """

    LOGBOOK_FILE = ".pipeline_logbook.json"
    LOCK_FILE = ".pipeline_logbook.lock"

    def __init__(self, output_base: str | Path):
        """Initialize logbook.

        Args:
            output_base: Base output directory where logbook is stored
        """
        self.output_base = Path(output_base)
        self.logbook_path = self.output_base / self.LOGBOOK_FILE
        self.lock_path = self.output_base / self.LOCK_FILE

    def _acquire_lock(self, timeout: float = 10.0) -> bool:
        """Acquire file lock for concurrent access.

        Args:
            timeout: Maximum seconds to wait for lock

        Returns:
            True if lock acquired, False if timeout
        """
        import time as _time

        self.output_base.mkdir(parents=True, exist_ok=True)
        start = _time.time()
        while _time.time() - start < timeout:
            try:
                # Create lock file exclusively
                fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return True
            except FileExistsError:
                _time.sleep(0.1)
        return False

    def _release_lock(self):
        """Release file lock."""
        import contextlib

        with contextlib.suppress(FileNotFoundError):
            self.lock_path.unlink()

    def _load(self) -> dict:
        """Load logbook from disk."""
        if self.logbook_path.exists():
            try:
                with open(self.logbook_path, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {"jobs": {}}
        return {"jobs": {}}

    def _save(self, data: dict):
        """Save logbook to disk."""
        self.output_base.mkdir(parents=True, exist_ok=True)
        with open(self.logbook_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def compute_file_hash(filepath: str | Path, chunk_size: int = 65536) -> str:
        """Compute hash of file for change detection.

        Uses MD5 of file size + first/last chunks for efficiency on large files.

        Args:
            filepath: Path to file
            chunk_size: Size of chunks to read (64KB default)

        Returns:
            Hex digest of hash
        """
        filepath = Path(filepath)
        hasher = hashlib.md5()

        file_size = filepath.stat().st_size
        hasher.update(str(file_size).encode())

        with open(filepath, "rb") as f:
            # Hash first chunk
            hasher.update(f.read(chunk_size))
            # Hash last chunk if file is large enough
            if file_size > chunk_size * 2:
                f.seek(-chunk_size, 2)
                hasher.update(f.read(chunk_size))

        return hasher.hexdigest()

    @staticmethod
    def extract_sample_id(input_path: str | Path) -> str:
        """Extract sample ID (SNxxx) from input path.

        Expects paths like: .../SNxxx_3D_MonYY/filename.nii
        Extracts just the SNxxx part (e.g., "SN002", "SN009").

        Args:
            input_path: Full path to input file

        Returns:
            Sample ID like "SN002"
        """
        import re

        p = Path(input_path)
        # Parent folder should contain the sample ID (e.g., SN002_3D_Feb24)
        parent_name = p.parent.name

        # Extract SNxxx pattern
        match = re.match(r"(SN\d+)", parent_name)
        if match:
            return match.group(1)

        # Fallback: use full parent name if no SNxxx pattern found
        return parent_name

    def get_output_dir(self, input_path: str | Path) -> Path:
        """Get the output directory for an input file.

        Args:
            input_path: Path to input file

        Returns:
            Output directory path (e.g., output/SN002)
        """
        sample_id = self.extract_sample_id(input_path)
        return self.output_base / sample_id

    def _check_in_progress_status(self, job: dict) -> tuple[bool, str]:
        """Check if an in-progress job is stale."""
        from datetime import datetime

        started = job.get("started_at", "")
        if started:
            try:
                start_time = datetime.strptime(started, "%Y-%m-%d %H:%M:%S")
                if (datetime.now() - start_time).total_seconds() > 86400:
                    return True, "previous processing stale (>24h)"
            except ValueError:
                pass
        return False, "processing in progress"

    def _check_completed_status(self, job: dict, input_path: Path) -> tuple[bool, str]:
        """Check if a completed job needs reprocessing due to input changes."""
        current_hash = self.compute_file_hash(input_path)
        if job.get("input_hash") != current_hash:
            return True, "input file changed since last processing"
        return False, "already processed (use force=True to rerun)"

    def should_process(self, input_path: str | Path, force: bool = False) -> tuple[bool, str]:
        """Check if a file should be processed.

        Args:
            input_path: Path to input file
            force: If True, always return True

        Returns:
            Tuple of (should_process, reason)
        """
        if force:
            return True, "forced reprocessing"

        input_path = Path(input_path)
        if not input_path.exists():
            return False, f"input file not found: {input_path}"

        if not self._acquire_lock():
            return False, "could not acquire logbook lock"

        try:
            data = self._load()
            job = data.get("jobs", {}).get(str(input_path.resolve()))

            if job is None:
                return True, "not previously processed"

            status = job.get("status")
            if status == "in_progress":
                return self._check_in_progress_status(job)
            if status == "failed":
                return True, "previous processing failed"
            if status == "completed":
                return self._check_completed_status(job, input_path)
            return True, "unknown status"

        finally:
            self._release_lock()

    def mark_started(self, input_path: str | Path, output_dir: str | Path, config: dict | None = None):
        """Mark a job as started.

        Args:
            input_path: Path to input file
            output_dir: Output directory for this job
            config: Optional pipeline configuration
        """
        input_path = Path(input_path)
        key = str(input_path.resolve())

        if not self._acquire_lock():
            raise RuntimeError("Could not acquire logbook lock")

        try:
            data = self._load()
            data.setdefault("jobs", {})[key] = {
                "input_path": str(input_path),
                "input_hash": self.compute_file_hash(input_path) if input_path.exists() else None,
                "output_dir": str(output_dir),
                "status": "in_progress",
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "completed_at": None,
                "config": config or {},
            }
            self._save(data)
        finally:
            self._release_lock()

    def mark_completed(self, input_path: str | Path, metrics: dict | None = None):
        """Mark a job as completed.

        Args:
            input_path: Path to input file
            metrics: Optional metrics from processing
        """
        key = str(Path(input_path).resolve())

        if not self._acquire_lock():
            raise RuntimeError("Could not acquire logbook lock")

        try:
            data = self._load()
            if key in data.get("jobs", {}):
                data["jobs"][key]["status"] = "completed"
                data["jobs"][key]["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                if metrics:
                    data["jobs"][key]["metrics"] = metrics
                self._save(data)
        finally:
            self._release_lock()

    def mark_failed(self, input_path: str | Path, error: str):
        """Mark a job as failed.

        Args:
            input_path: Path to input file
            error: Error message
        """
        key = str(Path(input_path).resolve())

        if not self._acquire_lock():
            raise RuntimeError("Could not acquire logbook lock")

        try:
            data = self._load()
            if key in data.get("jobs", {}):
                data["jobs"][key]["status"] = "failed"
                data["jobs"][key]["failed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                data["jobs"][key]["error"] = error
                self._save(data)
        finally:
            self._release_lock()

    def get_job(self, input_path: str | Path) -> dict | None:
        """Get job info for an input file.

        Args:
            input_path: Path to input file

        Returns:
            Job dictionary or None if not found
        """
        key = str(Path(input_path).resolve())

        if not self._acquire_lock():
            return None

        try:
            data = self._load()
            return data.get("jobs", {}).get(key)
        finally:
            self._release_lock()

    def list_jobs(self, status: str | None = None) -> list[dict]:
        """List all jobs, optionally filtered by status.

        Args:
            status: Filter by status (in_progress, completed, failed) or None for all

        Returns:
            List of job dictionaries
        """
        if not self._acquire_lock():
            return []

        try:
            data = self._load()
            jobs = list(data.get("jobs", {}).values())
            if status:
                jobs = [j for j in jobs if j.get("status") == status]
            return jobs
        finally:
            self._release_lock()
