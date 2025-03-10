"""
This module provides the reader functionality for AVI files in the napari plugin.
It implements the napari_get_reader interface so that napari can automatically call
this plugin when an AVI file or a directory containing AVI files is dropped in.
"""

import logging
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def napari_get_reader(path):
    """
    Returns a reader function if the path is a valid AVI file or a directory with AVI files.
    """
    if os.path.isdir(path):
        if any(f.lower().endswith(".avi") for f in os.listdir(path)):
            return reader_directory_function
    elif (isinstance(path, str) and path.lower().endswith(".avi")) or (
        isinstance(path, list)
        and all(isinstance(p, str) and p.lower().endswith(".avi") for p in path)
    ):
        return reader_function
    return None


def reader_function(path):
    """
    Read a single AVI file and return a list of LayerData tuples.
    """
    if isinstance(path, list) and len(path) == 1:
        path = path[0]
    if isinstance(path, list):
        return reader_directory_function(os.path.dirname(path[0]), filenames=path)

    video = read_video(path)
    if video is None:
        return []

    first_frame = video[0]
    gray_first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)
    masks, labeled_frame = detect_circles_and_create_masks(gray_first_frame)

    layers = []
    layers.append(
        (labeled_frame, {"name": f"{os.path.basename(path)} - ROIs"}, "image")
    )
    layers.append(
        (
            video,
            {"name": os.path.basename(path), "channel_axis": None, "rgb": True},
            "image",
        )
    )
    for i, mask in enumerate(masks):
        layers.append((mask, {"name": f"ROI {i+1} Mask", "visible": False}, "labels"))
    return layers


def reader_directory_function(path, filenames=None):
    """
    Read a directory of AVI files.
    """
    if filenames is None:
        filenames = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.lower().endswith(".avi")
        ]
    elif isinstance(filenames, list):
        pass
    else:
        return []

    if not filenames:
        return []

    layers = []
    first_video_path = filenames[0]
    first_frame = get_first_frame(first_video_path)
    if first_frame is None:
        return []

    gray_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)
    masks, labeled_frame = detect_circles_and_create_masks(gray_frame)

    layers.append(
        (
            labeled_frame,
            {"name": "Detected ROIs", "metadata": {"path": first_video_path}},
            "image",
        )
    )

    for filename in filenames:
        video = read_video(filename)
        if video is None:
            continue
        layers.append(
            (
                video,
                {"name": os.path.basename(filename), "channel_axis": None, "rgb": True},
                "image",
            )
        )

    for i, mask in enumerate(masks):
        layers.append((mask, {"name": f"ROI {i+1} Mask", "visible": False}, "labels"))
    return layers


def read_video(path):
    """
    Read a video file and return it as a numpy array.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        logger.error(f"Failed to open {path}")
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        return None
    return np.stack(frames)


def get_first_frame(path):
    """
    Get the first frame of a video file.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        logger.error(f"Failed to open {path}")
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def detect_circles_and_create_masks(gray_frame, min_radius=100, max_radius=125):
    """
    Detect circles in the first frame with enhanced contrast and create labeled masks for each ROI.

    Parameters
    ----------
    gray_frame : ndarray
        Grayscale image
    min_radius : int, optional
        Minimum radius for circle detection
    max_radius : int, optional
        Maximum radius for circle detection

    Returns
    -------
    masks : list of ndarray
        Binary masks for each detected ROI
    labeled_frame : ndarray
        Original frame with detected ROIs labeled
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_frame = clahe.apply(gray_frame)
    circles = cv2.HoughCircles(
        enhanced_frame,
        cv2.HOUGH_GRADIENT,
        dp=0.5,
        minDist=250,
        param1=80,
        param2=70,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    masks = []
    labeled_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for idx, circle in enumerate(circles[0, :]):
            mask = np.zeros(gray_frame.shape, dtype=np.uint8)
            cv2.circle(mask, (circle[0], circle[1]), circle[2], 255, thickness=-1)
            masks.append(mask)
            cv2.circle(labeled_frame, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
            cv2.putText(
                labeled_frame,
                f"{idx + 1}",
                (circle[0] - 10, circle[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
    return masks, labeled_frame


def process_video(file_path, masks, fps=5, block_length=5, skip_seconds=5):
    """
    Process a single video file and return ROI intensity changes with timestamps.

    Parameters
    ----------
    file_path : str
        Path to the video file
    masks : list of ndarray
        Binary masks for each ROI
    fps : int, optional
        Frames per second
    block_length : int, optional
        Number of frames to average
    skip_seconds : int, optional
        Number of seconds to skip between blocks

    Returns
    -------
    file_path : str
        Path to the processed video
    roi_changes : dict
        Dictionary mapping ROI index to list of (time, intensity_change) tuples
    video_duration : float
        Duration of the video in seconds
    """
    logger.info(f"Processing {file_path}...")
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        logger.error(f"Failed to open {file_path}")
        return file_path, None, 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps
    roi_changes = {roi_idx + 1: [] for roi_idx in range(len(masks))}
    frame_idx = 0
    prev_avg_frame = None
    prev_time = None
    while True:
        block_frames = []
        for _ in range(block_length):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            block_frames.append(gray)
            frame_idx += 1
        if len(block_frames) < block_length:
            break
        avg_frame = np.mean(block_frames, axis=0).astype(np.uint8)
        block_start_idx = frame_idx - block_length
        block_time_sec = block_start_idx / fps
        if prev_avg_frame is not None:
            diff_frame = cv2.absdiff(avg_frame, prev_avg_frame)
            diff_time_sec = 0.5 * (prev_time + block_time_sec)
            for roi_idx, mask in enumerate(masks, start=1):
                roi_diff = cv2.bitwise_and(diff_frame, diff_frame, mask=mask)
                total_intensity_change = np.sum(roi_diff)
                roi_changes[roi_idx].append((diff_time_sec, total_intensity_change))
        prev_avg_frame = avg_frame
        prev_time = block_time_sec
        skip_frames = skip_seconds * fps
        for _ in range(skip_frames):
            ret, _ = cap.read()
            if not ret:
                break
            frame_idx += 1
        if not ret:
            break
    cap.release()
    logger.info(f"Processed {file_path}, detected {len(masks)} ROIs.")
    return file_path, roi_changes, video_duration


def process_videos(
    directory, fps=5, block_length=5, skip_seconds=5, min_radius=100, max_radius=125
):
    """
    Process all .avi files in the specified directory using the same ROI masks.

    Parameters
    ----------
    directory : str
        Path to directory containing AVI files
    fps : int, optional
        Frames per second
    block_length : int, optional
        Number of frames to average
    skip_seconds : int, optional
        Number of seconds to skip between blocks
    min_radius : int, optional
        Minimum radius for circle detection
    max_radius : int, optional
        Maximum radius for circle detection

    Returns
    -------
    results : dict
        Dictionary mapping file paths to ROI changes
    durations : dict
        Dictionary mapping file paths to video durations
    masks : list of ndarray
        Binary masks for each ROI
    labeled_frame : ndarray
        Original frame with detected ROIs labeled
    """
    avi_files = [
        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".avi")
    ]
    if not avi_files:
        logger.warning("No AVI files found in the directory.")
        return {}, {}, [], None
    first_video = avi_files[0]
    logger.info(f"Using {first_video} to detect ROIs...")
    cap = cv2.VideoCapture(first_video)
    if not cap.isOpened():
        logger.error(f"Failed to open {first_video}")
        return {}, {}, [], None
    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        logger.error(f"Failed to read the first frame of {first_video}")
        return {}, {}, [], None
    gray_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    masks, labeled_frame = detect_circles_and_create_masks(
        gray_frame, min_radius, max_radius
    )
    if not masks:
        logger.warning(f"No ROIs detected in the first frame of {first_video}")
        return {}, {}, [], None
    results = {}
    durations = {}
    with ProcessPoolExecutor() as executor:
        future_map = {
            executor.submit(
                process_video, file_path, masks, fps, block_length, skip_seconds
            ): file_path
            for file_path in avi_files
        }
        for future, file_path in future_map.items():
            try:
                file_path, roi_changes, total_duration = future.result()
                if roi_changes is not None:
                    results[file_path] = roi_changes
                    durations[file_path] = total_duration
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
    return results, durations, masks, labeled_frame


def merge_results(results, durations):
    """
    Merge the results of all videos into a single continuous time series.
    """
    merged_results = {}
    cumulative_time = 0.0
    sorted_paths = sorted(results.keys())
    for path in sorted_paths:
        roi_changes = results[path]
        for roi, time_val_pairs in roi_changes.items():
            if roi not in merged_results:
                merged_results[roi] = []
            for t_sec, val in time_val_pairs:
                merged_results[roi].append((t_sec + cumulative_time, val))
        cumulative_time += durations[path]
    return merged_results


def get_roi_colors(rois):
    """
    Return a dict mapping ROI numbers to colors from matplotlib's default color cycle.
    """

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    roi_colors = {}
    for i, roi in enumerate(rois):
        roi_colors[roi] = color_cycle[i % len(color_cycle)]
    return roi_colors


def compute_roi_thresholds(merged_results, threshold_block_count=10):
    """
    Compute thresholds for each ROI based on the average of the first N blocks.
    """
    roi_thresholds = {}
    for roi, data in merged_results.items():
        sorted_data = sorted(data, key=lambda x: x[0])
        changes = [val for (_, val) in sorted_data]
        if len(changes) >= threshold_block_count:
            roi_thresholds[roi] = float(np.mean(changes[:threshold_block_count]))
        else:
            roi_thresholds[roi] = 0.0
    return roi_thresholds


def define_movement_events_with_noise(
    merged_results, roi_thresholds, noise_factor=0.01
):
    """
    Define movement events based on thresholds with added noise factor.
    """
    movement_data = {}
    for roi, data in merged_results.items():
        sorted_data = sorted(data, key=lambda x: x[0])
        thr = roi_thresholds.get(roi, 0.0)
        adjusted_threshold = thr - (thr * noise_factor)
        movement_states = []
        for t_sec, val in sorted_data:
            state = 1 if val >= adjusted_threshold else 0
            movement_states.append((t_sec, state))
        movement_data[roi] = movement_states
    return movement_data


def bin_fraction_movement(movement_data, bin_size=60):
    """
    Calculate fraction of time spent moving in each time bin.
    """
    fraction_data = {}
    for roi, data in movement_data.items():
        sorted_data = sorted(data, key=lambda x: x[0])
        if not sorted_data:
            fraction_data[roi] = []
            continue
        min_t = sorted_data[0][0]
        max_t = sorted_data[-1][0]
        bin_edges = np.arange(min_t, max_t + bin_size, bin_size)
        binned = []
        for start_t in bin_edges[:-1]:
            end_t = start_t + bin_size
            bin_data = [(t, s) for t, s in sorted_data if start_t <= t < end_t]
            if bin_data:
                total_movement = sum(state for _, state in bin_data)
                frac = total_movement / len(bin_data)
            else:
                frac = 0.0
            mid_t = start_t + bin_size / 2
            binned.append((mid_t, frac))
        fraction_data[roi] = binned
    return fraction_data


def bin_quiescence(movement_data, bin_size=60, move_threshold=0.5):
    """
    Quiescence logic: 1 = Quiescent, 0 = Active.
    """
    quiescence_data = {}
    for roi, data in movement_data.items():
        sorted_data = sorted(data, key=lambda x: x[0])
        if not sorted_data:
            quiescence_data[roi] = []
            continue
        min_t = sorted_data[0][0]
        max_t = sorted_data[-1][0]
        bin_edges = np.arange(min_t, max_t + bin_size, bin_size)
        binned = []
        for start_t in bin_edges[:-1]:
            end_t = start_t + bin_size
            bin_data = [(t, s) for t, s in sorted_data if start_t <= t < end_t]
            if bin_data:
                frac_move = sum(state for _, state in bin_data) / len(bin_data)
                quiescent_state = 0 if frac_move >= move_threshold else 1
            else:
                quiescent_state = 1
            mid_t = start_t + bin_size / 2
            binned.append((mid_t, quiescent_state))
        quiescence_data[roi] = binned
    return quiescence_data


def define_sleep(quiescence_data, consecutive_quiescence=480, bin_size=60):
    """
    Sleep detection: need 'consecutive_quiescence' seconds of quiescent (1) bins.
    """
    needed = consecutive_quiescence // bin_size
    sleep_data = {}
    for roi, data in quiescence_data.items():
        if not data:
            sleep_data[roi] = []
            continue
        times, states = zip(*data)
        arr = np.array(states, dtype=int)
        out = np.zeros(len(arr), dtype=int)
        run_len = 0
        for i in range(len(arr)):
            if arr[i] == 1:
                run_len += 1
            else:
                run_len = 0
            if run_len >= needed:
                for j in range(i - needed + 1, i + 1):
                    out[j] = 1
        sleep_data[roi] = list(zip(times, out))
    return sleep_data
