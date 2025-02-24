"""
This module provides the reader functionality for AVI files in the napari plugin.
"""

import os
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor


def napari_get_reader(path):
    """
    Returns a reader function if the path is a valid AVI file or directory with AVI files.

    Parameters
    ----------
    path : str
        Path to file or directory

    Returns
    -------
    callable or None
        If path is a recognized format, returns a function that can read it.
        Otherwise, returns None.
    """
    # If we get a directory, check if it contains AVI files
    if os.path.isdir(path):
        if any(f.lower().endswith('.avi') for f in os.listdir(path)):
            return reader_directory_function
    # If we get a file, check if it's an AVI file
    elif isinstance(path, str) and path.lower().endswith('.avi'):
        return reader_function
    # If we get a list of paths
    elif isinstance(path, list):
        if all(isinstance(p, str) and p.lower().endswith('.avi') for p in path):
            return reader_function
    return None


def reader_function(path):
    """
    Read a single AVI file and return a list of LayerData tuples.

    Parameters
    ----------
    path : str or list of str
        Path to the AVI file(s)

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples, where each tuple contains 
        (data, metadata, layer_type)
    """
    # If path is a list with one element, extract it
    if isinstance(path, list) and len(path) == 1:
        path = path[0]
    
    # If we got a list of files, call the directory function
    if isinstance(path, list):
        return reader_directory_function(os.path.dirname(path[0]), filenames=path)
    
    # Read the video
    video = read_video(path)
    if video is None:
        return []
    
    # Extract the first frame for ROI detection
    first_frame = video[0]
    gray_first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)
    
    # Get ROI masks and labeled frame
    masks, labeled_frame = detect_circles_and_create_masks(gray_first_frame)
    
    # Return layers
    layers = []
    
    # Add the first frame with detected ROIs
    layers.append((labeled_frame, 
                  {'name': f"{os.path.basename(path)} - ROIs"}, 
                  'image'))
    
    # Add the entire video as a layer
    layers.append((video, 
                  {'name': os.path.basename(path), 
                   'channel_axis': None,
                   'rgb': True},
                  'image'))
    
    # Return the masks as binary images for potential use
    for i, mask in enumerate(masks):
        layers.append((mask, 
                     {'name': f"ROI {i+1} Mask", 
                      'visible': False},
                     'labels'))
    
    return layers


def reader_directory_function(path, filenames=None):
    """
    Read a directory of AVI files.

    Parameters
    ----------
    path : str
        Path to directory containing AVI files
    filenames : list of str, optional
        List of filenames to read. If not provided, all AVI files in the directory will be read.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples, where each tuple contains 
        (data, metadata, layer_type)
    """
    if filenames is None:
        filenames = [os.path.join(path, f) for f in os.listdir(path) 
                    if f.lower().endswith('.avi')]
    elif isinstance(filenames, list):
        # If filenames are provided, use them directly
        pass
    else:
        return []
    
    # Get the first video for ROI detection
    if not filenames:
        return []
    
    layers = []
    
    # Use the first video to detect ROIs
    first_video_path = filenames[0]
    first_frame = get_first_frame(first_video_path)
    if first_frame is None:
        return []
    
    gray_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)
    masks, labeled_frame = detect_circles_and_create_masks(gray_frame)
    
    # Add the first frame with ROIs
    layers.append((labeled_frame, 
                  {'name': "Detected ROIs", 
                   'metadata': {'path': first_video_path}}, 
                  'image'))
    
    # For each video, add a layer
    for filename in filenames:
        video = read_video(filename)
        if video is None:
            continue
        
        layers.append((video, 
                      {'name': os.path.basename(filename),
                       'channel_axis': None,
                       'rgb': True},
                      'image'))
    
    # Add masks
    for i, mask in enumerate(masks):
        layers.append((mask, 
                     {'name': f"ROI {i+1} Mask", 
                      'visible': False},
                     'labels'))
    
    return layers


def read_video(path):
    """
    Read a video file and return it as a numpy array.

    Parameters
    ----------
    path : str
        Path to the video file

    Returns
    -------
    video : ndarray or None
        Video as a numpy array of shape (t, y, x, 3) if successful, None otherwise
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Failed to open {path}")
        return None
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    cap.release()
    
    if not frames:
        return None
    
    return np.stack(frames)


def get_first_frame(path):
    """
    Get the first frame of a video file.

    Parameters
    ----------
    path : str
        Path to the video file

    Returns
    -------
    frame : ndarray or None
        First frame as a numpy array if successful, None otherwise
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Failed to open {path}")
        return None
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb


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
        maxRadius=max_radius
    )

    masks = []
    labeled_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for idx, circle in enumerate(circles[0, :]):
            mask = np.zeros(gray_frame.shape, dtype=np.uint8)
            cv2.circle(mask, (circle[0], circle[1]), circle[2], 255, thickness=-1)
            masks.append(mask)

            # Label the detected ROI on the visualization frame
            cv2.circle(labeled_frame, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
            cv2.putText(
                labeled_frame,
                f"{idx + 1}",
                (circle[0] - 10, circle[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )
    return masks, labeled_frame


# Video analysis functions are implemented in the widget module
# but these core functions are defined here for reuse

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
    print(f"Processing {file_path}...")
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"Failed to open {file_path}")
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
    print(f"Processed {file_path}, detected {len(masks)} ROIs.")
    return file_path, roi_changes, video_duration


def process_videos(directory, fps=5, block_length=5, skip_seconds=5, min_radius=100, max_radius=125):
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
    avi_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.avi')]
    if not avi_files:
        print("No AVI files found in the directory.")
        return {}, {}, [], None

    first_video = avi_files[0]
    print(f"Using {first_video} to detect ROIs...")
    cap = cv2.VideoCapture(first_video)
    if not cap.isOpened():
        print(f"Failed to open {first_video}")
        return {}, {}, [], None

    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        print(f"Failed to read the first frame of {first_video}")
        return {}, {}, [], None

    gray_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    masks, labeled_frame = detect_circles_and_create_masks(gray_frame, min_radius, max_radius)
    if not masks:
        print(f"No ROIs detected in the first frame of {first_video}")
        return {}, {}, [], None

    results = {}
    durations = {}
    with ProcessPoolExecutor() as executor:
        future_map = {
            executor.submit(process_video, file_path, masks, fps, block_length, skip_seconds): file_path
            for file_path in avi_files
        }
        for future in future_map:
            file_path = future_map[future]
            try:
                file_path, roi_changes, total_duration = future.result()
                if roi_changes is not None:
                    results[file_path] = roi_changes
                    durations[file_path] = total_duration
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return results, durations, masks, labeled_frame