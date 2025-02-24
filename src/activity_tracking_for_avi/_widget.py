"""
This module provides the widget functionality for the napari plugin.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

from napari.layers import Image, Labels
from napari.types import LayerDataTuple
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QSpinBox, QDoubleSpinBox, QComboBox, QFileDialog, QTabWidget,
    QListWidget, QGroupBox, QFormLayout, QSlider, QCheckBox
)
from qtpy.QtCore import Qt, Signal, Slot
from napari.qt.threading import thread_worker
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ._reader import (
    detect_circles_and_create_masks, process_video, process_videos,
    read_video, get_first_frame
)


class ActivityAnalysisWidget(QWidget):
    """
    Widget for analyzing activity in AVI videos.
    """
    
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.setup_ui()
        
        # Analysis state variables
        self.directory = None
        self.video_path = None
        self.masks = []
        self.labeled_frame = None
        self.results = {}
        self.durations = {}
        self.merged_results = {}
        self.roi_thresholds = {}
        self.movement_data = {}
        self.fraction_data = {}
        self.quiescence_data = {}
        self.sleep_data = {}
        self.roi_colors = {}
        
        # Connect signals/slots
        self.btn_load_file.clicked.connect(self.load_file)
        self.btn_load_dir.clicked.connect(self.load_directory)
        self.btn_detect_rois.clicked.connect(self.detect_rois)
        self.btn_analyze.clicked.connect(self.run_analysis)
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
    
    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Create tabs
        self.tab_widget = QTabWidget()
        self.tab_input = QWidget()
        self.tab_analysis = QWidget()
        self.tab_results = QWidget()
        self.tab_widget.addTab(self.tab_input, "Input")
        self.tab_widget.addTab(self.tab_analysis, "Analysis")
        self.tab_widget.addTab(self.tab_results, "Results")
        layout.addWidget(self.tab_widget)
        
        # Set up the input tab
        self.setup_input_tab()
        
        # Set up the analysis tab
        self.setup_analysis_tab()
        
        # Set up the results tab
        self.setup_results_tab()
    
    def setup_input_tab(self):
        """Set up the input tab UI."""
        layout = QVBoxLayout()
        self.tab_input.setLayout(layout)
        
        # File/Directory loading section
        file_group = QGroupBox("Load Data")
        file_layout = QVBoxLayout()
        file_group.setLayout(file_layout)
        
        # File loading
        file_btn_layout = QHBoxLayout()
        self.btn_load_file = QPushButton("Load Video File")
        self.btn_load_dir = QPushButton("Load Directory")
        file_btn_layout.addWidget(self.btn_load_file)
        file_btn_layout.addWidget(self.btn_load_dir)
        file_layout.addLayout(file_btn_layout)
        
        # File info
        self.lbl_file_info = QLabel("No file loaded")
        file_layout.addWidget(self.lbl_file_info)
        
        layout.addWidget(file_group)
        
        # ROI detection parameters
        roi_group = QGroupBox("ROI Detection Parameters")
        roi_layout = QFormLayout()
        roi_group.setLayout(roi_layout)
        
        self.min_radius = QSpinBox()
        self.min_radius.setRange(50, 200)
        self.min_radius.setValue(100)
        roi_layout.addRow("Min Radius:", self.min_radius)
        
        self.max_radius = QSpinBox()
        self.max_radius.setRange(100, 250)
        self.max_radius.setValue(125)
        roi_layout.addRow("Max Radius:", self.max_radius)
        
        self.btn_detect_rois = QPushButton("Detect ROIs")
        roi_layout.addRow(self.btn_detect_rois)
        
        layout.addWidget(roi_group)
        
        # Add stretch to push everything to the top
        layout.addStretch()
    
    def setup_analysis_tab(self):
        """Set up the analysis tab UI."""
        layout = QVBoxLayout()
        self.tab_analysis.setLayout(layout)
        
        # Analysis parameters
        analysis_group = QGroupBox("Analysis Parameters")
        analysis_layout = QFormLayout()
        analysis_group.setLayout(analysis_layout)
        
        self.fps = QSpinBox()
        self.fps.setRange(1, 60)
        self.fps.setValue(5)
        analysis_layout.addRow("FPS:", self.fps)
        
        self.block_length = QSpinBox()
        self.block_length.setRange(1, 20)
        self.block_length.setValue(5)
        analysis_layout.addRow("Block Length:", self.block_length)
        
        self.skip_seconds = QSpinBox()
        self.skip_seconds.setRange(0, 60)
        self.skip_seconds.setValue(5)
        analysis_layout.addRow("Skip Seconds:", self.skip_seconds)
        
        self.threshold_blocks = QSpinBox()
        self.threshold_blocks.setRange(1, 100)
        self.threshold_blocks.setValue(10)
        analysis_layout.addRow("Threshold Block Count:", self.threshold_blocks)
        
        self.quiescence_threshold = QDoubleSpinBox()
        self.quiescence_threshold.setRange(0.0, 1.0)
        self.quiescence_threshold.setValue(0.5)
        self.quiescence_threshold.setSingleStep(0.05)
        analysis_layout.addRow("Quiescence Threshold:", self.quiescence_threshold)
        
        self.consecutive_quiescent = QSpinBox()
        self.consecutive_quiescent.setRange(60, 1800)
        self.consecutive_quiescent.setValue(480)
        self.consecutive_quiescent.setSingleStep(60)
        analysis_layout.addRow("Consecutive Quiescent Seconds:", self.consecutive_quiescent)
        
        layout.addWidget(analysis_group)
        
        # Run analysis button
        self.btn_analyze = QPushButton("Run Analysis")
        layout.addWidget(self.btn_analyze)
        
        # Progress indicator
        self.lbl_progress = QLabel("Status: Ready")
        layout.addWidget(self.lbl_progress)
        
        # Add stretch to push everything to the top
        layout.addStretch()
    
    def setup_results_tab(self):
        """Set up the results tab UI."""
        layout = QVBoxLayout()
        self.tab_results.setLayout(layout)
        
        # Plot selection
        plot_group = QGroupBox("Plot Selection")
        plot_layout = QVBoxLayout()
        plot_group.setLayout(plot_layout)
        
        self.plot_type = QComboBox()
        self.plot_type.addItems([
            "Raw Intensity Changes", 
            "Movement States", 
            "Fraction Movement", 
            "Quiescence States", 
            "Sleep States"
        ])
        plot_layout.addWidget(self.plot_type)
        
        # Plot buttons
        self.btn_plot = QPushButton("Generate Plot")
        self.btn_plot.clicked.connect(self.generate_plot)
        plot_layout.addWidget(self.btn_plot)
        
        layout.addWidget(plot_group)
        
        # Plot canvas
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
    
    def load_file(self):
        """Load a single video file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "", "Video Files (*.avi)"
        )
        if not filepath:
            return
        
        self.video_path = filepath
        self.directory = os.path.dirname(filepath)
        self.lbl_file_info.setText(f"Loaded file: {os.path.basename(filepath)}")
        
        # Load the video into napari
        self.viewer.layers.select_all()
        self.viewer.layers.remove_selected()
        
        video = read_video(filepath)
        if video is not None:
            self.viewer.add_image(
                video, 
                name=os.path.basename(filepath),
                rgb=True
            )
            self.viewer.dims.set_point(0, 0)  # Show first frame
    
    def load_directory(self):
        """Load a directory of video files."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory with Video Files"
        )
        if not directory:
            return
        
        self.directory = directory
        self.video_path = None
        
        # Count AVI files
        avi_files = [f for f in os.listdir(directory) if f.lower().endswith('.avi')]
        self.lbl_file_info.setText(f"Loaded directory: {os.path.basename(directory)} with {len(avi_files)} AVI files")
    
    def detect_rois(self):
        """Detect ROIs in the loaded video or first video in directory."""
        if self.video_path:
            # Single file mode
            first_frame = get_first_frame(self.video_path)
            if first_frame is None:
                self.lbl_file_info.setText("Error: Could not read the video file")
                return
            
            gray_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)
            self.masks, self.labeled_frame = detect_circles_and_create_masks(
                gray_frame,
                self.min_radius.value(),
                self.max_radius.value()
            )
            
            # Clear existing layers
            self.viewer.layers.select_all()
            self.viewer.layers.remove_selected()
            
            # Show the labeled frame
            self.viewer.add_image(
                self.labeled_frame,
                name="Detected ROIs",
                rgb=True
            )
            
            # Add original video
            video = read_video(self.video_path)
            if video is not None:
                self.viewer.add_image(
                    video,
                    name=os.path.basename(self.video_path),
                    rgb=True,
                    visible=False
                )
            
            # Add the masks
            for i, mask in enumerate(self.masks):
                self.viewer.add_labels(
                    mask, 
                    name=f"ROI {i+1} Mask",
                    visible=False
                )
            
            self.lbl_file_info.setText(f"Detected {len(self.masks)} ROIs in {os.path.basename(self.video_path)}")
        
        elif self.directory:
            # Directory mode
            avi_files = [os.path.join(self.directory, f) for f in os.listdir(self.directory) 
                        if f.lower().endswith('.avi')]
            if not avi_files:
                self.lbl_file_info.setText("Error: No AVI files found in directory")
                return
            
            first_video = avi_files[0]
            first_frame = get_first_frame(first_video)
            if first_frame is None:
                self.lbl_file_info.setText(f"Error: Could not read {os.path.basename(first_video)}")
                return
            
            gray_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2GRAY)
            self.masks, self.labeled_frame = detect_circles_and_create_masks(
                gray_frame,
                self.min_radius.value(),
                self.max_radius.value()
            )
            
            # Clear existing layers
            self.viewer.layers.select_all()
            self.viewer.layers.remove_selected()
            
            # Show the labeled frame
            self.viewer.add_image(
                self.labeled_frame,
                name="Detected ROIs",
                rgb=True
            )
            
            # Add the first video for reference
            video = read_video(first_video)
            if video is not None:
                self.viewer.add_image(
                    video,
                    name=os.path.basename(first_video),
                    rgb=True,
                    visible=False
                )
            
            # Add the masks
            for i, mask in enumerate(self.masks):
                self.viewer.add_labels(
                    mask, 
                    name=f"ROI {i+1} Mask",
                    visible=False
                )
            
            self.lbl_file_info.setText(f"Detected {len(self.masks)} ROIs in {os.path.basename(first_video)}")
        
        else:
            self.lbl_file_info.setText("Error: No video file or directory loaded")
    
    @thread_worker
    def _run_analysis(self):
        """Run the analysis in a separate thread."""
        if not self.masks:
            return "Error: No ROIs detected. Please detect ROIs first."
        
        if self.video_path:
            # Single file mode
            file_path, roi_changes, duration = process_video(
                self.video_path, 
                self.masks,
                fps=self.fps.value(),
                block_length=self.block_length.value(),
                skip_seconds=self.skip_seconds.value()
            )
            
            if roi_changes is None:
                return "Error processing video."
            
            self.results = {file_path: roi_changes}
            self.durations = {file_path: duration}
            
        elif self.directory:
            # Directory mode
            self.results, self.durations, _, _ = process_videos(
                self.directory,
                fps=self.fps.value(),
                block_length=self.block_length.value(),
                skip_seconds=self.skip_seconds.value(),
                min_radius=self.min_radius.value(),
                max_radius=self.max_radius.value()
            )
            
            if not self.results:
                return "Error processing videos in directory."
        
        else:
            return "Error: No video file or directory loaded."
        
        # Merge results
        self.merged_results = self.merge_results()
        
        # Get consistent colors for ROIs
        self.roi_colors = self.get_roi_colors(sorted(self.merged_results.keys()))
        
        # Analysis pipeline
        self.roi_thresholds = self.compute_roi_thresholds(
            self.merged_results, 
            self.threshold_blocks.value()
        )
        
        self.movement_data = self.define_movement_events_with_noise(
            self.merged_results, 
            self.roi_thresholds
        )
        
        self.fraction_data = self.bin_fraction_movement(
            self.movement_data
        )
        
        self.quiescence_data = self.bin_quiescence(
            self.movement_data,
            move_threshold=self.quiescence_threshold.value()
        )
        
        self.sleep_data = self.define_sleep(
            self.quiescence_data,
            consecutive_quiescence=self.consecutive_quiescent.value()
        )
        
        return "Analysis completed successfully!"