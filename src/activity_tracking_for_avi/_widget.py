import os

import cv2
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ._reader import (
    bin_fraction_movement,
    bin_quiescence,
    compute_roi_thresholds,
    define_movement_events_with_noise,
    define_sleep,
    detect_circles_and_create_masks,
    get_roi_colors,
    merge_results,
    process_video,
    process_videos,
)


def get_first_frame(video_path):
    """
    Get the first frame of a video file.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


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
        self.current_worker = None
        self.progress_timer = None
        self.progress_value = 0
        self.progress_dialog = None

        # Connect signals/slots
        self.btn_load_file.clicked.connect(self.load_file)
        self.btn_load_dir.clicked.connect(self.load_directory)
        self.btn_detect_rois.clicked.connect(self.detect_rois)
        self.btn_analyze.clicked.connect(self.run_analysis)
        self.btn_stop.clicked.connect(self.stop_analysis)
        self.btn_plot.clicked.connect(self.generate_plot)
        self.btn_save_plot.clicked.connect(self.save_current_plot)
        self.btn_save_all_plots.clicked.connect(self.save_all_plots)
        self.btn_save_results.clicked.connect(self.save_results)
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

    def setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.tab_widget = QTabWidget()
        self.tab_input = QWidget()
        self.tab_analysis = QWidget()
        self.tab_results = QWidget()
        self.tab_widget.addTab(self.tab_input, "Input")
        self.tab_widget.addTab(self.tab_analysis, "Analysis")
        self.tab_widget.addTab(self.tab_results, "Results")
        layout.addWidget(self.tab_widget)

        self.setup_input_tab()
        self.setup_analysis_tab()
        self.setup_results_tab()

    def setup_input_tab(self):
        layout = QVBoxLayout()
        self.tab_input.setLayout(layout)

        file_group = QGroupBox("Load Data")
        file_layout = QVBoxLayout()
        file_group.setLayout(file_layout)

        file_btn_layout = QHBoxLayout()
        self.btn_load_file = QPushButton("Load Video File")
        self.btn_load_dir = QPushButton("Load Directory")
        file_btn_layout.addWidget(self.btn_load_file)
        file_btn_layout.addWidget(self.btn_load_dir)
        file_layout.addLayout(file_btn_layout)

        self.lbl_file_info = QLabel("No file loaded")
        file_layout.addWidget(self.lbl_file_info)
        layout.addWidget(file_group)

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
        layout.addStretch()

    def setup_analysis_tab(self):
        layout = QVBoxLayout()
        self.tab_analysis.setLayout(layout)

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
        analysis_layout.addRow(
            "Consecutive Quiescent Seconds:", self.consecutive_quiescent
        )

        layout.addWidget(analysis_group)

        # Add analyze and stop buttons in the same row
        button_layout = QHBoxLayout()
        self.btn_analyze = QPushButton("Run Analysis")
        self.btn_stop = QPushButton("Stop Analysis")
        self.btn_stop.setEnabled(False)  # Disabled until analysis starts
        button_layout.addWidget(self.btn_analyze)
        button_layout.addWidget(self.btn_stop)
        layout.addLayout(button_layout)

        # Add progress bar (kept for UI consistency but will be hidden)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(False)  # Hide until analysis starts
        layout.addWidget(self.progress_bar)

        self.lbl_progress = QLabel("Status: Ready")
        layout.addWidget(self.lbl_progress)
        layout.addStretch()

    def setup_results_tab(self):
        layout = QVBoxLayout()
        self.tab_results.setLayout(layout)

        # Add time range selector
        time_range_group = QGroupBox("Time Range Selection")
        time_range_layout = QHBoxLayout()
        time_range_group.setLayout(time_range_layout)

        time_range_layout.addWidget(QLabel("Start Time (s):"))
        self.time_start = QSpinBox()
        self.time_start.setRange(0, 100000)
        self.time_start.setValue(0)
        time_range_layout.addWidget(self.time_start)

        time_range_layout.addWidget(QLabel("End Time (s):"))
        self.time_end = QSpinBox()
        self.time_end.setRange(0, 100000)
        self.time_end.setValue(100000)
        time_range_layout.addWidget(self.time_end)

        self.use_time_range = QCheckBox("Apply Time Range")
        time_range_layout.addWidget(self.use_time_range)

        layout.addWidget(time_range_group)

        # Plot configuration group
        plot_group = QGroupBox("Plot Configuration")
        plot_layout = QVBoxLayout()
        plot_group.setLayout(plot_layout)

        # Plot type selection
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Plot Type:"))
        self.plot_type = QComboBox()
        self.plot_type.addItems(
            [
                "Raw Intensity Changes",
                "Movement States",
                "Fraction Movement",
                "Quiescence States",
                "Sleep States",
            ]
        )
        type_layout.addWidget(self.plot_type)
        plot_layout.addLayout(type_layout)

        # Add visual style options
        style_layout = QHBoxLayout()
        style_layout.addWidget(QLabel("DPI:"))
        self.plot_dpi = QSpinBox()
        self.plot_dpi.setRange(72, 600)
        self.plot_dpi.setValue(150)
        style_layout.addWidget(self.plot_dpi)

        style_layout.addWidget(QLabel("Figure Width:"))
        self.plot_width = QSpinBox()
        self.plot_width.setRange(5, 30)
        self.plot_width.setValue(12)
        style_layout.addWidget(self.plot_width)

        style_layout.addWidget(QLabel("Height Per ROI:"))
        self.plot_height_per_roi = QDoubleSpinBox()
        self.plot_height_per_roi.setRange(0.2, 2.0)
        self.plot_height_per_roi.setValue(0.5)
        self.plot_height_per_roi.setSingleStep(0.1)
        style_layout.addWidget(self.plot_height_per_roi)

        plot_layout.addLayout(style_layout)

        # Button row
        button_layout = QHBoxLayout()
        self.btn_plot = QPushButton("Generate Plot")
        self.btn_save_plot = QPushButton("Save Current Plot")
        self.btn_save_all_plots = QPushButton("Save All Plots")
        self.btn_save_results = QPushButton("Save Results")

        button_layout.addWidget(self.btn_plot)
        button_layout.addWidget(self.btn_save_plot)
        button_layout.addWidget(self.btn_save_all_plots)
        button_layout.addWidget(self.btn_save_results)

        plot_layout.addLayout(button_layout)
        layout.addWidget(plot_group)

        # Use a scroll area for the figure to handle different size plots
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        figure_container = QWidget()
        figure_layout = QVBoxLayout(figure_container)

        self.figure = Figure(figsize=(10, 8), dpi=150)
        self.canvas = FigureCanvas(self.figure)
        figure_layout.addWidget(self.canvas)

        scroll_area.setWidget(figure_container)
        layout.addWidget(scroll_area, 1)  # Give it stretch factor of 1

    def load_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "", "Video Files (*.avi)"
        )
        if not filepath:
            return

        self.video_path = filepath
        self.directory = os.path.dirname(filepath)

        # Get first frame of the selected video
        first_frame = get_first_frame(filepath)
        if first_frame is None:
            self.lbl_file_info.setText(
                f"Error: Could not read {os.path.basename(filepath)}"
            )
            return

        # Convert BGR to RGB for proper display
        rgb_first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

        # Clear previous layers and add first frame to viewer
        self.viewer.layers.select_all()
        self.viewer.layers.remove_selected()
        self.viewer.add_image(rgb_first_frame, name="First Frame", rgb=True)

        self.lbl_file_info.setText(f"Loaded file: {os.path.basename(filepath)}")

        # Reset analysis state
        self.masks = []
        self.labeled_frame = None

    def load_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if not directory:
            return

        self.directory = directory
        self.video_path = None

        # Find all AVI files in the directory
        avi_files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower().endswith(".avi")
        ]

        if not avi_files:
            self.lbl_file_info.setText("No AVI files found in the selected directory")
            return

        # Get first frame of the first video
        first_video = avi_files[0]
        first_frame = get_first_frame(first_video)

        if first_frame is None:
            self.lbl_file_info.setText(
                f"Error: Could not read {os.path.basename(first_video)}"
            )
            return

        # Convert BGR to RGB for proper display
        rgb_first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

        # Clear previous layers and add first frame to viewer
        self.viewer.layers.select_all()
        self.viewer.layers.remove_selected()
        self.viewer.add_image(
            rgb_first_frame,
            name=f"First Frame - {os.path.basename(first_video)}",
            rgb=True,
        )

        self.lbl_file_info.setText(
            f"Loaded directory: {os.path.basename(directory)} ({len(avi_files)} AVI files)"
        )

        # Reset analysis state
        self.masks = []
        self.labeled_frame = None

    def detect_rois(self):
        # Check if we have a video or directory loaded
        if not (self.video_path or self.directory):
            self.lbl_file_info.setText("Error: No video file or directory loaded")
            return

        # Get the first frame and process it
        if self.video_path:
            first_frame = get_first_frame(self.video_path)
            source_name = os.path.basename(self.video_path)
        else:  # Use directory
            avi_files = [
                os.path.join(self.directory, f)
                for f in os.listdir(self.directory)
                if f.lower().endswith(".avi")
            ]
            if not avi_files:
                self.lbl_file_info.setText("Error: No AVI files found in directory")
                return

            first_frame = get_first_frame(avi_files[0])
            source_name = os.path.basename(avi_files[0])

        if first_frame is None:
            self.lbl_file_info.setText(f"Error: Could not read {source_name}")
            return

        # Convert to grayscale for ROI detection
        gray_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

        # Detect ROIs
        self.masks, self.labeled_frame = detect_circles_and_create_masks(
            gray_frame, self.min_radius.value(), self.max_radius.value()
        )

        # Convert labeled frame from BGR to RGB for display
        rgb_labeled_frame = cv2.cvtColor(self.labeled_frame, cv2.COLOR_BGR2RGB)

        # Remove all existing layers
        self.viewer.layers.select_all()
        self.viewer.layers.remove_selected()

        # Add the labeled frame with detected ROIs
        self.viewer.add_image(rgb_labeled_frame, name="Detected ROIs", rgb=True)

        # Add mask layers (hidden by default)
        for i, mask in enumerate(self.masks):
            self.viewer.add_labels(mask, name=f"ROI {i+1} Mask", visible=False)

        self.lbl_file_info.setText(f"Detected {len(self.masks)} ROIs in {source_name}")

    @thread_worker
    def _run_analysis(self):
        if not self.masks:
            return "Error: No ROIs detected. Please detect ROIs first."

        self.lbl_progress.setText("Status: Running analysis...")

        if self.video_path:
            # Single video mode
            file_path, roi_changes, duration = process_video(
                self.video_path,
                self.masks,
                fps=self.fps.value(),
                block_length=self.block_length.value(),
                skip_seconds=self.skip_seconds.value(),
            )
            if roi_changes is None:
                return "Error processing video."
            self.results = {file_path: roi_changes}
            self.durations = {file_path: duration}
        elif self.directory:
            # Directory mode - process all videos
            self.results, self.durations, _, _ = process_videos(
                self.directory,
                fps=self.fps.value(),
                block_length=self.block_length.value(),
                skip_seconds=self.skip_seconds.value(),
                min_radius=self.min_radius.value(),
                max_radius=self.max_radius.value(),
            )
            if not self.results:
                return "Error processing videos in directory."
        else:
            return "Error: No video file or directory loaded."

        # Process and analyze the results
        self.merged_results = merge_results(self.results, self.durations)
        self.roi_colors = get_roi_colors(sorted(self.merged_results.keys()))
        self.roi_thresholds = compute_roi_thresholds(
            self.merged_results, self.threshold_blocks.value()
        )
        self.movement_data = define_movement_events_with_noise(
            self.merged_results, self.roi_thresholds
        )
        self.fraction_data = bin_fraction_movement(self.movement_data)
        self.quiescence_data = bin_quiescence(
            self.movement_data, move_threshold=self.quiescence_threshold.value()
        )
        self.sleep_data = define_sleep(
            self.quiescence_data,
            consecutive_quiescence=self.consecutive_quiescent.value(),
        )

        # Set the time range end value based on actual data
        max_time = 0
        for _roi, data in self.merged_results.items():

            if data:
                times, _ = zip(*data)
                max_time = max(times) if max(times) > max_time else max_time
        self.time_end.setValue(int(max_time))

        return "Analysis completed successfully!"

    def update_fake_progress(self):
        """Simulate progress that moves faster at the beginning and slows down"""
        if self.progress_value < 90:  # Max out at 90% until truly complete
            # Slow down as we approach higher percentages
            increment = max(1, int((95 - self.progress_value) / 10))
            self.progress_value += increment
        self.progress_dialog.setValue(self.progress_value)

    def run_analysis(self):
        # Start the background worker
        worker = self._run_analysis()
        worker.returned.connect(self.on_analysis_complete)
        worker.finished.connect(self.on_analysis_finished)
        worker.start()

        # Store the current worker
        self.current_worker = worker

        # Progress dialog setup
        self.progress_dialog = QProgressDialog(
            "Analyzing videos...", "Cancel", 0, 100, self
        )
        self.progress_dialog.setWindowTitle("Analysis Progress")
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.canceled.connect(
            self.stop_analysis
        )  # Connect cancel button
        self.progress_dialog.setWindowModality(Qt.WindowModal)

        # Simulate progress with timer
        self.progress_value = 0
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_fake_progress)
        self.progress_timer.start(100)  # Update every 100ms

        # Update the UI
        self.btn_analyze.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_progress.setText("Status: Running analysis...")
        self.progress_dialog.show()

    def stop_analysis(self):
        """Stop the currently running analysis"""
        if self.current_worker is not None:
            self.current_worker.quit()
            self.lbl_progress.setText("Status: Analysis stopped by user")
            self.btn_analyze.setEnabled(True)
            self.btn_stop.setEnabled(False)

            # Clean up progress dialog and timer
            if hasattr(self, "progress_timer") and self.progress_timer.isActive():
                self.progress_timer.stop()
            if hasattr(self, "progress_dialog") and self.progress_dialog:
                self.progress_dialog.close()

            self.current_worker = None

    def on_analysis_finished(self):
        """Called when the worker finishes, regardless of result"""
        # Stop the timer and clean up
        if hasattr(self, "progress_timer") and self.progress_timer.isActive():
            self.progress_timer.stop()
        if hasattr(self, "progress_dialog") and self.progress_dialog:
            self.progress_dialog.setValue(100)  # Show 100% before closing
            self.progress_dialog.close()

        self.current_worker = None
        self.btn_stop.setEnabled(False)

    def on_analysis_complete(self, message):
        self.lbl_progress.setText(f"Status: {message}")
        self.btn_analyze.setEnabled(True)
        self.btn_stop.setEnabled(False)

        # Switch to results tab if analysis was successful
        if isinstance(message, str) and "completed successfully" in message:
            self.tab_widget.setCurrentIndex(2)  # Switch to Results tab

    def _apply_time_range(self, data_dict):
        """Filter data to only include points within the selected time range."""
        if not self.use_time_range.isChecked():
            return data_dict

        start_time = self.time_start.value()
        end_time = self.time_end.value()

        filtered_dict = {}
        for roi, data in data_dict.items():
            filtered_data = [(t, val) for t, val in data if start_time <= t <= end_time]
            if filtered_data:  # Only include ROIs that have data in the time range
                filtered_dict[roi] = filtered_data

        return filtered_dict

    def generate_plot(self):
        """Generate plots that match the standalone script output style."""
        # Get the selected plot type
        plot_type = self.plot_type.currentText()

        # Check if we have data to plot
        if not self.merged_results:
            self.lbl_progress.setText("Status: No data to plot. Run analysis first.")
            return

        # Apply time range filter to data
        if plot_type == "Raw Intensity Changes":
            filtered_data = self._apply_time_range(self.merged_results)
        elif plot_type == "Movement States":
            filtered_data = self._apply_time_range(self.movement_data)
        elif plot_type == "Fraction Movement":
            filtered_data = self._apply_time_range(self.fraction_data)
        elif plot_type == "Quiescence States":
            filtered_data = self._apply_time_range(self.quiescence_data)
        elif plot_type == "Sleep States":
            filtered_data = self._apply_time_range(self.sleep_data)
        else:
            return

        if not filtered_data:
            self.lbl_progress.setText("Status: No data in selected time range.")
            return

        # Clear the figure
        self.figure.clear()

        # Get time range info for title
        time_range_text = ""
        if self.use_time_range.isChecked():
            time_range_text = (
                f" (Time: {self.time_start.value()}-{self.time_end.value()}s)"
            )

        # Get all ROIs in a consistent order
        all_rois = sorted(filtered_data.keys())
        num_rois = len(all_rois)

        # Determine global time range for consistent x-axis across plots
        global_min_time = float("inf")
        global_max_time = float("-inf")

        for roi in all_rois:
            data = filtered_data[roi]
            if data:
                times, _ = zip(*data)
                if times:
                    global_min_time = (
                        min(times) if min(times) < global_min_time else global_min_time
                    )

                    global_max_time = (
                        max(times) if max(times) > global_max_time else global_max_time
                    )

        if global_min_time == float("inf"):
            global_min_time = 0
        if global_max_time == float("-inf"):
            global_max_time = 100000

        # Adjust padding for x-axis
        x_padding = (global_max_time - global_min_time) * 0.02
        x_min = global_min_time - x_padding
        x_max = global_max_time + x_padding

        # Create consistent color scheme for all ROIs
        if not self.roi_colors:
            self.roi_colors = get_roi_colors(all_rois)

        # Configure figure based on plot type
        if plot_type == "Raw Intensity Changes":
            # Raw intensity changes plot
            self.figure.set_size_inches(self.plot_width.value(), 6)
            ax = self.figure.add_subplot(111)

            offset_step = 5e5

            for idx, roi in enumerate(all_rois):
                data = filtered_data[roi]
                if not data:
                    continue

                times, changes = zip(*data)
                times = np.array(times)
                changes = np.array(changes)

                offset = idx * offset_step
                offset_changes = changes + offset

                # Plot the intensity changes
                ax.plot(
                    times, offset_changes, color=self.roi_colors[roi], linewidth=0.8
                )

                # Add threshold line if available
                if roi in self.roi_thresholds:
                    threshold = self.roi_thresholds[roi]
                    offset_threshold = threshold + offset
                    ax.axhline(
                        offset_threshold,
                        xmin=0,
                        xmax=1,
                        color="gray",
                        linestyle="--",
                        linewidth=0.8,
                    )

                # Add ROI label at the right side
                y_pos = offset + np.median(changes) if len(changes) > 0 else offset
                ax.text(
                    x_max + x_padding / 2,
                    y_pos,
                    f"ROI {roi}",
                    color=self.roi_colors[roi],
                    ha="left",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                )

            # Set labels and title
            ax.set_title(
                f"ROI Intensity Changes Over Time (with Thresholds){time_range_text}"
            )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Pixel Intensity Change (Sum of 8-bit Values) + Offset")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(x_min, x_max)

        else:
            # Determine plot configuration based on plot type
            if plot_type == "Fraction Movement":
                title = (
                    f"Fraction of Movement per 60s Binning Interval{time_range_text}"
                )
                binary_states = False
                step_type = None
                fill_alpha = 0
            elif plot_type == "Movement States":
                title = f"Movement State (1=Movement, 0=No Movement){time_range_text}"
                binary_states = True
                y_labels = ["No", "Yes"]
                step_type = "post"
                fill_alpha = 0.3
            elif plot_type == "Quiescence States":
                title = f"Quiescence State (1=Quiescent, 0=Active){time_range_text}"
                binary_states = True
                y_labels = ["Active", "Quiescent"]
                step_type = "mid"
                fill_alpha = 0.3
            elif plot_type == "Sleep States":
                title = f"Sleep State (1=Sleep, 0=Awake){time_range_text}"
                binary_states = True
                y_labels = ["Awake", "Sleep"]
                step_type = "mid"
                fill_alpha = 0.2  # Lighter fill for sleep state
            else:
                return

            # Create a more compact layout
            height_per_roi = self.plot_height_per_roi.value()
            self.figure.set_size_inches(
                self.plot_width.value(), num_rois * height_per_roi
            )

            # Create subplots with shared x-axis
            axes = self.figure.subplots(num_rois, 1, sharex=True)
            if num_rois == 1:
                axes = [axes]  # Make sure axes is a list even with one subplot

            for i, roi in enumerate(all_rois):
                ax = axes[i]
                data = filtered_data[roi]

                if not data or len(data) == 0:
                    # Add text for ROIs with no data
                    ax.text(
                        0.5,
                        0.5,
                        "No data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )

                    # Set up y-axis and roi label
                    if binary_states:
                        ax.set_yticks([0, 1])
                        ax.set_yticklabels(y_labels)
                        ax.set_ylim(-0.1, 1.1)
                    else:
                        ax.set_ylim(0, 1.05)
                        ax.set_yticks([0, 0.5, 1.0])

                    # Add ROI label
                    ax.text(
                        x_max + x_padding / 2,
                        0.5,
                        f"ROI {roi}",
                        transform=ax.get_yaxis_transform(),
                        color=self.roi_colors[roi],
                        ha="left",
                        va="center",
                        fontsize=8,
                        fontweight="bold",
                    )
                    continue

                times, values = zip(*data)
                times = np.array(times)
                values = np.array(values)

                # Different plotting styles based on plot type
                if binary_states:
                    # For movement, quiescence, sleep (binary states)
                    ax.step(
                        times,
                        values,
                        where=step_type,
                        color=self.roi_colors[roi],
                        linewidth=0.8,
                    )

                    # For sleep states, use a different fill style
                    if plot_type == "Sleep States":
                        # Fill with a light background
                        ax.fill_between(
                            times,
                            values,
                            0,
                            step=step_type,
                            alpha=fill_alpha,
                            color=self.roi_colors[roi],
                        )
                    else:
                        # For other binary states, use a thin fill
                        ax.fill_between(
                            times,
                            values,
                            0,
                            step=step_type,
                            alpha=fill_alpha,
                            color=self.roi_colors[roi],
                        )

                    ax.set_yticks([0, 1])
                    ax.set_yticklabels(y_labels)
                    ax.set_ylim(-0.1, 1.1)
                else:
                    # For fraction movement (continuous values)
                    # Use line plot without markers for cleaner appearance
                    ax.plot(times, values, color=self.roi_colors[roi], linewidth=0.8)

                    # Add threshold line for fraction movement
                    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
                    ax.set_ylim(0, 1.05)
                    ax.set_yticks([0, 0.5, 1.0])

                # Add ROI label on the right side
                ax.text(
                    x_max + x_padding / 2,
                    0.5,
                    f"ROI {roi}",
                    transform=ax.get_yaxis_transform(),
                    color=self.roi_colors[roi],
                    ha="left",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                )

                # Add subtle grid
                ax.grid(True, alpha=0.2)

                # Remove most tick labels for cleaner appearance
                if i < num_rois - 1:  # Not the last subplot
                    ax.set_xticklabels([])

            # Set common x-axis limits and title
            for ax in axes:
                ax.set_xlim(x_min, x_max)

            # Add title and x-axis label
            self.figure.suptitle(title, fontsize=10)
            axes[-1].set_xlabel("Time (s)")

        # Set tight layout and draw
        self.figure.tight_layout()
        self.canvas.draw()

        self.lbl_progress.setText(f"Status: Generated {plot_type} plot")

    def save_current_plot(self):
        """Save the current plot to a file."""
        if not hasattr(self, "figure") or self.figure is None:
            self.lbl_progress.setText("Status: No plot to save.")
            return

        # Ask for filename with directory
        current_plot_type = self.plot_type.currentText().lower().replace(" ", "_")
        filepath, filter_used = QFileDialog.getSaveFileName(
            self,
            "Save Current Plot",
            os.path.join(self.directory or "", f"{current_plot_type}.png"),
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)",
        )

        if not filepath:
            return

        try:
            dpi = self.plot_dpi.value()
            self.figure.savefig(filepath, dpi=dpi, bbox_inches="tight")
            self.lbl_progress.setText(
                f"Status: Plot saved to {os.path.basename(filepath)}"
            )
        except Exception as e:
            self.lbl_progress.setText(f"Status: Error saving plot: {e!s}")

    def save_all_plots(self):
        """Save all plot types to files."""
        if not self.merged_results:
            self.lbl_progress.setText(
                "Status: No analysis results available. Run analysis first."
            )
            return

        # Get directory to save plots
        save_dir = QFileDialog.getExistingDirectory(
            self, "Select Directory to Save All Plots", self.directory or ""
        )

        if not save_dir:
            return

        try:
            # Create a timestamp-based directory
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            plots_dir = os.path.join(save_dir, f"plots_{timestamp}")
            os.makedirs(plots_dir, exist_ok=True)

            # Remember current plot type
            current_plot = self.plot_type.currentText()
            current_time_range = self.use_time_range.isChecked()

            # Temporarily disable time range to get full plots
            self.use_time_range.setChecked(False)

            dpi = self.plot_dpi.value()
            self.lbl_progress.setText("Status: Generating and saving all plots...")

            # Generate and save each plot type
            for plot_type in [
                "Raw Intensity Changes",
                "Movement States",
                "Fraction Movement",
                "Quiescence States",
                "Sleep States",
            ]:
                # Set the plot type and generate the plot
                self.plot_type.setCurrentText(plot_type)
                self.generate_plot()

                # Save the figure
                filename = plot_type.lower().replace(" ", "_") + ".png"
                self.figure.savefig(
                    os.path.join(plots_dir, filename), dpi=dpi, bbox_inches="tight"
                )

            # Restore original plot type and time range
            self.plot_type.setCurrentText(current_plot)
            self.use_time_range.setChecked(current_time_range)
            self.generate_plot()  # Regenerate the original plot

            self.lbl_progress.setText(f"Status: All plots saved to {plots_dir}")

        except Exception as e:
            self.lbl_progress.setText(f"Status: Error saving plots: {e!s}")
            # Make sure to restore original plot type
            self.plot_type.setCurrentText(current_plot)
            self.use_time_range.setChecked(current_time_range)
            self.generate_plot()

    def save_results(self):
        """Save analysis results to CSV files."""
        if not self.merged_results:
            self.lbl_progress.setText("Status: No results to save. Run analysis first.")
            return

        # Ask user for save directory
        save_dir = QFileDialog.getExistingDirectory(
            self, "Select Directory to Save Results", self.directory or ""
        )
        if not save_dir:
            return

        try:
            # Create directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)

            # Apply time range filter if enabled
            apply_filter = self.use_time_range.isChecked()

            # Save raw intensity changes
            filtered_data = (
                self._apply_time_range(self.merged_results)
                if apply_filter
                else self.merged_results
            )
            self._save_data_dict(
                filtered_data,
                os.path.join(save_dir, "raw_intensity_changes.csv"),
                "Time (s)",
                "Intensity Change",
            )

            # Save movement states
            filtered_data = (
                self._apply_time_range(self.movement_data)
                if apply_filter
                else self.movement_data
            )
            self._save_data_dict(
                filtered_data,
                os.path.join(save_dir, "movement_states.csv"),
                "Time (s)",
                "Movement (1=Yes, 0=No)",
            )

            # Save fraction movement
            filtered_data = (
                self._apply_time_range(self.fraction_data)
                if apply_filter
                else self.fraction_data
            )
            self._save_data_dict(
                filtered_data,
                os.path.join(save_dir, "fraction_movement.csv"),
                "Time (s)",
                "Fraction Movement",
            )

            # Save quiescence states
            filtered_data = (
                self._apply_time_range(self.quiescence_data)
                if apply_filter
                else self.quiescence_data
            )
            self._save_data_dict(
                filtered_data,
                os.path.join(save_dir, "quiescence_states.csv"),
                "Time (s)",
                "Quiescence (1=Quiescent, 0=Active)",
            )

            # Save sleep states
            filtered_data = (
                self._apply_time_range(self.sleep_data)
                if apply_filter
                else self.sleep_data
            )
            self._save_data_dict(
                filtered_data,
                os.path.join(save_dir, "sleep_states.csv"),
                "Time (s)",
                "Sleep (1=Sleep, 0=Awake)",
            )

            # Save ROI threshold values
            with open(os.path.join(save_dir, "roi_thresholds.csv"), "w") as f:
                f.write("ROI,Threshold\n")
                for roi, threshold in self.roi_thresholds.items():
                    f.write(f"{roi},{threshold}\n")

            # Save analysis parameters
            with open(os.path.join(save_dir, "analysis_parameters.txt"), "w") as f:
                f.write(f"FPS: {self.fps.value()}\n")
                f.write(f"Block Length: {self.block_length.value()}\n")
                f.write(f"Skip Seconds: {self.skip_seconds.value()}\n")
                f.write(f"Threshold Block Count: {self.threshold_blocks.value()}\n")
                f.write(f"Quiescence Threshold: {self.quiescence_threshold.value()}\n")
                f.write(
                    f"Consecutive Quiescent Seconds: {self.consecutive_quiescent.value()}\n"
                )

                if self.use_time_range.isChecked():
                    f.write(
                        f"Time Range Applied: {self.time_start.value()}-{self.time_end.value()} seconds\n"
                    )
                else:
                    f.write("Time Range Applied: No\n")

            # Save source information
            with open(os.path.join(save_dir, "source_info.txt"), "w") as f:
                if self.video_path:
                    f.write(f"Single Video: {self.video_path}\n")
                elif self.directory:
                    f.write(f"Directory: {self.directory}\n")
                    f.write("Video files processed:\n")
                    for path in sorted(self.results.keys()):
                        f.write(f"- {path}\n")

            # Save current plot as image
            dpi = self.plot_dpi.value()
            self.figure.savefig(
                os.path.join(
                    save_dir,
                    f"{self.plot_type.currentText().lower().replace(' ', '_')}.png",
                ),
                dpi=dpi,
                bbox_inches="tight",
            )

            # Save all plots if needed
            plots_dir = os.path.join(save_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)

            # Remember current plot type and time range setting
            current_plot = self.plot_type.currentText()
            current_time_range = self.use_time_range.isChecked()

            # Temporarily disable time range to get full plots
            if apply_filter:
                self.use_time_range.setChecked(False)

            # Generate and save each plot type
            for plot_type in [
                "Raw Intensity Changes",
                "Movement States",
                "Fraction Movement",
                "Quiescence States",
                "Sleep States",
            ]:
                # Skip if this is the current plot (already saved above)
                if plot_type == current_plot and not apply_filter:
                    continue

                # Set the plot type and generate
                self.plot_type.setCurrentText(plot_type)
                self.generate_plot()

                # Save the figure
                filename = plot_type.lower().replace(" ", "_") + ".png"
                self.figure.savefig(
                    os.path.join(plots_dir, filename), dpi=dpi, bbox_inches="tight"
                )

            # Restore original plot type and time range
            self.plot_type.setCurrentText(current_plot)
            self.use_time_range.setChecked(current_time_range)
            self.generate_plot()  # Regenerate the original plot

            self.lbl_progress.setText(f"Status: Results saved to {save_dir}")

        except Exception as e:
            self.lbl_progress.setText(f"Status: Error saving results: {e!s}")
            # Make sure to restore original plot type
            self.plot_type.setCurrentText(current_plot)
            self.use_time_range.setChecked(current_time_range)
            self.generate_plot()

    def _save_data_dict(self, data_dict, filename, x_label):
        """Helper method to save a dictionary of data to a CSV file."""
        all_rois = sorted(data_dict.keys())

        if not all_rois:
            # Write empty file with header if no data
            with open(filename, "w") as f:
                f.write(f"{x_label}\n")
            return

        all_times = set()

        # Collect all time points
        for _roi, data in data_dict.items():

            for t, _ in data:
                all_times.add(t)

        all_times = sorted(all_times)

        if not all_times:
            # Write empty file with header if no time points
            with open(filename, "w") as f:
                header = [x_label] + [f"ROI {roi}" for roi in all_rois]
                f.write(",".join(header) + "\n")
            return

        # Create a dictionary mapping time to values for each ROI
        time_to_values = {}
        for t in all_times:
            time_to_values[t] = {}

        for roi, data in data_dict.items():
            for t, val in data:
                time_to_values[t][roi] = val

        # Write to CSV
        with open(filename, "w") as f:
            # Header
            header = [x_label] + [f"ROI {roi}" for roi in all_rois]
            f.write(",".join(header) + "\n")

            # Data rows
            for t in all_times:
                row = [str(t)]
                for roi in all_rois:
                    val = time_to_values[t].get(roi, "")
                    row.append(str(val))
                f.write(",".join(row) + "\n")

    def on_tab_changed(self, index):
        """Update UI when tabs change."""
        if index == 2:  # Results tab
            if not self.merged_results:
                self.lbl_progress.setText(
                    "Status: No analysis results available. Run analysis first."
                )
            else:
                self.lbl_progress.setText(
                    "Status: Analysis results ready. Select plot type and generate plot."
                )
        elif index == 0:  # Input tab
            if self.video_path:
                self.lbl_file_info.setText(
                    f"Loaded file: {os.path.basename(self.video_path)}"
                )
            elif self.directory:
                avi_files = [
                    f for f in os.listdir(self.directory) if f.lower().endswith(".avi")
                ]
                self.lbl_file_info.setText(
                    f"Loaded directory: {os.path.basename(self.directory)} ({len(avi_files)} AVI files)"
                )
            else:
                self.lbl_file_info.setText("No file loaded")
