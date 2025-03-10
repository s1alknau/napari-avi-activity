# napari-avi-activity

[![License BSD-3](https://img.shields.io/pypi/l/activity-tracking-for-avi.svg?color=green)](https://github.com/s1alknau/napari-avi-activity/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/activity-tracking-for-avi.svg?color=green)](https://pypi.org/project/activity-tracking-for-avi)
[![Python Version](https://img.shields.io/pypi/pyversions/activity-tracking-for-avi.svg?color=green)](https://python.org)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/activity-tracking-for-avi)](https://napari-hub.org/plugins/activity-tracking-for-avi)

A napari plugin for analyzing activity and sleep behavior from AVI video files.

## Description

This plugin provides tools for analyzing activity and sleep patterns in video recordings. It is designed to work with AVI files containing time-lapse recordings of subjects (such as animals or organisms) that need to be tracked for movement and sleep states.

Key features:
- Automatic detection of regions of interest (ROIs) in video frames
- Analysis of movement within each ROI over time
- Calculation of quiescence periods and sleep states
- Interactive data visualization
- Batch processing of multiple video files

## Installation

You can install `activity-tracking-for-avi` via [pip]:

```bash
pip install activity-tracking-for-avi
```

## Usage

1. Start napari:
```bash
napari
```

2. Open the plugin from the `Plugins` menu

3. Load your video files:
   - Click "Load Video File" for a single video
   - Click "Load Directory" for a folder of videos

4. Adjust ROI detection parameters and click "Detect ROIs"

5. Set analysis parameters (FPS, thresholds, etc.) and click "Run Analysis"

6. View results in the "Results" tab, where you can generate various plots:
   - Raw intensity changes
   - Movement states
   - Fraction of movement
   - Quiescence states
   - Sleep states

## Data Format

The plugin expects AVI video files. For optimal performance:
- Frame rate should be consistent across videos
- Subjects should be clearly distinguishable from the background
- Camera position should be fixed

## Contributing

Contributions are welcome. Please feel free to submit a Pull Request.

## License

Distributed under the terms of the [BSD-3] license,
"activity-tracking-for-avi" is free and open source software.

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[file an issue]: https://github.com/s1alknau/napari-avi-activity/issues
[pip]: https://pypi.org/project/pip/
