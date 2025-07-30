# Live/Dead Cell Analysis Tool

## Description
An automated fluorescence microscopy image analysis tool for live/dead cell quantification. This tool processes dual-channel fluorescence images and provides comprehensive cell counting with advanced visualization.

## Features

### 1. Image Processing
- Processes dual-channel fluorescence microscopy images
  - Green channel: Live cells
  - RED channel: Dead cells
- Supports TIFF formats
- Advanced background correction
- Automatic intensity balancing

### 2. Cell Detection & Analysis
- Watershed-based cell segmentation
- Intelligent threshold detection
- Morphological filtering
- Size-based cell filtering
- Overlapping cell separation
- Automated cell counting

### 3. Visualization
- Multi-channel composite images
- Scale bar addition
- Cell boundary marking
- Analysis overlays
- Statistical visualization
- Color-coded segmentation maps

### 4. Output Generation
- Combined channel images
- Individual channel images
- Cell analysis images
- Statistical reports
- CSV summaries
- Visual analysis plots

## Installation

```bash
# Install required packages
pip install numpy pillow opencv-python scipy scikit-image matplotlib
```

## Usage

1. Prepare your image files:
   - Place images in folders with 'Default' subdirectories
   - Supported formats: CZI, TIFF

2. Run the script:
```python
python LiveDead_Cell_counting.py
```

3. Output files:
   - Combined channel images
   - Analysis images with cell detection
   - Statistical reports
   - Summary tables

## Dependencies
- numpy
- PIL (Pillow)
- opencv-python
- scipy
- scikit-image
- matplotlib

## Parameters
- `intensity_balance`: Control channel intensities
- `brightness_factor`: Adjust overall brightness
- `baseline_percentile`: Background correction
- `min_cell_size`: Minimum cell area
- `max_cell_size`: Maximum cell area

## Output Files
1. Images:
   - `*_combined.tif`: Merged channels
   - `*_cell_analysis.tif`: Detection results
   - `*_watershed_segmentation.tif`: Cell separation
   - `*_color_masks.tif`: Detection masks

2. Reports:
   - `cell_count_summary.csv`: Numerical results
   - `advanced_cell_results_*.txt`: Detailed analysis
   - `advanced_analysis_*.png`: Visual summary

## Notes
- Optimized for fluorescence microscopy images
- Supports batch processing
- Includes automatic scale bar generation
- Provides statistical analysis
