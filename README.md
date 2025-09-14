# FreshStepCounter

A Python-based step detection system that analyzes 3D ankle motion data using Principal Component Analysis (PCA) and signal processing techniques to accurately count walking steps.

## Overview

FreshStepCounter processes motion capture data from ankle tracking to detect individual steps by:
1. Extracting 3D coordinates for left and right ankles
2. Applying PCA to reduce dimensionality and capture primary motion patterns
3. Cleaning signals by removing non-stationarity and trends
4. Detecting step events through zero-crossing analysis of the difference signal

## Features

- **Robust Data Cleaning**: Handles missing values, zeros, and noise in motion data
- **PCA-based Analysis**: Uses Principal Component Analysis to extract dominant motion patterns
- **Advanced Signal Processing**: Removes trends, drift, and non-stationarity from signals
- **Zero-Crossing Detection**: Identifies step events through intersection analysis
- **Interactive Visualizations**: Generates detailed HTML plots showing the entire analysis pipeline
- **Comprehensive Reporting**: Provides step counts and timing analysis

## Requirements

```python
pandas  
numpy
plotly
scikit-learn
scipy
```

Install dependencies:
```bash
pip install pandas numpy plotly scikit-learn scipy
```

## Input Data Format

The system expects CSV files with the following columns:
- `Left AnkleX`: Left ankle X-coordinate
- `Left AnkleY`: Left ankle Y-coordinate  
- `Left AnkleDepth`: Left ankle Z-coordinate (depth)
- `Right AnkleX`: Right ankle X-coordinate
- `Right AnkleY`: Right ankle Y-coordinate
- `Right AnkleDepth`: Right ankle Z-coordinate (depth)

Data should be captured at 25 FPS (configurable via `self.fps` parameter).

## Usage

### Basic Usage

```python
from fresh_step_counter import analyze_steps

# Analyze steps from CSV file
total_steps = analyze_steps('your_data.csv', 'results.html')
print(f"Steps detected: {total_steps}")
```

### Advanced Usage

```python
from fresh_step_counter import FreshStepCounter

# Create counter instance
counter = FreshStepCounter()

# Run step-by-step analysis
steps = (counter
         .load_csv('your_data.csv')
         .extract_coordinates()
         .compute_pca()
         .remove_nonstationarity()
         .find_intersections()
         .create_plots('output.html')
         .print_summary())

# Access results
print(f"Total steps: {counter.total_steps}")
print(f"Step frames: {counter.step_frames}")
```

## Algorithm Details

### 1. Data Preprocessing
- Loads CSV data and cleans column names
- Handles missing values using forward-fill, backward-fill, and mean imputation
- Converts zero values to NaN for proper cleaning

### 2. PCA Analysis
- Applies PCA to 3D ankle coordinates for each foot
- Extracts the first principal component (PC1) capturing dominant motion
- PC1 represents the primary direction of ankle movement

### 3. Non-Stationarity Removal
- **Linear Detrending**: Removes linear trends from signals
- **Quadratic Detrending**: Eliminates quadratic trends
- **High-pass Filtering**: Removes slow drift (< 0.1 Hz) using Butterworth filter

### 4. Step Detection
- Computes difference signal (Left PC1 - Right PC1)
- Applies median filtering (size=9) to smooth the difference signal
- Detects zero-crossings as step events
- Each crossing represents alternating foot contact

### 5. Visualization
Creates a comprehensive 4-panel plot showing:
- Raw PC1 signals for both ankles
- Cleaned PC1 signals after preprocessing
- Difference signal with detected zero-crossings
- Final results with step markers

## Output

### Console Output
```
Loading CSV: your_data.csv
Loaded 400 frames
==================================================
EXTRACTING ANKLE COORDINATES
COMPUTING PCA
REMOVING NON-STATIONARITY
FINDING PC1 INTERSECTIONS
CREATING PLOTS: results.html
Plot saved: results.html
FINAL SUMMARY
==================================================
Video duration: 16.0 seconds
Total steps detected: 12
==================================================
```

### HTML Visualization
- Interactive Plotly graphs showing the complete analysis pipeline
- Time-series plots with step markers
- Hover tooltips for detailed data inspection

## Configuration

### Frame Rate
```python
counter = FreshStepCounter()
counter.fps = 30.0  # Change from default 25 FPS
```

### Filter Parameters
Modify in the `remove_nonstationarity()` method:
- `cutoff_freq`: High-pass filter cutoff (default: 0.1 Hz)
- `butter()` parameters: Filter order and type

### Smoothing Parameters
Adjust in `find_intersections()`:
- `median_filter()` size: Window size for smoothing (default: 9)

## Validation and Accuracy

The system is designed for walking gait analysis and works best with:
- Clear alternating foot motion patterns
- Consistent walking pace
- Good quality motion capture data
- Minimal occlusion or tracking errors

## Limitations

- Optimized for normal walking gaits
- May require parameter tuning for different movement patterns
- Assumes 25 FPS data by default
- Requires both left and right ankle data

## Troubleshooting

### Common Issues

**No steps detected:**
- Check if ankle coordinates are valid (non-zero, non-NaN)
- Verify data contains actual walking motion
- Adjust filter parameters for different movement speeds

**Too many/few steps:**
- Modify median filter size in `find_intersections()`
- Adjust high-pass filter cutoff frequency
- Check frame rate setting matches your data

**Poor signal quality:**
- Ensure input data has minimal noise
- Verify coordinate system consistency
- Check for tracking occlusions or errors

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Citation

If you use this algorithm in your research, please cite:
```bibtex
@software{fresh_step_counter,
  title={FreshStepCounter: PCA-based Step Detection for 3D Pose Estimation Data},
  author={[Dmitrii Smirnov]},
  year={2024},
  url={https://github.com/Dmitrii-Smirnov/DeepTrack3D}
}
```

## Related Work

This algorithm is designed to work with:
- [DeepTrack3D](https://github.com/Dmitrii-Smirnov/DeepTrack3D) - 3D pose estimation system
