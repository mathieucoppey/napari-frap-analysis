# Napari FRAP Analysis Plugin

A robust Napari plugin for performing **Fluorescence Recovery After Photobleaching (FRAP)** analysis on single-cell movies. This tool provides an end-to-end workflow from nucleus segmentation to intensity tracking and curve fitting.

## Features

-   **Nucleus Segmentation**: Automatically segment nuclei in 3D (2D+Time) using Otsu thresholding and morphological operations.
-   **FRAP ROI Tracking**: Automatically track the movement of the nucleus and adjust the FRAP ROI position/rotation accordingly for accurate intensity measurement over time.
-   **Batch Processing**: Efficiently computes metrics across all frames.
-   **Curve Fitting**:
    -   Models: Single or Double Diffusive populations.
    -   Immobile Fraction support.
    -   Customizable parameter bounds.
-   **Results Export**: Save curves and fitting parameters to CSV. "Append" mode allows aggregating results from multiple movies into a single dataset.

## Installation

### Prerequisites
-   Python 3.9+
-   Napari

### Quick Start
1.  **Create a conda environment**:
    ```bash
    conda create -n napari_frap python=3.9
    conda activate napari_frap
    ```

2.  **Install dependencies**:
    ```bash
    pip install napari[all] matplotlib scipy qtpy scikit-image
    ```

3.  **Install the plugin**:
    Navigate to the plugin directory and install in editable mode:
    ```bash
    cd path/to/napari-frap-analysis
    pip install -e .
    ```

## Usage Workflow

1.  **Open Napari**:
    ```bash
    napari
    ```
2.  **Launch Plugin**: Go to `Plugins` > `napari-frap-analysis` > `FrapWidget`.
3.  **Load Data**: Drag and drop your `.tif` movie into Napari. Ensure it is selected in the "Image Layer" dropdown.

### 1. Segmentation (Optional but Recommended)
Switch to the **"Segmentation"** tab to automatically find the nucleus.
-   Adjust **Gaussian Sigma** (blur), **Threshold Factor** (sensitivity), and **Closing/Opening** radii.
-   Enable **"Live Preview"** to see the mask on the current frame.
-   Click **"Segment Nucleus (All Frames)"** to generate the `Nucleus Labels` layer.

### 2. Define ROIs
Switch to the **"Analysis"** tab.
-   **Background**: Click "Add ROI Layers" (if needed) and draw a rectangle in the `Background` layer (blue).
-   **FRAP Region**: Draw an ellipse/circle in the `FRAP` layer (red) over the bleached spot.
-   **Nucleus**:
    -   If you ran Segmentation (`Nucleus Labels` exists), you can skip drawing a manual nucleus ROI.
    -   Otherwise, draw a polygon in the `Nucleus` layer (green).

### 3. Track FRAP ROI (Critical for Moving Cells)
If the cell moves during the movie, the FRAP ROI needs to follow it.
-   Ensure **Nucleus Labels** are generated (Step 1).
-   Check **"Track FRAP ROI with Nucleus"**.
-   **Wait** for the computation to finish. The plugin will calculate the nucleus rotation and translation for every frame and update the FRAP ROI in 3D.
-   *Note*: You can uncheck this to restore the original manual ROI.

### 4. Analysis & Fitting
-   **Parameters**: Set the **Time Interval** (sec/frame) and **Bleach Frame** index.
-   **Fitting**:
    -   Choose Model: `1 Diffusive Population` or `2 Diffusive Populations`.
    -   Check `Immobile Fraction` to include it in the fit.
    -   Adjust **Min/Max bounds** for parameters if the fit is unstable.
-   The Data and Fit curves are plotted in real-time.

### 5. Export Results
-   **Save New**: Saves `_curves.csv` (time traces) and `_params.csv` (fit results).
-   **Append to**: Selects an existing `_params.csv` to append the current movie's results to. Ideal for processing multiple cells into one master file.

## License
BSD-3-Clause
