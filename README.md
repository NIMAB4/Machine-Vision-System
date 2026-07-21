# Comprehensive Machine Vision & Image Processing System

A full-featured desktop application for **machine vision and image processing**, built with Python using **Tkinter**, **OpenCV**, **NumPy**, **Pillow**, and **Matplotlib**. The application provides an interactive GUI where users can load images, apply a wide range of image processing algorithms, visualize results side-by-side, and inspect histograms — all organized into four academic sections.

---

## Table of Contents

- [Features](#features)
- [Screenshots & UI Layout](#screenshots--ui-layout)
- [Installation](#installation)
- [Usage](#usage)
- [Project Architecture](#project-architecture)
- [Algorithms & Filters Reference](#algorithms--filters-reference)
  - [Section 1: Fundamentals, Interpolation & Shapes](#section-1-fundamentals-interpolation--shapes)
  - [Section 2: Spatial Domain Enhancement](#section-2-spatial-domain-enhancement)
  - [Section 3: Frequency Domain Filtering](#section-3-frequency-domain-filtering)
  - [Section 4: Restoration, Segmentation & Color](#section-4-restoration-segmentation--color)
- [Test Patterns](#test-patterns)
- [Dependencies](#dependencies)
- [License](#license)

---

## Features

- **Side-by-side display** of original and processed images on interactive canvases.
- **50+ image processing algorithms** organized into 4 academic sections.
- **Dynamic parameter sliders** that adapt to the selected algorithm (kernel size, thresholds, sigma, gamma, cutoff frequency, filter order, etc.).
- **Histogram visualization** for both grayscale and color (RGB) images using Matplotlib.
- **Undo / Reset** support with full processing history stack.
- **Commit step** feature for chaining multiple operations sequentially.
- **Built-in test patterns** (grid/frequency lines, noisy coins, brightness gradient) for experimentation without loading external images.
- **Image statistics panel** showing dimensions, channels, data type, mean intensity, and standard deviation.
- **Save processed images** in JPEG or PNG format.
- **Persian-friendly UI** with RTL-compatible labels and Tahoma font.

---

## Screenshots & UI Layout

The application window is divided into:

| Section | Description |
|---|---|
| **Top Control Bar** | Buttons for loading, saving, committing, undo, reset, histogram display, and test patterns. |
| **Left Sidebar** | Section/category selector, algorithm selector, two dynamic parameter sliders, an apply button, and an image statistics panel. |
| **Right Display Area** | Two side-by-side canvases showing the original input image and the processed result. |

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Steps

```bash
# Clone the repository
git clone https://github.com/NIMAB4/Comprehensive-Machine-Vision-Image-Processing-System.git
cd Comprehensive-Machine-Vision-Image-Processing-System

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

---

## Usage

1. **Launch** the application by running `python main.py`.
2. **Load an image** using the "Load Image" button, or select a built-in **test pattern**.
3. **Select a section** from the category dropdown to filter relevant algorithms.
4. **Select an algorithm** from the algorithm dropdown.
5. **Adjust parameters** using the dynamic sliders (labels update automatically based on the selected algorithm).
6. **Click "Apply Filter/Algorithm"** to process the image.
7. **Commit the step** if you want to chain another operation on top of the current result.
8. **Undo** to revert to the previous state, or **Reset** to return to the default pattern.
9. **View the histogram** of the processed image at any time.
10. **Save** the final processed image as JPEG or PNG.

---

## Project Architecture

```
├── main.py               # Main application source code
├── main_old.py            # Previous version (backup)
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation (English)
└── README-FA.md           # Project documentation (Persian)
```

The entire application is contained in a single Python file (`main.py`) structured as follows:

- **`AdvancedVisionApp` class** — The main application class that encapsulates:
  - `create_layout()` — Builds the entire Tkinter GUI (top bar, sidebar, canvases).
  - `load_image()` / `save_image()` — File I/O for loading and saving images.
  - `apply_algorithm()` — Central dispatcher that routes to the correct image processing operation based on user selection.
  - `show_images()` / `display_on_canvas()` — Rendering logic that scales and displays images on Tkinter canvases using PIL.
  - `show_histogram()` — Opens a Matplotlib-powered histogram window.
  - `commit_to_base()` / `undo_action()` / `reset_image()` — History management for chaining and reverting operations.
  - `update_info()` — Computes and displays image statistics (dimensions, channels, mean, std).
  - `reset_params()` — Dynamically reconfigures slider ranges and labels based on the selected algorithm.
  - `load_default_pattern()` / `show_patterns_dialog()` — Generates built-in geometric and mathematical test patterns.

---

## Algorithms & Filters Reference

### Section 1: Fundamentals, Interpolation & Shapes

#### Grayscale Conversion
- **Function:** `cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)`
- **Description:** Converts a color (RGB) image to a single-channel grayscale image using the standard luminosity formula: `Y = 0.299R + 0.587G + 0.114B`. The result is converted back to a 3-channel image for uniform display.
- **Parameters:** None.

#### Nearest Neighbor Resizing
- **Function:** `cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)`
- **Description:** Resizes the image using **nearest neighbor interpolation**. Each pixel in the output image is assigned the value of the nearest pixel in the input image. This is the fastest interpolation method but may produce blocky artifacts (aliasing) when upscaling.
- **Parameters:**
  - **Scale Percentage** (Slider 1, range 10–300%): The resize factor as a percentage of the original size.

#### Bilinear Resizing
- **Function:** `cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)`
- **Description:** Resizes the image using **bilinear interpolation**. Computes the output pixel value as a weighted average of the four nearest input pixels (2×2 neighborhood). Produces smoother results than nearest neighbor with moderate computational cost.
- **Parameters:**
  - **Scale Percentage** (Slider 1, range 10–300%): The resize factor as a percentage of the original size.

#### Bicubic Resizing
- **Function:** `cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)`
- **Description:** Resizes the image using **bicubic interpolation**. Computes the output pixel value as a weighted average of 16 nearest input pixels (4×4 neighborhood) using a cubic polynomial. Produces the smoothest results among the three interpolation methods but is computationally most expensive.
- **Parameters:**
  - **Scale Percentage** (Slider 1, range 10–300%): The resize factor as a percentage of the original size.

#### Euclidean Distance Transform
- **Function:** `cv2.distanceTransform(thresh, cv2.DIST_L2, 5)`
- **Description:** Computes the **Euclidean distance transform** of a binary image. For each foreground pixel, calculates its distance to the nearest background pixel using the L2 (Euclidean) norm: `D(p) = √((x₁-x₂)² + (y₁-y₂)²)`. The result is normalized to [0, 255] for visualization.
- **Parameters:**
  - **Binary Threshold** (Slider 1, range 0–255): Threshold value for binarizing the input image.

#### City Block Distance (D4)
- **Function:** `cv2.distanceTransform(thresh, cv2.DIST_L1, 3)`
- **Description:** Computes the **City Block (Manhattan) distance transform** using the L1 norm: `D₄(p) = |x₁-x₂| + |y₁-y₂|`. Also known as **D4 distance** because each pixel has 4 neighbors at unit distance. The result is normalized to [0, 255].
- **Parameters:**
  - **Binary Threshold** (Slider 1, range 0–255): Threshold value for binarizing the input image.

#### Chessboard Distance (D8)
- **Function:** `cv2.distanceTransform(thresh, cv2.DIST_C, 3)`
- **Description:** Computes the **Chessboard (Chebyshev) distance transform** using the L∞ norm: `D₈(p) = max(|x₁-x₂|, |y₁-y₂|)`. Also known as **D8 distance** because each pixel has 8 neighbors at unit distance (including diagonals). The result is normalized to [0, 255].
- **Parameters:**
  - **Binary Threshold** (Slider 1, range 0–255): Threshold value for binarizing the input image.

#### Find Contours
- **Function:** `cv2.findContours()` + `cv2.drawContours()`
- **Description:** Detects the boundaries (contours) of objects in a binary image. The image is first converted to grayscale, then binarized using `cv2.THRESH_BINARY_INV` at threshold 127. Contours are extracted using `RETR_EXTERNAL` mode (only outermost contours) with `CHAIN_APPROX_SIMPLE` approximation. Detected contours are drawn in green on the original image.
- **Parameters:** None.

#### Hough Line Transform
- **Function:** `cv2.HoughLines()`
- **Description:** Detects straight lines using the **Standard Hough Transform** in the parameter space (ρ, θ). The process involves: (1) Canny edge detection with thresholds 50 and 150, (2) transformation to Hough space with ρ resolution = 1 pixel and θ resolution = π/180 radians, (3) drawing lines with accumulator votes above the threshold (limited to 15 lines maximum).
- **Parameters:**
  - **Accumulator Threshold** (Slider 1, range 20–150): Minimum votes required to detect a line.

---

### Section 2: Spatial Domain Enhancement

#### Intensity Transformations

##### Negative (Image Inversion)
- **Formula:** `s = 255 - r`
- **Description:** Inverts all pixel intensities, producing a photographic negative. Useful for enhancing white or light details embedded in dark regions.
- **Parameters:** None.

##### Log Transformation
- **Formula:** `s = c · log(1 + r)`, where `c = 255 / log(1 + max_val)`
- **Description:** Applies a logarithmic transformation to compress the dynamic range. Maps a narrow range of low-intensity values to a wider output range, and compresses higher-intensity values. Useful for displaying Fourier spectra and enhancing dark regions.
- **Parameters:** None.

##### Gamma Correction (Power-Law Transformation)
- **Formula:** `s = 255 · (r / 255)^γ`
- **Description:** Applies a power-law transformation using a lookup table (LUT) for efficiency. γ < 1 brightens the image; γ = 1 is identity; γ > 1 darkens the image.
- **Parameters:**
  - **Gamma value** (Slider 1, range 1–50, divided by 10): The gamma exponent (e.g., slider value 10 = γ = 1.0).

##### Histogram Equalization
- **Function:** `cv2.equalizeHist()`
- **Description:** Enhances contrast by redistributing pixel intensities so the output histogram is approximately uniform. For color images, converts to **YUV** and equalizes only the **Y (luminance)** channel. Based on the **Cumulative Distribution Function (CDF)**: `s = T(r) = (L-1) · CDF(r)`.
- **Parameters:** None.

##### Histogram Matching (Specification)
- **Description:** Matches the image histogram to a user-defined **target Gaussian distribution**. Computes the CDF of both the input image and the target distribution, then builds a lookup table (LUT) that maps each input intensity to the corresponding output intensity so the output histogram approximates the specified distribution.
- **Process:**
  1. Compute CDF of the input grayscale image.
  2. Generate a target Gaussian PDF with user-specified mean and standard deviation.
  3. Compute CDF of the target distribution.
  4. Build a mapping LUT by matching CDF values.
  5. Apply the LUT to transform the image.
- **Parameters:**
  - **Target Mean** (Slider 1, range 10–240): Mean of the target Gaussian distribution.
  - **Target Std Dev** (Slider 2, range 5–100): Standard deviation of the target distribution.

##### Local Histogram Equalization (CLAHE)
- **Function:** `cv2.createCLAHE(clipLimit, tileGridSize)`
- **Description:** **Contrast Limited Adaptive Histogram Equalization** — performs histogram equalization locally within small tiles (sub-regions) of the image rather than globally. A **clip limit** prevents over-amplification of noise by limiting the histogram bin heights before computing the CDF. For color images, operates on the Y channel in YUV space.
- **Parameters:**
  - **Grid Size** (Slider 1, range 2–32): Size of the local tile grid (e.g., 8 means 8×8 tiles).
  - **Clip Limit** (Slider 2, range 5–100, divided by 10): Contrast threshold for clipping.

#### Spatial Smoothing Filters

##### Gaussian Blur
- **Function:** `cv2.GaussianBlur(img, (k, k), sigma)`
- **Kernel Formula:** `G(x, y) = (1 / 2πσ²) · exp(-(x² + y²) / 2σ²)`
- **Description:** Applies a Gaussian low-pass filter. Kernel weights decrease with distance from the center following a bell-curve distribution. Effectively reduces Gaussian noise while preserving edges better than a box filter.
- **Parameters:**
  - **Kernel Size** (Slider 1, range 3–25, must be odd): Size of the convolution kernel.
  - **Sigma** (Slider 2, range 1–100, divided by 10): Standard deviation of the Gaussian.

##### Median Blur
- **Function:** `cv2.medianBlur(img, k)`
- **Description:** A non-linear filter that replaces each pixel with the **median** of its neighborhood. Extremely effective at removing **salt-and-pepper noise** while preserving edges.
- **Parameters:**
  - **Kernel Size** (Slider 1, range 3–25, must be odd).

##### Average Blur (Box Filter)
- **Function:** `cv2.blur(img, (k, k))`
- **Description:** Replaces each pixel with the arithmetic mean of its neighborhood. Equivalent to convolution with a kernel where all coefficients equal `1/k²`.
- **Parameters:**
  - **Kernel Size** (Slider 1, range 3–25, must be odd).

##### Bilateral Filter
- **Function:** `cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)`
- **Description:** An edge-preserving smoothing filter that considers both spatial distance and intensity difference. Smooths flat regions while keeping edges sharp.
- **Parameters:**
  - **Spatial Sigma (σ_s)** (Slider 1, range 5–50).
  - **Color Sigma (σ_r)** (Slider 2, range 5–100).

#### Edge Detection Filters

##### Laplacian
- **Function:** `cv2.Laplacian(gray, cv2.CV_64F)`
- **Description:** A second-order derivative operator: `∇²f = ∂²f/∂x² + ∂²f/∂y²`. Isotropic, detects edges in all directions simultaneously. Sensitive to noise.
- **Parameters:** None.

##### Sobel X (Horizontal Edge Detection)
- **Function:** `cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)`
- **Description:** First-order derivative in the x-direction. Detects vertical edges. Kernel: `[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]`.
- **Parameters:** None.

##### Sobel Y (Vertical Edge Detection)
- **Function:** `cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)`
- **Description:** First-order derivative in the y-direction. Detects horizontal edges. Kernel: `[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]`.
- **Parameters:** None.

##### Sobel Combined
- **Function:** `cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)`
- **Description:** Combines Sobel X and Sobel Y with weighted sum: `|G| = 0.5·|Gx| + 0.5·|Gy|`.
- **Parameters:** None.

##### Robert Cross Edge Detector
- **Description:** A classic **2×2 diagonal difference** edge detector. Uses two kernels that compute gradients along the two diagonal directions:
  ```
  Gx = [[1, 0], [0, -1]]    Gy = [[0, 1], [-1, 0]]
  ```
  The edge magnitude is computed as: `|G| = √(Gx² + Gy²)`.
- **Parameters:** None.

##### Canny Edge Detector
- **Function:** `cv2.Canny(gray, threshold1, threshold2)`
- **Description:** Multi-stage optimal edge detector (John F. Canny, 1986): (1) Gaussian smoothing, (2) gradient computation via Sobel, (3) non-maximum suppression, (4) hysteresis thresholding with two thresholds.
- **Parameters:**
  - **Low Threshold** (Slider 1, range 1–150).
  - **High Threshold** (Slider 2, range 10–250).

##### Fuzzy Contrast Enhancement
- **Description:** Enhances image contrast using a **fuzzy S-curve membership function**. The algorithm:
  1. Normalizes pixel intensities to membership values μ ∈ [0, 1].
  2. Applies a quadratic intensification operator:
     - If μ ≤ 0.5: `μ' = 2μ²`
     - If μ > 0.5: `μ' = 1 - 2(1-μ)²`
  3. Maps back to the original intensity range.
  This stretches contrast by pushing values away from 0.5 toward 0 or 1.
- **Parameters:** None.

#### Morphological Operations

All morphological operations use a **square structuring element** of user-defined size.

##### Erosion
- **Function:** `cv2.erode(img, kernel, iterations=1)`
- **Description:** Replaces each pixel with the **minimum** in its neighborhood. Objects shrink, gaps widen.

##### Dilation
- **Function:** `cv2.dilate(img, kernel, iterations=1)`
- **Description:** Replaces each pixel with the **maximum** in its neighborhood. Objects grow, gaps fill.

##### Opening (Erosion → Dilation)
- **Function:** `cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)`
- **Description:** Removes small bright noise while preserving larger object shapes.

##### Closing (Dilation → Erosion)
- **Function:** `cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)`
- **Description:** Fills small dark holes within bright objects.

##### Morphological Gradient
- **Function:** `cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)`
- **Description:** Difference between dilation and erosion: `Gradient = Dilation(f) - Erosion(f)`. Produces object boundaries.

**Common Parameter:**
- **Structuring Element Size** (Slider 1, range 3–15, must be odd).

#### Thresholding / Segmentation

##### Global Thresholding
- **Function:** `cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)`
- **Description:** Binarizes the image with a single global threshold.
- **Parameters:**
  - **Threshold Value** (Slider 1, range 0–255).

##### Adaptive Mean Threshold
- **Function:** `cv2.adaptiveThreshold(..., cv2.ADAPTIVE_THRESH_MEAN_C, ...)`
- **Description:** Local threshold based on the **mean** of a neighborhood block: `T(x,y) = mean(neighborhood) - C`.
- **Parameters:**
  - **Block Size** (Slider 1, range 3–45, must be odd).
  - **Constant C** (Slider 2, range -20 to 40).

##### Adaptive Gaussian Threshold
- **Function:** `cv2.adaptiveThreshold(..., cv2.ADAPTIVE_THRESH_GAUSSIAN_C, ...)`
- **Description:** Local threshold using a **Gaussian-weighted sum** of the neighborhood. Produces smoother results than mean-based.
- **Parameters:**
  - **Block Size** (Slider 1, range 3–45, must be odd).
  - **Constant C** (Slider 2, range -20 to 40).

##### Otsu's Thresholding
- **Function:** `cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)`
- **Description:** Automatic global thresholding that finds the optimal threshold by minimizing **intra-class variance** (or maximizing **inter-class variance**). No manual parameter needed.
- **Parameters:** None (automatically determined).

---

### Section 3: Frequency Domain Filtering

#### FFT Spectrum (Fourier Transform Magnitude Spectrum)
- **Function:** `np.fft.fft2()` + `np.fft.fftshift()`
- **Description:** Computes the 2D Discrete Fourier Transform and displays the log-scaled magnitude spectrum: `S = 20 · log(|F(u,v)| + 1)`. The DC component is shifted to the center.
- **Parameters:** None.

#### Ideal Low-Pass Filter
- **Transfer Function:** `H(u,v) = 1 if D(u,v) ≤ D0, else 0`
- **Description:** Binary circular mask that passes all frequencies within radius D0 and blocks everything outside. Applied in the frequency domain via FFT → multiply → inverse FFT.
- **Parameters:**
  - **Cutoff Radius D0** (Slider 1, range 5–60).

#### Ideal High-Pass Filter
- **Transfer Function:** `H(u,v) = 0 if D(u,v) ≤ D0, else 1`
- **Description:** Complement of the ideal low-pass filter. Blocks low frequencies, passes high frequencies to isolate edges and fine details.
- **Parameters:**
  - **Cutoff Radius D0** (Slider 1, range 5–60).

#### Butterworth Low-Pass Filter
- **Transfer Function:** `H(u,v) = 1 / (1 + (D(u,v) / D0)^(2n))`
- **Description:** A smooth low-pass filter without the sharp cutoff ringing artifacts of the ideal filter. The **order n** controls the steepness of the transition — higher orders approach the ideal filter's sharp cutoff.
- **Parameters:**
  - **Cutoff Frequency D0** (Slider 1, range 5–200).
  - **Filter Order n** (Slider 2, range 1–10).

#### Butterworth High-Pass Filter
- **Transfer Function:** `H(u,v) = 1 / (1 + (D0 / D(u,v))^(2n))`
- **Description:** Complement of the Butterworth low-pass. Smoothly attenuates low frequencies while passing high frequencies.
- **Parameters:**
  - **Cutoff Frequency D0** (Slider 1, range 5–200).
  - **Filter Order n** (Slider 2, range 1–10).

#### Gaussian Low-Pass Filter (Frequency Domain)
- **Transfer Function:** `H(u,v) = exp(-D²(u,v) / (2·D0²))`
- **Description:** A frequency domain low-pass filter with a smooth Gaussian profile. No ringing artifacts. The transition from passband to stopband follows a Gaussian curve.
- **Parameters:**
  - **Cutoff Frequency D0** (Slider 1, range 5–200).

#### Gaussian High-Pass Filter (Frequency Domain)
- **Transfer Function:** `H(u,v) = 1 - exp(-D²(u,v) / (2·D0²))`
- **Description:** Complement of the Gaussian low-pass. Smoothly isolates high-frequency content (edges, textures).
- **Parameters:**
  - **Cutoff Frequency D0** (Slider 1, range 5–200).

#### High-Frequency Emphasis Filtering
- **Transfer Function:** `H_hfe(u,v) = 0.5 + k · H_hp(u,v)`
- **Description:** Combines a **Gaussian high-pass filter** with a DC offset to retain some low-frequency background while emphasizing edges and fine details. The constant 0.5 preserves the overall image brightness, while the factor `k` controls the degree of high-frequency emphasis.
- **Parameters:**
  - **Cutoff Frequency D0** (Slider 1, range 5–200).
  - **Emphasis Factor k** (Slider 2, range 5–50, divided by 10).

#### Homomorphic Filtering
- **Description:** Addresses **non-uniform illumination** by operating in the log domain. The image is modeled as `f(x,y) = i(x,y) · r(x,y)` (illumination × reflectance). Taking the log separates them additively. A frequency domain filter then compresses the illumination (low-frequency) component and enhances the reflectance (high-frequency) component.
- **Process:**
  1. Compute `log(1 + image)`.
  2. Apply 2D FFT.
  3. Multiply by filter: `H(u,v) = (γH - γL) · (1 - exp(-D²/(2D0²))) + γL` with `γL = 0.25`, `γH = 1.5`.
  4. Inverse FFT.
  5. Compute `exp() - 1` to return to the spatial domain.
- **Parameters:**
  - **Cutoff Frequency D0** (Slider 1, range 5–200).

#### Ideal Bandpass Filter
- **Description:** Passes only frequencies within a band `[D0 - W/2, D0 + W/2]`, blocking everything else. Useful for isolating specific frequency components.
- **Parameters:**
  - **Center Frequency D0** (Slider 1, range 5–200).
  - **Bandwidth W** (Slider 2, range 2–100).

#### Ideal Bandreject Filter
- **Description:** Complement of the ideal bandpass. Blocks frequencies within the band `[D0 - W/2, D0 + W/2]` and passes everything else. Useful for removing periodic noise at known frequencies.
- **Parameters:**
  - **Center Frequency D0** (Slider 1, range 5–200).
  - **Bandwidth W** (Slider 2, range 2–100).

#### Butterworth Bandpass Filter
- **Transfer Function:** `H_bp = 1 - H_br`, where `H_br = 1 / (1 + ((D·W) / (D²-D0²))^(2n))`
- **Description:** A smooth bandpass filter derived as the complement of the Butterworth bandreject filter. The Butterworth profile eliminates ringing artifacts present in ideal filters.
- **Parameters:**
  - **Center Frequency D0** (Slider 1, range 5–200).
  - **Bandwidth W** (Slider 2, range 2–100).

#### Butterworth Bandreject Filter
- **Transfer Function:** `H(u,v) = 1 / (1 + ((D·W) / (D²-D0²))^(2n))`
- **Description:** Smoothly rejects a band of frequencies centered at D0 with bandwidth W. The Butterworth shape provides a gradual transition.
- **Parameters:**
  - **Center Frequency D0** (Slider 1, range 5–200).
  - **Bandwidth W** (Slider 2, range 2–100).

#### Ideal Notch Reject Filter
- **Description:** Removes a specific frequency component and its conjugate symmetric counterpart from the spectrum. The algorithm automatically detects the **dominant frequency peak** (excluding DC) and places circular reject regions of radius D0 around it and its symmetric point.
- **Transfer Function:** `H = 0` where `D1 ≤ D0` or `D2 ≤ D0`, else `H = 1`.
- **Parameters:**
  - **Notch Radius D0** (Slider 1, range 2–50).

#### Butterworth Notch Reject Filter
- **Transfer Function:** `H(u,v) = 1 / (1 + (D0² / (D1·D2))^n)`
- **Description:** A smooth notch reject filter that suppresses a specific frequency pair with a Butterworth profile, avoiding ringing artifacts.
- **Parameters:**
  - **Notch Radius D0** (Slider 1, range 2–50).
  - **Filter Order n** (Slider 2, range 1–10).

#### Gaussian Notch Reject Filter
- **Transfer Function:** `H(u,v) = (1 - exp(-D1²/(2D0²))) · (1 - exp(-D2²/(2D0²)))`
- **Description:** A Gaussian-profile notch reject filter that smoothly attenuates a specific frequency pair without ringing.
- **Parameters:**
  - **Notch Radius D0** (Slider 1, range 2–50).

---

### Section 4: Restoration, Segmentation & Color

#### Noise Generation

##### Add Salt & Pepper Noise
- **Description:** Randomly sets a percentage of pixels to pure white (255, "salt") or pure black (0, "pepper") with a 50/50 split. Simulates sensor failures or transmission errors.
- **Parameters:**
  - **Noise Percentage** (Slider 1, range 1–40%).

##### Add Gaussian Noise
- **Formula:** `noisy(x,y) = image(x,y) + N(0, σ)`
- **Description:** Adds random values from a normal distribution to each pixel. Simulates electronic sensor noise.
- **Parameters:**
  - **Sigma** (Slider 1, range 5–100).

#### Restoration Filters (Order-Statistics & Adaptive)

##### Geometric Mean Filter
- **Description:** Computes the **geometric mean** of pixel values in the neighborhood. Implemented as: `exp(mean(log(pixel + 1))) - 1`. Achieves smoothing comparable to the arithmetic mean but tends to lose fewer details. Effective for **Gaussian noise** reduction.
- **Parameters:**
  - **Kernel Size** (Slider 1, range 3–15, must be odd).

##### Harmonic Mean Filter
- **Formula:** `f̂(x,y) = (m·n) / Σ(1/g(s,t))`
- **Description:** Computes the **harmonic mean** of the neighborhood: the reciprocal of the mean of reciprocals. Works well for **salt noise** (white spikes) but fails for pepper noise.
- **Parameters:**
  - **Kernel Size** (Slider 1, range 3–15, must be odd).

##### Contraharmonic Mean Filter
- **Formula:** `f̂(x,y) = Σg(s,t)^(Q+1) / Σg(s,t)^Q`
- **Description:** A generalized mean filter controlled by the **order Q**. Positive Q eliminates **pepper noise**; negative Q eliminates **salt noise**. At Q = 0, reduces to the arithmetic mean; at Q = -1, reduces to the harmonic mean.
- **Parameters:**
  - **Kernel Size** (Slider 1, range 3–15, must be odd).
  - **Order Q** (Slider 2, range -50 to 50, divided by 10).

##### Max Filter
- **Function:** `cv2.dilate(img, kernel)`
- **Description:** Replaces each pixel with the **maximum** value in its neighborhood. Effectively a grayscale dilation. Best for removing **pepper noise** (dark spots).
- **Parameters:**
  - **Window Size** (Slider 1, range 3–15, must be odd).

##### Min Filter
- **Function:** `cv2.erode(img, kernel)`
- **Description:** Replaces each pixel with the **minimum** value in its neighborhood. Effectively a grayscale erosion. Best for removing **salt noise** (bright spots).
- **Parameters:**
  - **Window Size** (Slider 1, range 3–15, must be odd).

##### Midpoint Filter
- **Formula:** `f̂(x,y) = (max + min) / 2`
- **Description:** Computes the average of the maximum and minimum values in the neighborhood. Works best for **uniformly distributed** and **Gaussian** noise.
- **Parameters:**
  - **Window Size** (Slider 1, range 3–15, must be odd).

##### Alpha-Trimmed Mean Filter
- **Description:** Sorts all pixels in the neighborhood window, removes the `d/2` lowest and `d/2` highest values, and computes the mean of the remaining pixels. When `d = 0`, it becomes an arithmetic mean filter; when `d = m·n - 1`, it becomes a median filter. Best for **combined** (mixed) noise types such as salt-and-pepper combined with Gaussian.
- **Parameters:**
  - **Neighborhood Size** (Slider 1, range 3–15, must be odd).
  - **Trimming Count d** (Slider 2, range 0–14): Number of extreme values to discard.

##### Adaptive Local Noise Reduction Filter
- **Description:** An adaptive filter that adjusts its behavior pixel-by-pixel based on the **local statistics** (mean and variance) compared to the overall noise variance. The filter applies: `f̂(x,y) = g(x,y) - (σ²_noise / σ²_local) · (g(x,y) - μ_local)`. If local variance is high (edge), less smoothing is applied; if low (flat region), more smoothing is applied.
- **Parameters:**
  - **Window Size** (Slider 1, range 3–15, must be odd).
  - **Noise Variance** (Slider 2, range 10–1000): Estimated global noise variance.

##### Adaptive Median Filter
- **Description:** An advanced median filter with a **variable window size**. For each pixel, the algorithm:
  1. Starts with a 3×3 window.
  2. Checks if the median is an impulse (equal to min or max).
  3. If yes, increases the window size by 2 and repeats.
  4. If the median is valid, checks if the pixel itself is an impulse and replaces it with the median if so.
  5. Stops at the user-defined maximum window size.
  This makes it far more effective than standard median filtering for **high-density salt-and-pepper noise**.
- **Parameters:**
  - **Maximum Window Size** (Slider 1, range 3–21, must be odd).

#### Morphological Operations

(Same as in Section 2 — Erosion, Dilation, Opening, Closing, Morphological Gradient)

#### Thresholding / Segmentation

(Same as in Section 2 — Global, Adaptive Mean, Adaptive Gaussian, Otsu's)

#### Color Space Conversions

##### RGB to HSV
- **Function:** `cv2.cvtColor(img, cv2.COLOR_RGB2HSV)`
- **Description:** Converts to **HSV (Hue, Saturation, Value)** color space. H (0–180 in OpenCV) represents color type, S (0–255) represents purity, V (0–255) represents brightness.
- **Parameters:** None.

##### RGB to LAB (CIELAB)
- **Function:** `cv2.cvtColor(img, cv2.COLOR_RGB2LAB)`
- **Description:** Converts to the **perceptually uniform** CIELAB color space. L = lightness, a = green-red axis, b = blue-yellow axis.
- **Parameters:** None.

##### RGB to YCrCb
- **Function:** `cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)`
- **Description:** Converts to YCrCb color space used in video compression (JPEG, MPEG). Y = luminance, Cr = red-difference chrominance, Cb = blue-difference chrominance.
- **Parameters:** None.

---

## Test Patterns

The application includes three built-in test patterns:

| Pattern | Description | Best Used For |
|---|---|---|
| **Grid / Frequency Lines** | White vertical lines and gray horizontal lines on a black background. | Frequency domain analysis, FFT spectrum visualization. |
| **Noisy Coins** | Gray circles of varying brightness on a dark background with salt-and-pepper noise. | Noise removal (median/adaptive filters), segmentation, contour detection. |
| **Brightness Gradient** | Smooth vertical gradient from black (top) to white (bottom). | Histogram equalization, gamma correction, thresholding. |

A **default geometric pattern** is also loaded on startup (split background, green circle, orange rectangle, text overlay).

---

## Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Core image processing (filtering, edge detection, morphology, color conversion, FFT, thresholding, contours, Hough transform, distance transforms, interpolation). |
| `numpy` | Numerical array operations, noise generation, FFT computation, pixel-level manipulation, custom filter implementations. |
| `Pillow` | Image format conversion between OpenCV/NumPy arrays and Tkinter-compatible `PhotoImage` objects. |
| `matplotlib` | Histogram visualization with `FigureCanvasTkAgg` backend for embedding plots in Tkinter. |

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

```bash
python main.py
```

The application window will open maximized with a default test pattern loaded. You can immediately start experimenting with algorithms or load your own image.

---

## License

This project was developed as an academic coursework project for a **Machine Vision and Image Processing** course.