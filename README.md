# Comprehensive Machine Vision & Image Processing System

A full-featured desktop application for **machine vision and image processing**, built with Python using **Tkinter**, **OpenCV**, **NumPy**, **Pillow**, and **Matplotlib**. The application provides an interactive GUI where users can load images, apply a wide range of image processing algorithms, visualize results side-by-side, and inspect histograms — all organized by academic textbook chapters.

---

## Table of Contents

- [Features](#features)
- [Screenshots & UI Layout](#screenshots--ui-layout)
- [Installation](#installation)
- [Usage](#usage)
- [Project Architecture](#project-architecture)
- [Algorithms & Filters Reference](#algorithms--filters-reference)
  - [Chapter 1: Introduction](#chapter-1-introduction)
  - [Chapter 2: Digital Image Fundamentals](#chapter-2-digital-image-fundamentals)
  - [Chapter 3: Gray-Level Transformations & Spatial Filtering](#chapter-3-gray-level-transformations--spatial-filtering)
  - [Chapter 4: Frequency Domain Filtering](#chapter-4-frequency-domain-filtering)
  - [Chapter 5: Image Restoration & Noise](#chapter-5-image-restoration--noise)
  - [Chapter 6: Color Image Processing](#chapter-6-color-image-processing)
- [Test Patterns](#test-patterns)
- [Dependencies](#dependencies)
- [License](#license)

---

## Features

- **Side-by-side display** of original and processed images on interactive canvases.
- **30+ image processing algorithms** organized into 6 academic chapters.
- **Dynamic parameter sliders** that adapt to the selected algorithm (kernel size, thresholds, sigma, gamma, etc.).
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
| **Left Sidebar** | Chapter/category selector, algorithm selector, two dynamic parameter sliders, an apply button, and an image statistics panel. |
| **Right Display Area** | Two side-by-side canvases showing the original input image and the processed result. |

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Steps

```bash
# Clone the repository
git clone <repository-url>
cd <project-directory>

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

---

## Usage

1. **Launch** the application by running `python main.py`.
2. **Load an image** using the "Load Image" button, or select a built-in **test pattern**.
3. **Select a chapter** from the category dropdown to filter relevant algorithms.
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
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation (this file)
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

### Chapter 1: Introduction

#### Grayscale Conversion
- **Function:** `cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)`
- **Description:** Converts a color (RGB) image to a single-channel grayscale image by computing a weighted sum of the R, G, and B channels. The standard luminosity formula is used: `Y = 0.299R + 0.587G + 0.114B`. The result is converted back to a 3-channel image for uniform display.
- **Parameters:** None.

---

### Chapter 2: Digital Image Fundamentals

#### Find Contours
- **Function:** `cv2.findContours()` + `cv2.drawContours()`
- **Description:** Detects the boundaries (contours) of objects in a binary image. The image is first converted to grayscale, then binarized using a fixed threshold (127). Contours are extracted using `cv2.RETR_EXTERNAL` mode (only outermost contours) with `cv2.CHAIN_APPROX_SIMPLE` approximation (compresses horizontal, vertical, and diagonal segments into their endpoints). Detected contours are drawn in green on the original image.
- **Algorithm Details:**
  - Thresholding: `cv2.THRESH_BINARY_INV` at value 127.
  - Contour retrieval mode: `RETR_EXTERNAL` — retrieves only the extreme outer contours.
  - Contour approximation: `CHAIN_APPROX_SIMPLE` — removes redundant points and compresses the contour.
- **Parameters:** None.

#### Hough Line Transform
- **Function:** `cv2.HoughLines()`
- **Description:** Detects straight lines in an image using the **Standard Hough Transform**. The algorithm works in the Hough parameter space (ρ, θ), where each point in the image edge map votes for all lines passing through it. Lines that accumulate enough votes (above the threshold) are detected. The process involves:
  1. Edge detection using Canny (thresholds 50 and 150).
  2. Transformation to Hough space with ρ resolution = 1 pixel and θ resolution = π/180 radians.
  3. Lines with accumulator votes above the user-defined threshold are drawn on the image (limited to 15 lines maximum).
- **Parameters:**
  - **Accumulator Threshold** (Slider 1, range 20–150): Minimum number of votes required to consider a line.

---

### Chapter 3: Gray-Level Transformations & Spatial Filtering

This chapter contains the largest collection of algorithms, covering intensity transformations, spatial smoothing filters, edge detection filters, morphological operations, and thresholding/segmentation techniques.

#### Intensity Transformations

##### Negative (Image Inversion)
- **Formula:** `s = 255 - r` (for each pixel value `r`)
- **Description:** Inverts all pixel intensities, producing a photographic negative. Bright pixels become dark and vice versa. Useful for enhancing white or light details embedded in dark regions.
- **Parameters:** None.

##### Log Transformation
- **Formula:** `s = c · log(1 + r)`, where `c = 255 / log(1 + max_val)`
- **Description:** Applies a logarithmic transformation to compress the dynamic range of images with large variations in pixel intensity. Maps a narrow range of low-intensity values to a wider range of output levels, and compresses higher-intensity values. The constant `c` is computed to scale the output to the full [0, 255] range.
- **Use Case:** Displaying Fourier spectra, enhancing dark regions of an image.
- **Parameters:** None.

##### Gamma Correction (Power-Law Transformation)
- **Formula:** `s = 255 · (r / 255)^γ`
- **Description:** Applies a power-law (gamma) transformation to adjust image brightness and contrast. The transformation uses a lookup table (LUT) for efficiency.
  - **γ < 1:** Brightens the image (expands low-intensity range, compresses high-intensity range).
  - **γ = 1:** No change (identity transformation).
  - **γ > 1:** Darkens the image (compresses low-intensity range, expands high-intensity range).
- **Parameters:**
  - **Gamma value** (Slider 1, range 1–50, divided by 10): The gamma exponent (e.g., slider value 10 = γ = 1.0).

##### Histogram Equalization
- **Function:** `cv2.equalizeHist()`
- **Description:** Enhances image contrast by redistributing pixel intensities so that the output histogram is approximately uniform (flat). For color images, the algorithm converts to **YUV** color space and equalizes only the **Y (luminance)** channel, preserving color information. For grayscale images, equalization is applied directly.
- **Mathematical Basis:** The transformation function is based on the **Cumulative Distribution Function (CDF)** of the image histogram: `s = T(r) = (L-1) · CDF(r)`, where `L = 256`.
- **Parameters:** None.

#### Spatial Smoothing Filters

##### Gaussian Blur
- **Function:** `cv2.GaussianBlur(img, (k, k), sigma)`
- **Description:** Applies a **Gaussian low-pass filter** that smooths the image by convolving it with a 2D Gaussian kernel. The kernel weights decrease with distance from the center, following a bell-curve distribution. This filter effectively reduces **Gaussian noise** while preserving edges better than a simple average filter.
- **Kernel Formula:** `G(x, y) = (1 / 2πσ²) · exp(-(x² + y²) / 2σ²)`
- **Parameters:**
  - **Kernel Size** (Slider 1, range 3–25, must be odd): Size of the convolution kernel.
  - **Sigma** (Slider 2, range 1–100, divided by 10): Standard deviation of the Gaussian distribution.

##### Median Blur
- **Function:** `cv2.medianBlur(img, k)`
- **Description:** A **non-linear** smoothing filter that replaces each pixel with the **median** value of its neighborhood. Unlike linear filters, the median filter is extremely effective at removing **salt-and-pepper (impulse) noise** while preserving sharp edges.
- **Algorithm:** Sorts all pixel values in the kernel window and selects the middle value.
- **Parameters:**
  - **Kernel Size** (Slider 1, range 3–25, must be odd): Size of the neighborhood window.

##### Average Blur (Box Filter)
- **Function:** `cv2.blur(img, (k, k))`
- **Description:** Applies a simple **mean (average) filter** that replaces each pixel with the arithmetic mean of all pixels in its neighborhood. This is the simplest spatial smoothing filter and is equivalent to convolving with a kernel where all coefficients are equal to `1/k²`.
- **Parameters:**
  - **Kernel Size** (Slider 1, range 3–25, must be odd): Size of the averaging window.

##### Bilateral Filter
- **Function:** `cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)`
- **Description:** An **edge-preserving** smoothing filter that considers both **spatial distance** and **intensity difference** when averaging pixels. Unlike Gaussian blur, the bilateral filter assigns lower weights to pixels that are spatially close but have very different intensities, effectively smoothing flat regions while keeping edges sharp.
- **Dual Gaussian Weighting:**
  - **Spatial Gaussian (σ_s):** Controls the influence of distant pixels (spatial domain).
  - **Range Gaussian (σ_r):** Controls the influence of pixels with different intensities (intensity domain).
- **Parameters:**
  - **Spatial Sigma (σ_s)** (Slider 1, range 5–50): Spatial distance influence.
  - **Color Sigma (σ_r)** (Slider 2, range 5–100): Intensity difference tolerance.

#### Edge Detection Filters

##### Laplacian
- **Function:** `cv2.Laplacian(gray, cv2.CV_64F)`
- **Description:** A **second-order derivative** operator that detects edges by computing the Laplacian of the image. It highlights regions of rapid intensity change (edges) in all directions simultaneously. The Laplacian kernel approximates `∇²f = ∂²f/∂x² + ∂²f/∂y²`.
- **Properties:** Isotropic (direction-independent), sensitive to noise, detects both edges and noise.
- **Parameters:** None.

##### Sobel X (Horizontal Edge Detection)
- **Function:** `cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)`
- **Description:** Computes the **first-order derivative** in the horizontal (x) direction using the Sobel operator. Detects **vertical edges** (intensity changes along the horizontal axis). The 3×3 Sobel kernel for the x-direction is:
  ```
  [-1  0  1]
  [-2  0  2]
  [-1  0  1]
  ```
- **Parameters:** None.

##### Sobel Y (Vertical Edge Detection)
- **Function:** `cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)`
- **Description:** Computes the **first-order derivative** in the vertical (y) direction using the Sobel operator. Detects **horizontal edges** (intensity changes along the vertical axis). The 3×3 Sobel kernel for the y-direction is:
  ```
  [-1 -2 -1]
  [ 0  0  0]
  [ 1  2  1]
  ```
- **Parameters:** None.

##### Sobel Combined
- **Description:** Computes both Sobel X and Sobel Y gradients and combines them using a **weighted sum**: `|G| = 0.5·|Gx| + 0.5·|Gy|`. This provides a combined edge magnitude map that highlights edges in all directions.
- **Function:** `cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)`
- **Parameters:** None.

##### Canny Edge Detector
- **Function:** `cv2.Canny(gray, threshold1, threshold2)`
- **Description:** A multi-stage edge detection algorithm developed by **John F. Canny** (1986). It is considered the **optimal edge detector** and involves four steps:
  1. **Gaussian smoothing** to reduce noise.
  2. **Gradient computation** using Sobel operators to find edge intensity and direction.
  3. **Non-maximum suppression** to thin edges to single-pixel width.
  4. **Hysteresis thresholding** using two thresholds to classify edges as strong, weak, or non-edges. Weak edges connected to strong edges are kept; others are suppressed.
- **Parameters:**
  - **Low Threshold** (Slider 1, range 1–150): Lower bound for hysteresis thresholding.
  - **High Threshold** (Slider 2, range 10–250): Upper bound for hysteresis thresholding.

#### Morphological Operations

All morphological operations use a **square structuring element** of user-defined size.

##### Erosion
- **Function:** `cv2.erode(img, kernel, iterations=1)`
- **Description:** Erodes (shrinks) bright regions by replacing each pixel with the **minimum** value in its neighborhood defined by the structuring element. Useful for removing small white noise and separating connected objects.
- **Effect:** Objects shrink, gaps widen, thin connections break.

##### Dilation
- **Function:** `cv2.dilate(img, kernel, iterations=1)`
- **Description:** Dilates (expands) bright regions by replacing each pixel with the **maximum** value in its neighborhood defined by the structuring element. Useful for filling small holes and connecting nearby objects.
- **Effect:** Objects grow, small gaps fill, nearby objects merge.

##### Opening (Erosion → Dilation)
- **Function:** `cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)`
- **Description:** An **erosion followed by a dilation** using the same structuring element. Removes small bright spots (noise) while preserving the overall shape and size of larger objects.
- **Use Case:** Noise removal from binary images.

##### Closing (Dilation → Erosion)
- **Function:** `cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)`
- **Description:** A **dilation followed by an erosion** using the same structuring element. Fills small dark holes and gaps within bright objects while preserving their overall shape and size.
- **Use Case:** Filling gaps in detected contours or segmented regions.

##### Morphological Gradient
- **Function:** `cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)`
- **Description:** The **difference between dilation and erosion** of the image: `Gradient = Dilation(f) - Erosion(f)`. Produces an outline (boundary) of the objects in the image.
- **Use Case:** Edge/boundary extraction.

**Common Parameter for all morphological operations:**
- **Structuring Element Size** (Slider 1, range 3–15, must be odd): Size of the square kernel.

#### Thresholding / Segmentation

##### Global Thresholding
- **Function:** `cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)`
- **Description:** Converts a grayscale image to binary using a single global threshold value. Pixels above the threshold become white (255); pixels below become black (0).
- **Parameters:**
  - **Threshold Value** (Slider 1, range 0–255): The boundary intensity value.

##### Adaptive Mean Threshold
- **Function:** `cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize, C)`
- **Description:** Computes a local threshold for each pixel based on the **mean** intensity of its neighborhood block. The threshold for each pixel is: `T(x,y) = mean(neighborhood) - C`. This method handles images with varying illumination much better than global thresholding.
- **Parameters:**
  - **Block Size** (Slider 1, range 3–45, must be odd): Size of the local neighborhood.
  - **Constant C** (Slider 2, range -20 to 40): Value subtracted from the computed mean.

##### Adaptive Gaussian Threshold
- **Function:** `cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize, C)`
- **Description:** Similar to adaptive mean thresholding, but uses a **Gaussian-weighted sum** of the neighborhood values instead of a simple mean. This gives more weight to pixels closer to the center of the neighborhood block, producing smoother results.
- **Parameters:**
  - **Block Size** (Slider 1, range 3–45, must be odd): Size of the local neighborhood.
  - **Constant C** (Slider 2, range -20 to 40): Value subtracted from the weighted mean.

##### Otsu's Thresholding
- **Function:** `cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)`
- **Description:** An **automatic** global thresholding method that computes the optimal threshold value by minimizing the **intra-class variance** (or equivalently, maximizing the **inter-class variance**) of the two pixel classes (foreground and background). No manual threshold parameter is needed — the algorithm determines the best value from the image histogram.
- **Mathematical Basis:** Exhaustively searches all possible threshold values and selects the one that minimizes `σ²_w(t) = w_0(t)·σ²_0(t) + w_1(t)·σ²_1(t)`, where `w_i` and `σ²_i` are the weight and variance of each class.
- **Parameters:** None (automatically determined).

---

### Chapter 4: Frequency Domain Filtering

#### FFT Spectrum (Fourier Transform Magnitude Spectrum)
- **Function:** `np.fft.fft2()` + `np.fft.fftshift()`
- **Description:** Computes the **2D Discrete Fourier Transform (DFT)** of the grayscale image and displays its **magnitude spectrum**. The FFT converts the image from the spatial domain to the frequency domain, where low frequencies represent smooth regions and high frequencies represent edges and fine details. The spectrum is shifted so that the zero-frequency (DC) component is at the center. A logarithmic scaling is applied for visualization: `S = 20 · log(|F(u,v)| + 1)`.
- **Parameters:** None.

#### Ideal Low-Pass Filter (Frequency Domain)
- **Description:** Removes high-frequency components (edges, noise, fine details) by multiplying the frequency spectrum with a circular binary mask. All frequencies within a radius `D0` from the center (DC component) are **passed**, and all frequencies outside are **blocked** (set to zero). The result is then transformed back to the spatial domain using the inverse FFT.
- **Process:**
  1. Compute 2D FFT and shift the zero-frequency component to center.
  2. Create a circular binary mask of radius `D0`.
  3. Multiply the shifted spectrum with the mask.
  4. Compute the inverse FFT to obtain the filtered image.
- **Transfer Function:** `H(u,v) = 1 if D(u,v) ≤ D0, else 0`
- **Parameters:**
  - **Cutoff Radius D0** (Slider 1, range 5–60): The frequency cutoff radius in pixels.

#### Ideal High-Pass Filter (Frequency Domain)
- **Description:** Removes low-frequency components (smooth regions, background) by multiplying the frequency spectrum with the **complement** of the ideal low-pass mask. All frequencies **outside** the radius `D0` are passed, and all frequencies **within** are blocked. This isolates edges, textures, and fine details.
- **Transfer Function:** `H(u,v) = 0 if D(u,v) ≤ D0, else 1`
- **Parameters:**
  - **Cutoff Radius D0** (Slider 1, range 5–60): The frequency cutoff radius in pixels.

---

### Chapter 5: Image Restoration & Noise

#### Add Salt & Pepper Noise (Impulse Noise)
- **Description:** Artificially adds **impulse noise** to the image by randomly setting a percentage of pixels to either pure white (255 — "salt") or pure black (0 — "pepper"). The number of salt and pepper pixels is split evenly (50/50). This type of noise simulates sensor failures or transmission errors.
- **Algorithm:**
  1. Calculate the number of noisy pixels: `count = percentage × total_pixels × 0.5`.
  2. Randomly select pixel coordinates and set them to white (salt).
  3. Randomly select another set of pixel coordinates and set them to black (pepper).
- **Parameters:**
  - **Noise Percentage** (Slider 1, range 1–40): Percentage of total pixels to be corrupted.

#### Add Gaussian Noise
- **Description:** Adds **additive Gaussian noise** to the image. Random values drawn from a normal distribution `N(0, σ)` are added to each pixel. This simulates sensor noise and electronic interference commonly found in real-world imaging systems.
- **Formula:** `noisy(x,y) = image(x,y) + N(0, σ)`
- **Parameters:**
  - **Sigma (Standard Deviation)** (Slider 1, range 5–100): Controls the noise intensity.

---

### Chapter 6: Color Image Processing

#### RGB to HSV
- **Function:** `cv2.cvtColor(img, cv2.COLOR_RGB2HSV)`
- **Description:** Converts the image from the **RGB** color space to **HSV (Hue, Saturation, Value)**. HSV separates color information (hue, saturation) from intensity (value), making it more intuitive for color-based analysis and segmentation.
  - **H (Hue):** Represents the color type (0–180 in OpenCV).
  - **S (Saturation):** Represents color purity (0–255).
  - **V (Value):** Represents brightness (0–255).
- **Parameters:** None.

#### RGB to LAB (CIELAB)
- **Function:** `cv2.cvtColor(img, cv2.COLOR_RGB2LAB)`
- **Description:** Converts the image to the **CIELAB** color space, which is designed to be **perceptually uniform** — meaning that equal numerical changes correspond to equal perceived color differences.
  - **L:** Lightness (0–100).
  - **a:** Green–Red axis.
  - **b:** Blue–Yellow axis.
- **Use Case:** Color difference measurement, color-based segmentation, image analysis where perceptual uniformity is important.
- **Parameters:** None.

#### RGB to YCrCb
- **Function:** `cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)`
- **Description:** Converts the image to the **YCrCb** color space, commonly used in video compression (JPEG, MPEG). It separates luminance from chrominance:
  - **Y:** Luminance (brightness).
  - **Cr:** Red-difference chrominance.
  - **Cb:** Blue-difference chrominance.
- **Use Case:** Skin detection, video compression analysis, chroma subsampling studies.
- **Parameters:** None.

---

## Test Patterns

The application includes three built-in test patterns for experimentation:

| Pattern | Description | Best Used For |
|---|---|---|
| **Grid / Frequency Lines** | White vertical lines and gray horizontal lines on a black background. | Frequency domain analysis, FFT spectrum visualization. |
| **Noisy Coins** | Gray circles of varying brightness on a dark background with salt-and-pepper noise injected. | Noise removal (median filter), segmentation, contour detection. |
| **Brightness Gradient** | A smooth vertical gradient from black (top) to white (bottom). | Histogram equalization, gamma correction, thresholding tests. |

Additionally, a **default geometric pattern** is loaded on startup featuring a split background, a filled green circle, an orange rectangle, and text overlay — suitable for general-purpose algorithm testing.

---

## Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Core image processing algorithms (filtering, edge detection, morphology, color conversion, FFT, thresholding, contours, Hough transform). |
| `numpy` | Numerical array operations, noise generation, FFT computation, and pixel-level manipulation. |
| `Pillow` | Image format conversion between OpenCV/NumPy arrays and Tkinter-compatible `PhotoImage` objects. |
| `matplotlib` | Histogram visualization using the `FigureCanvasTkAgg` backend for embedding plots in Tkinter windows. |

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