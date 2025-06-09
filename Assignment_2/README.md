# Name : Venkata Reddy Attala (700770541 - vxa05410)
# Deep Learning Assignment - README

This repository contains multiple Jupyter notebooks focusing on different tasks related to image processing, normalization techniques, and convolution operations using TensorFlow, OpenCV, and scikit-learn.

## 📁 Files Overview

### 1. `assignment2_q1.ipynb`
**Topic:** Convolution Operation Using TensorFlow  
**Description:**  
- Demonstrates how to perform 2D convolution on a manually defined 5x5 matrix using a 3x3 Laplacian-like kernel.
- Uses NumPy and TensorFlow for data reshaping and convolution.
- The matrix is reshaped to match the 4D format required by TensorFlow’s convolution layers.

### 2. `Assignment_2_task_2.ipynb`
**Topic:** Pooling Operations (Max & Average)  
**Description:**  
- A random 4x4 matrix is created as input.
- Applies both **max pooling** and **average pooling** using TensorFlow.
- Results are printed for visual comparison:
  - Original matrix
  - Max pooled matrix (2x2)
  - Average pooled matrix (2x2)

### 3. `assignment_2_task_3.ipynb`
**Topic:** Normalization, Standardization, and Logistic Regression  
**Description:**  
- Uses the **Iris dataset** from `sklearn.datasets`.
- Applies **Min-Max Normalization** and **Z-score Standardization**.
- Visualizes histograms of normalized and standardized data.
- Trains a **Logistic Regression** classifier on:
  - Raw data
  - Normalized data
  - Standardized data
- Compares accuracy of the model across different scaling techniques.

### 4. `Assignment_q2_task1.ipynb`
**Topic:** Edge Detection with Sobel Filters using OpenCV  
**Description:**  
- A synthetic grayscale gradient image is generated using NumPy.
- Applies **Sobel X** and **Sobel Y** filters using `cv2.filter2D`.
- Detects vertical and horizontal edges in the synthetic image.
- Can be extended to combine both gradients for complete edge detection.

---

## 🛠 Requirements

- Python 3.x
- NumPy
- OpenCV (`cv2`)
- TensorFlow
- Matplotlib
- scikit-learn
- pandas

Install all requirements using:
```bash
pip install numpy opencv-python tensorflow matplotlib scikit-learn pandas
```

---

## 📌 Usage

To run the notebooks:
1. Open in Google Colab (badge provided at the top of each notebook), or
2. Run locally using Jupyter Notebook or VSCode.
