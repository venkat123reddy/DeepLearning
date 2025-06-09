# Deep Learning Assignment - Detailed Report

## 📘 assignment_2_task_3.ipynb
**Objective:**
<a href="https://colab.research.google.com/github/venkat123reddy/DeepLearning/blob/main/assignment_2_task_3.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

**Work Done:**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Convert to DataFrame for better visualization
df = pd.DataFrame(X, columns=iris.feature_names)

print("Original Dataset (first 5 rows):")
print(df.head())

# Step 2: Min-Max Normalization
min_max_scaler = MinMaxScaler()
X_norm = min_max_scaler.fit_transform(X)
df_norm = pd.DataFrame(X_norm, columns=iris.feature_names)
print("\nMin-Max Normalized Dataset (first 5 rows):")
print(df_norm.head())

# Step 3: Z-score Standardization
std_scaler = StandardScaler()
X_std = std_scaler.fit_transform(X)
df_std = pd.DataFrame(X_std, columns=iris.feature_names)
print("\nZ-score Standardized Dataset (first 5 rows):")
print(df_std.head())

# Visualization: Histograms
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(X_norm.flatten(), bins=20, color='skyblue')
plt.title('Min-Max Normalization')

plt.subplot(1, 2, 2)
plt.hist(X_std.flatten(), bins=20, color='salmon')
plt.title('Z-score Standardization')
plt.tight_layout()
plt.show()

# Step 4: Train Logistic Regression Model

# Split original dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
orig_acc = accuracy_score(y_test, model.predict(X_test))

# On Normalized Data
Xn_train, Xn_test, yn_train, yn_test = train_test_split(X_norm, y, test_size=0.3, random_state=42)
model.fit(Xn_train, yn_train)
norm_acc = accuracy_score(yn_test, model.predict(Xn_test))

# On Standardized Data
Xs_train, Xs_test, ys_train, ys_test = train_test_split(X_std, y, test_size=0.3, random_state=42)
model.fit(Xs_train, ys_train)
std_acc = accuracy_score(ys_test, model.predict(Xs_test))

print(f"\nAccuracy without scaling: {orig_acc:.4f}")
print(f"Accuracy with Min-Max Normalization: {norm_acc:.4f}")
print(f"Accuracy with Z-score Standardization: {std_acc:.4f}")

print('''Technique	Use When...
Normalization
(Min-Max Scaling)	- we know the data has fixed bounds (e.g., pixel values 0-255)
- we want features between [0, 1] or [-1, 1]
Standardization
(Z-score Scaling)	- Data follows a Gaussian (normal) distribution
- Model assumes centered data (e.g., Logistic Regression, SVM, PCA)
- Deep Learning models prefer zero-mean inputs for faster convergence''')
```

**Outcomes:**
```python
No outcomes identified.
```

**Learnings:**
No additional learnings documented.

---

## 📘 assignment2_q1.ipynb
**Objective:**
<a href="https://colab.research.google.com/github/venkat123reddy/DeepLearning/blob/main/assignment2_q1.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

**Work Done:**
```python
import numpy as np
import tensorflow as tf


# Define the input matrix (5x5)
input_matrix = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
], dtype=np.float32)


# Reshape input to 4D tensor: [batch, height, width, channels]
input_tensor = input_matrix.reshape(1, 5, 5, 1)

```

**Outcomes:**
```python
# Function to perform convolution and print output
def perform_convolution(stride, padding):
    result = tf.nn.conv2d(input_tensor, kernel_tensor, strides=[1, stride, stride, 1], padding=padding)
    print(f"\nStride = {stride}, Padding = '{padding}'")
    print(result.numpy().squeeze())  # Remove extra dimensions for display


# Perform required convolutions
perform_convolution(stride=1, padding='VALID')
perform_convolution(stride=1, padding='SAME')
perform_convolution(stride=2, padding='VALID')
perform_convolution(stride=2, padding='SAME')
```

**Learnings:**
No additional learnings documented.

---

## 📘 Assignment_2_task_2.ipynb
**Objective:**
<a href="https://colab.research.google.com/github/venkat123reddy/DeepLearning/blob/main/Assignment_2_task_2.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

**Work Done:**
```python
import tensorflow as tf
import numpy as np

# Create a 4x4 random input matrix
input_matrix = np.random.randint(0, 10, (1, 4, 4, 1)).astype(np.float32)

# Max Pooling (2x2)
max_pool = tf.nn.max_pool2d(input_matrix, ksize=2, strides=2, padding='VALID')

# Average Pooling (2x2)
avg_pool = tf.nn.avg_pool2d(input_matrix, ksize=2, strides=2, padding='VALID')

# Display results
print("Original 4x4 matrix:\n", input_matrix[0, :, :, 0])  # ✅ remove .numpy()
print("\nMax Pooled (2x2):\n", max_pool[0, :, :, 0].numpy())  # ✅ this is a Tensor
print("\nAverage Pooled (2x2):\n", avg_pool[0, :, :, 0].numpy())  # ✅ this is a Tensor

```

**Outcomes:**
```python
No outcomes identified.
```

**Learnings:**
No additional learnings documented.

---

## 📘 Assignment_q2_task1.ipynb
**Objective:**
<a href="https://colab.research.google.com/github/venkat123reddy/DeepLearning/blob/main/Assignment_q2_task1.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

**Work Done:**
```python
import cv2
import numpy as np
from matplotlib import pyplot as plt
# Generate a synthetic grayscale image (gradient for test)
image = np.tile(np.arange(0, 255, 10, dtype=np.uint8), (25, 1))
# Define Sobel X and Y filters
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)

sobel_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float32)
```

**Outcomes:**
```python
# Apply filters
edge_x = cv2.filter2D(image, -1, sobel_x)
edge_y = cv2.filter2D(image, -1, sobel_y)

# Plot results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Sobel X")
plt.imshow(edge_x, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Sobel Y")
plt.imshow(edge_y, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
```

**Learnings:**
No additional learnings documented.

---

