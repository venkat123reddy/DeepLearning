
# 📘 Assignment 1: Deep Learning with TensorFlow

Name : Venkata Reddy Attala (700770541 - vxa05410@ucmo.edu)

This assignment focuses on three foundational aspects of deep learning using TensorFlow:
1. Tensor operations and reshaping
2. Understanding and comparing loss functions
3. Training a neural network and analyzing performance using TensorBoard

Each question builds hands-on understanding through coding and visualization.

---

## ✅ Question 1: Tensor Manipulations & Reshaping

### 🧩 Objective:
Work with TensorFlow tensors by reshaping, transposing, and broadcasting. These operations are essential when preparing data or modifying layers in deep learning models.

### 🔍 What was done:
1. **Created a random tensor** of shape `(4, 6)` using TensorFlow.
2. **Printed the rank and shape** of the tensor using `tf.rank()` and `tf.shape()` to understand its dimensional structure.
3. **Reshaped** the tensor into `(2, 3, 4)` using `tf.reshape()`, which rearranges data without changing the values.
4. **Transposed** it to `(3, 2, 4)` using `tf.transpose()` to swap axes for operations like matrix multiplication.
5. **Broadcasted** a smaller tensor of shape `(1, 4)` to add to the larger tensor. Broadcasting automatically expands the smaller tensor to match the shape of the larger one.

### 💡 Key Concept – Broadcasting:
In TensorFlow, **broadcasting** allows tensors of different shapes to be used together in arithmetic operations. TensorFlow automatically expands the smaller tensor so that the shapes align without manual replication.

### 🖥️ Output Observations:
- Rank before and after reshaping was printed successfully.
- Broadcasting worked without any errors, demonstrating TensorFlow's efficiency in handling shape differences.

---

## ✅ Question 2: Loss Functions & Hyperparameter Tuning

### 🧩 Objective:
Understand how model loss is calculated using different loss functions and observe how changes in predictions affect the loss.

### 🔍 What was done:
1. Defined **true labels** (`y_true`) as one-hot encoded vectors.
2. Created two sets of predictions (`y_pred1`, `y_pred2`) where the second set was slightly less accurate.
3. Calculated:
   - **Mean Squared Error (MSE):** Measures average squared differences between actual and predicted values. Common in regression.
   - **Categorical Cross-Entropy (CCE):** Measures the difference between true and predicted class probabilities. Used in classification tasks.
4. Slightly modified predictions and **observed changes** in both MSE and CCE.
5. Created a **bar chart using Matplotlib** to compare both loss values for the two prediction sets.

### 📊 Insights:
- Both MSE and CCE increased as predictions got worse.
- **CCE was more sensitive** to small changes in predicted class probabilities, which is expected in classification problems.
- Visualization made it easier to compare how each loss function behaves.

---

## ✅ Question 3: Train a Neural Network & Log to TensorBoard

### 🧩 Objective:
Train a neural network on the MNIST dataset and analyze its training behavior using TensorBoard.

### 🔍 What was done:
1. Loaded the **MNIST dataset**, which contains 28x28 grayscale images of handwritten digits (0–9).
2. Preprocessed the data by normalizing pixel values to [0, 1].
3. Built a simple **feedforward neural network** using `tf.keras.Sequential`, with:
   - Flattening input layer
   - Hidden dense layers with ReLU activation
   - Output layer with softmax activation for 10 classes
4. Compiled and trained the model for **5 epochs**.
5. Used a **TensorBoard callback** to log training metrics to the `logs/fit/` directory.

### 📈 TensorBoard Analysis (from Question_4.docx):

#### 1️⃣ What patterns do you observe in the accuracy curves?
- **Training accuracy** improves consistently.
- **Validation accuracy** improves initially, but may plateau or decrease if the model overfits.

#### 2️⃣ How can TensorBoard help detect overfitting?
- You can visually **compare training and validation curves**.
- If training loss keeps decreasing but validation loss starts increasing, it’s a clear sign of overfitting.

#### 3️⃣ What happens when you increase the number of epochs?
- At first, accuracy improves for both training and validation.
- After too many epochs, the model starts **memorizing** the training data.
- This leads to **overfitting**, where training accuracy keeps increasing but validation accuracy drops.

---

## 📂 Project Structure

```
.
├── Assignment_1_Q1.ipynb      # Tensor operations: reshaping, transposing, broadcasting
├── Assignment_1_q2.ipynb      # Loss function comparison using MSE and Cross-Entropy
├── Assignment_1_Q3.ipynb      # MNIST model training + TensorBoard logging
├── Question_4.docx            # Answers related to TensorBoard analysis
└── README.md                  # This explanation file
```

---

## 🎓 What I Learned

- How to work with multi-dimensional tensors using reshape, transpose, and broadcast.
- How loss functions behave and how prediction quality affects loss values.
- How to build a neural network, train it on real data, and use TensorBoard for performance monitoring.
- The importance of visual tools like TensorBoard to detect issues like **overfitting** early during training.
