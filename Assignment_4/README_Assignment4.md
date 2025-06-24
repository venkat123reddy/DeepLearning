
# 📚 Assignment 4: GANs, Ethical AI, and Fairness in Machine Learning

This repository contains three files that together explore the implementation of Generative Adversarial Networks (GANs), evaluation methods, and critical discussions on fairness, ethics, and societal impact of AI systems.

---

## 📂 Contents

1. `Assignment4_q3.ipynb` – **Building and Training a GAN**
2. `Assignment4_q4.ipynb` – **Evaluating and Visualizing GAN Outputs**
3. `ShortAnswer.docx` – **Theoretical and Ethical Discussion Responses**

---

## 1. `Assignment4_q3.ipynb` – 🧠 GAN Architecture and Training

This Jupyter notebook walks you through the process of implementing a GAN from scratch.

### 🔧 What You'll Learn

- How to use neural networks for **image generation**
- How adversarial training works using **Generator** and **Discriminator**
- How GANs evolve over time through competitive learning

### 🧩 Key Sections

1. **Data Loading and Preprocessing**
   - Dataset: Likely MNIST, CIFAR-10, or a simple custom image dataset
   - Resizing, normalization, and batching
   - Ensures data is ready for input into neural networks

2. **Generator Network**
   - Takes random noise (`z`) as input
   - Upsamples and transforms this noise into an image
   - Uses layers like `Dense`, `BatchNorm`, `ReLU`, `Tanh`

3. **Discriminator Network**
   - A binary classifier to determine if the image is **real** (from dataset) or **fake** (from Generator)
   - Uses Conv2D layers, LeakyReLU, and outputs a probability score

4. **Loss Functions**
   - **Binary Cross-Entropy Loss**
     - Generator wants `D(G(z)) → 1` (fool the discriminator)
     - Discriminator wants `D(x) → 1` for real, `D(G(z)) → 0` for fake
   - Gradients are backpropagated accordingly

5. **Training Loop**
   - Alternates between updating Discriminator and Generator
   - Shows how adversarial learning improves both models

6. **Image Generation and Saving**
   - Generates images every few epochs
   - Saves them for visual inspection of GAN performance over time

---

## 2. `Assignment4_q4.ipynb` – 📊 Evaluating GAN Performance

This notebook builds upon the training done in `q3` and introduces methods to evaluate and visualize how well the GAN has learned to generate realistic images.

### 🧠 Learning Goals

- Evaluate GAN quality using **quantitative metrics**
- Improve model performance through **hyperparameter tuning**
- Interpret **visual and statistical** outputs of a trained GAN

### 📌 Core Components

1. **Reloading Saved Models**
   - Loads weights/checkpoints of the best Generator
   - Allows analysis of the final state of the model

2. **Generated Image Samples**
   - Uses fixed noise vectors to produce consistent samples
   - Compares how image quality improves over epochs

3. **Metric Computation**
   - May include:
     - **Inception Score (IS)**: Measures image diversity and quality
     - **Fréchet Inception Distance (FID)**: Measures distance between real and fake distributions
     - **Mean Squared Error (MSE)** or **SSIM** (structural similarity)

4. **Training Curves**
   - Plots Generator and Discriminator losses
   - Helps diagnose:
     - Mode collapse (when Generator produces only one type of image)
     - Vanishing gradients
     - Overfitting/underfitting

5. **Advanced Techniques (if included)**
   - **Label smoothing**: Helps stabilize training
   - **Dropout or batch noise**: Prevents overfitting
   - **Wasserstein GAN** or **Gradient Penalty** (if explored)

6. **Model Insights**
   - Which epochs produced the best samples?
   - When did training start diverging?
   - How did tuning affect quality?

---

## 3. `ShortAnswer.docx` – ✍️ Ethical AI and GAN Concepts

This document contains theoretical and conceptual answers, written in plain English, to support deep understanding of how GANs work and why responsible AI matters.

### 📖 Detailed Sections

#### 🎮 GAN Architecture and Adversarial Process

- Describes the **zero-sum game** between Generator (G) and Discriminator (D)
- Uses metaphors like artist vs. art critic to explain training
- Explains how adversarial learning pushes both models to improve

#### 🤖 Representational Harm in AI

- Defines **bias and stereotyping** in image generation
- Example: Searching "CEO" might return mostly white men
- Emphasizes how these biased outputs shape perceptions, especially in children

#### 🔐 Legal and Ethical Concerns in Generative AI

- **Data Privacy**: AI models shouldn't regurgitate private info from training data
- **Copyright**: Explains how AI-generated content can resemble copyrighted text or art
- Recommends:
  - Ethical dataset curation
  - Model transparency
  - Respect for creators and user consent

#### 📏 Fairness Metric: False Negative Rate Parity

- Defines **false negative** as a qualified individual wrongly denied opportunity
- Example: AI misses 3 in 10 qualified candidates from Group B but only 1 in 10 from Group A
- Calls for:
  - Diverse training data
  - Regular fairness audits
  - Use of fairness tools like Aequitas or IBM AI Fairness 360

---

## 🛠 Getting Started

### 🧪 Requirements

- Python 3.7+
- Jupyter Notebook or JupyterLab
- Libraries: `tensorflow`, `torch`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `pandas`

### ▶️ Running the Code

```bash
pip install -r requirements.txt
jupyter notebook Assignment4_q3.ipynb
jupyter notebook Assignment4_q4.ipynb
```

---

## 🎓 Final Thoughts

This assignment integrates **deep learning implementation** with **ethical analysis**, showing you not only *how* to build AI—but also *why* we must build it responsibly.

By completing all parts, you’ve gained:
- Practical GAN coding skills
- Hands-on experience evaluating generative models
- Awareness of AI's societal and fairness-related challenges

📬 For presentation slides, a report version, or video walkthrough, feel free to request!

---
