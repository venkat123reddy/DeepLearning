
# Assignment 3 - Deep Learning Tasks

This repository includes five tasks demonstrating the application of key NLP and deep learning techniques. Each notebook addresses a different concept, from text generation to transformer attention mechanisms and sentiment analysis.

---

## 📘 Task 1: Character-Level Text Generation
**Notebook**: `Assignment3_task_1.ipynb`

This task implements a character-level language model for generating text using an LSTM network.

### Steps:
- **Load Dataset**: Load a small chunk (10,000 characters) of text.
- **Preprocess**: Convert characters to numerical sequences and prepare batches.
- **Model Design**: Construct an LSTM model using TensorFlow/Keras.
- **State Management**: Reset states in the LSTM layer during training.
- **Generate Text**: Use the trained model to generate text in an autoregressive manner.

---

## 📘 Task 2: Text Preprocessing with spaCy and NLTK
**Notebook**: `Assignment_3_task2.ipynb`

This task focuses on cleaning raw text data using spaCy and NLTK libraries.

### Steps:
- **Install Dependencies**: Set up `spaCy`, download English model.
- **Preprocessing Pipeline**:
  - Tokenization
  - Removal of stop words and punctuation
  - Stemming using NLTK's PorterStemmer
- **Testing**: Apply the pipeline to a sample sentence to verify its correctness.

---

## 📘 Task 3: Named Entity Recognition (NER) with spaCy
**Notebook**: `Assignment3_task3.ipynb`

This task extracts named entities from text using spaCy’s NER capabilities.

### Steps:
- **Install and Load spaCy**: Download and load English model.
- **NER Extraction**:
  - Input a custom sentence.
  - Use spaCy’s `nlp` pipeline to parse and extract named entities.
  - Print entities with their labels.

---

## 📘 Task 4: Scaled Dot-Product Attention
**Notebook**: `Assignment3_task4.ipynb`

This task implements the Scaled Dot-Product Attention mechanism from Transformer models.

### Steps:
- **Inputs**: Define dummy queries (Q), keys (K), and values (V).
- **Dot Product Calculation**: Compute Q · Kᵀ.
- **Scaling**: Divide by the square root of the key dimension.
- **Softmax**: Normalize scores to get attention weights.
- **Weighted Sum**: Multiply weights with values to produce the attention output.

---

## 📘 Task 5: Sentiment Analysis using Hugging Face Transformers
**Notebook**: `Assignment3_task5.ipynb`

This task uses Hugging Face’s `transformers` library for sentiment analysis.

### Steps:
- **Install transformers** library.
- **Load Pipeline**: Use the pre-trained sentiment-analysis pipeline.
- **Run Inference**: Pass a custom sentence to the pipeline.
- **Output**: Display sentiment label and confidence score.

---

## 💡 Getting Started
1. Clone the repository.
2. Open each notebook in Google Colab.
3. Follow the steps in each cell to execute the code and observe outputs.
