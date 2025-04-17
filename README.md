# 🌤️ Multiclass Weather Prediction using CNN

This project focuses on classifying weather conditions from images into multiple categories using a Convolutional Neural Network (CNN). By leveraging deep learning techniques, the model aims to accurately predict weather types, aiding in applications like weather forecasting and environmental monitoring.

---

## 📂 Dataset

The model is trained on the [Multi-class Weather Dataset](https://www.kaggle.com/datasets/saurabhshahane/multi-class-weather-dataset), which comprises images categorized into the following classes:

- Cloudy
- Rainy
- Sunrise
- Shine

---

## 🧠 Model Architecture

The CNN model is built using TensorFlow and Keras, featuring:

- Multiple convolutional layers with ReLU activation
- MaxPooling layers to reduce spatial dimensions
- Dropout layers to prevent overfitting
- Fully connected (Dense) layers
- Output layer with softmax activation for multiclass classification

---

## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/Srijan-Ratrey/Multiclass-weather-prediction.git
cd Multiclass-weather-prediction
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

1. Download and extract the dataset into the `data/` directory.
2. Run the notebook or training script to train the model.
3. Use the trained model to make predictions on new images.

---

## 📊 Results

The model performs well across all weather classes.

![Model Accuracy](demo.png)

---
