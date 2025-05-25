Handwritten Digit Classification:

This project demonstrates various machine learning and deep learning techniques to classify handwritten digits using the Optical Recognition of Handwritten Digits dataset (load_digits) and the MNIST dataset.

Project Structure
handwritten_digit_classification.py — Main Python script covering:

ML models: SVM, KNN, Random Forest, Logistic Regression, MLP

DL model: CNN using TensorFlow/Keras

Evaluation metrics: Accuracy, Precision, Recall, F1-score

Custom image prediction with CNN

Models Used :
Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

Random Forest

Logistic Regression

Multi-layer Perceptron (MLP)

Convolutional Neural Network (CNN) using MNIST

Custom Digit Prediction
The CNN model is extended to classify user-supplied digit images (e.g., 4.jpeg) after preprocessing:

Grayscale conversion

Resizing to 28x28

Inversion

Normalization

 How to Run
1. Clone the Repository
git clone https://github.com/your-username/handwritten-digit-classification.git
cd handwritten-digit-classification
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
OR manually install:
pip install numpy matplotlib opencv-python scikit-learn tensorflow
3. Run the Script
python handwritten_digit_classification.py
