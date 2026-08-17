# customer_value_prediction
Deep learning regression project for predicting customer value using TensorFlow, Keras, and Keras Tuner with hyperparameter optimization.
# Customer Value Prediction

A Deep Learning regression project for predicting customer value using TensorFlow, Keras, and Keras Tuner.

## Project Overview

The goal of this project is to predict the `customer_value` of a customer based on 15 numerical features.

- 10,000 samples
- 15 input features
- 1 continuous target
- Supervised Learning
- Regression
- Deep Neural Network

## Technologies

- Python
- TensorFlow
- Keras
- Keras Tuner
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

## Neural Network

The model uses a Deep Neural Network with three hidden layers:

```text
Input
  ↓
100 Neurons
  ↓
150 Neurons
  ↓
75 Neurons
  ↓
Output
Keras Tuner is used to optimize important hyperparameters such as the number of neurons and learning rate.
Machine Learning Pipeline
Dataset
   ↓
EDA
   ↓
Preprocessing
   ↓
Train / Validation / Test Split
   ↓
Feature & Target Scaling
   ↓
Neural Network
   ↓
Hyperparameter Optimization
   ↓
Training
   ↓
Evaluation
   ↓
Prediction
Model Evaluation
The final model achieved the following results on the test set:
Metric
Score
MAE
150.30
MSE
35804.63
RMSE
189.22
R²
0.9424
An R² score of approximately 0.94 indicates strong predictive performance on the test dataset.
Prediction
The trained model can be used to predict the value of a new customer.
prediction = predict_customer_value(new_customer)

print(prediction)
The prediction pipeline applies the same feature scaling used during training and then converts the prediction back to the original target scale.
Model Saving
The trained model and scalers are saved for future use:
model.save("customer_value_model.keras")

joblib.dump(scaler, "feature_scaler.pkl")
joblib.dump(y_scaler, "target_scaler.pkl")
This allows the model to be reused without retraining.
  ## Project Structure
customer-value-prediction/
│
├── customer_value_regression.ipynb
├── customer_value_regression.csv
├── customer_value_model.keras
├── feature_scaler.pkl
├── target_scaler.pkl
├── training_history.json
├── requirements.txt
└── README.md
## Key Concepts
Deep Learning Regression
TensorFlow & Keras
Neural Networks
Keras Tuner
Hyperparameter Optimization
Feature Scaling
Target Scaling
Early Stopping
Learning Rate Scheduling
Model Checkpointing
MAE, MSE, RMSE and R²
Model Persistence
Prediction on New Data
Future Improvements
Feature Engineering
Model Comparison
Cross-Validation
API Deployment
Web Application
Cloud Deployment
 ## Author
 ## Nima Hossienian
Machine Learning & Deep Learning Portfolio Project
