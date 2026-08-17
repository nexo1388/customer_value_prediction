# Customer Value Prediction

A Deep Learning regression project for predicting customer value using TensorFlow, Keras, and Keras Tuner with hyperparameter optimization.

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

## Neural Network Architecture

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
```

Keras Tuner is used to optimize important hyperparameters such as the number of neurons and learning rate.

## Machine Learning Pipeline

```text
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
```

## Model Evaluation

The final model achieved the following results on the test set:

| Metric | Score |
|---|---:|
| MAE | 150.30 |
| MSE | 35804.63 |
| RMSE | 189.22 |
| R² | 0.9424 |

An R² score of approximately 0.94 indicates strong predictive performance on the test dataset.

## Prediction

The trained model can be used to predict the value of a new customer.

```python
prediction = predict_customer_value(new_customer)

print(prediction)
```

The prediction pipeline applies the same feature scaling used during training and then converts the prediction back to the original target scale.

## Model Saving

The trained model and scalers are saved for future use:

```python
model.save("customer_value_model.keras")

joblib.dump(scaler, "feature_scaler.pkl")
joblib.dump(y_scaler, "target_scaler.pkl")
```

This allows the model to be reused without retraining.

## Key Concepts

- Deep Learning Regression
- TensorFlow & Keras
- Neural Networks
- Keras Tuner
- Hyperparameter Optimization
- Feature Scaling
- Target Scaling
- Early Stopping
- Learning Rate Scheduling
- Model Checkpointing
- MAE, MSE, RMSE and R²
- Model Persistence
- Prediction on New Data

## Future Improvements

- Feature Engineering
- Model Comparison
- Cross-Validation

## Author

Nima Hossienian

## Machine Learning & Deep Learning Portfolio Project
