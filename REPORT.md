# House Price Prediction - Project Report

## 1. Introduction
This project aims to predict house prices based on various features using a regression model. The trained model is deployed as an API using FastAPI.

## 2. Data Preprocessing
- **Missing Values**: Filled missing numerical values with the median.
- **Feature Scaling**: StandardScaler was used for numerical features.
- **Categorical Encoding**: One-hot encoding was applied to categorical features.
- **Feature Selection**: Highly correlated and redundant features were removed.

## 3. Model Training & Optimization
- **Algorithms Used**: RandomForestRegressor.
- **Hyperparameter Tuning**: GridSearchCV optimized hyperparameters like `n_estimators` and `max_depth`.
- **Evaluation Metrics**:
  - MAE: Measures the average absolute error.
  - RMSE: Gives an error measurement in the same units as the target.
  - R² Score: Explains the variance captured by the model.

## 4. Model Deployment
- **FastAPI** was used to build an API with a `/predict` endpoint.
- The model is serialized and loaded via **Pickle**.
- API takes JSON input and returns predicted house prices.

## 5. Future Improvements
- Deploy API to **AWS/GCP/Azure**.
- Use **DVC/MLflow** for model versioning.
- Implement a **frontend UI** for better user interaction.

## 6. Conclusion
This project successfully builds and deploys a house price prediction model using machine learning techniques. The API provides predictions in real-time, and further improvements can enhance its capabilities.

---
For full project code, visit the GitHub repository [Insert Link].