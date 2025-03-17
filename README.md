# California House Price Prediction Model

This project is part of the **Machine Learning Engineer Assessment**. The objective is to build and deploy a machine learning model to predict house prices based on given features. The project follows a structured pipeline that includes data preprocessing, model training, optimization, deployment as a REST API, and documentation.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
3. [Setup Instructions](#setup-instructions)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Training and Evaluation](#model-training-and-evaluation)
6. [Model Deployment](#model-deployment)
7. [API Usage](#api-usage)
8. [Results and Report](#results-and-report)
9. [Bonus Implementations](#bonus-implementations)

---

## Project Overview

The aim of this project is to develop a **House Price Prediction Model** that:

1. Preprocesses the data by handling missing values, scaling, encoding, and feature selection.
2. Trains a regression model and evaluates its performance.
3. Optimizes hyperparameters to improve prediction accuracy.
4. Deploys the model as a REST API using **FastAPI**.
5. (Optional) Uses Docker for containerization and cloud deployment.

---

## Technologies Used

- **Python 3.x**
- **FastAPI** - For serving the model as a REST API.
- **Docker** - For containerizing the application.
- **Scikit-learn** - For model building and evaluation.
- **Pandas & NumPy** - For data preprocessing.
- **Matplotlib & Seaborn** - For data visualization.
- **Pickle** - For saving the trained model.
- **Postman/CURL** - For API testing.
- **DVC/MLflow** (optional) - For model versioning.
- **GCP** (optional) - For hosting the API.

---

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/username/house-price-prediction.git
   cd house-price-prediction
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt

3. Run the FastAPI application:
   ```bash
   uvicorn notebooks.main:app --reload
   ```
5. Access the API
  Open your browser and go to http://127.0.0.1:8000 to see the home endpoint.
  Visit http://127.0.0.1:8000/docs to access the Swagger UI for testing your API.
4. Test API using CURL or Postman::
   ```bash
   curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{"MedInc":3.5,"HouseAge":15,"AveRooms":5,"AveBedrms":1.0,"Population":1500,"AveOccup":3.0,"Latitude":34.0,"Longitude":-118.0}'
   ```
5. To run the application using Docker:
   ```bash
   docker build -t house-price-prediction .
   docker run -p 127.0.0.1:8000 house-price-prediction
   ```

---

## Data Preprocessing

- Loaded the dataset and performed **Exploratory Data Analysis (EDA)**.
- Handled missing values and outliers.
- Encoded categorical variables using **OneHotEncoder** and **LabelEncoder**.
- Scaled numerical features using **StandardScaler**.
- Visualized feature correlations and distributions.

---

## Model Training and Evaluation

- Models Trained:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - XGBoost Regressor
- Performance Metrics:
  - **RMSE (Root Mean Squared Error)**
  - **MAE (Mean Absolute Error)**
  - **R² Score**
- Hyperparameter optimization using **GridSearchCV** and **RandomizedSearchCV**.
- Saved the trained model using **Pickle** or **Joblib**.

---

## Model Deployment

- The model is deployed as a REST API using **Flask/FastAPI**.
- API Endpoint:
  - `/predict` - Accepts JSON input and returns the predicted house price.
- Example Request:
  ```bash
  curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"MedInc":3.5,"HouseAge":15,"AveRooms":5,"AveBedrms":1.0,"Population":1500,"AveOccup":3.0,"Latitude":34.0,"Longitude":-118.0}'
  ```
- Example Response:
  ```json
  {
    "predicted_price": 1.266559819284305
  }
  ```

---

## API Usage

To test the API, you can use **Postman** or **CURL**. Follow the example request and response formats as shown above.
### Deployed RestAPI
https://predict-backend-762184650075.us-central1.run.app/docs#/default/predict_price_predict_post

---

## Results and Report

The model achieved the following performance on the test set:

- RMSE: *0.283556643705316*
- MAE: *0.4379576582612578*
- R² Score: *0.8050985964067019*

The detailed report and analysis are documented in the `report.md` file.

---

## Bonus Implementations

- **Logging and Error Handling:** Integrated into the API.
- **Model Versioning:** Implemented using **DVC/MLflow**.
- **Cloud Deployment:** API hosted on **GCP**.
- **Frontend UI:** A simple web interface for users to input features and get predictions.

---

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
