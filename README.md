# Used Car Price Prediction

## Project Overview

This project is an end-to-end Machine Learning web application designed to predict the selling price of used cars based on various vehicle specifications and features. It covers data collection, preprocessing, model building, evaluation, model saving, and deployment using Streamlit with SQLite prediction tracking

## Dataset

The dataset contains information about used cars, including:

* Car Brand
* Car Age(Years)
* Kilometers Driven
* Fuel Type
* Transmission Type
* Mileage(kmpl)
* Engine Capacity (CC)
* Number of Previous Owners

The target variable is:

* Selling Price

The dataset contains 15,411 records.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Joblib



## Machine Learning Model
- **Algorithm:** Linear Regression
- **Preprocessing:** `OneHotEncoder` via `ColumnTransformer` for categorical features (`fuel_type`, `transmission`, `brand`).
- **Pipeline Structure:** End-to-end Scikit-Learn `Pipeline` integrating categorical preprocessing and the linear regression model to prevent data leakage and ensure smooth inference.

The dataset is divided into:

* 80% Training Data
* 20% Testing Data

## Model Evaluation

The model is evaluated using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

### Results

* MAE: 366398.22
* MSE: 169436328902.93
* RMSE: 411626.44
* R² Score: -61.1785

## Model Saving

The trained model pipeline is serialized using Joblib and saved as a binary artifact in the `model/` directory:
- Path: `model/car_price_model.pkl`

## Streamlit Application

An interactive web application built using Streamlit that provides:
- A clean interface with input fields for entering car details.
- Real-time selling price predictions based on input values.
- Integrated SQLite database (`predictions.db`) to log every prediction request along with timestamps and input parameters.
- Prediction history viewer embedded directly within the app.

### Input Features

* Car Brand
* Car Age(Years)
* Kilometers Driven
* Fuel Type
* Transmission Type
* Mileage(kmpl)
* Engine Capaacity(CC)
* Number of Previous Owners


The application displays the predicted used-car selling price.

## Project Structure

```text

used_car_prediction/
├── dataset/
│   └── used_car_data.csv
├── model/
│   └── car_price_model.pkl
├── screenshots/
│   
├── .gitignore
├── README.md
├── app.py
├── predictions.db
├── requirements.txt
└── train_model.py
```

## How to Run the Project

### 1. Install required libraries

```bash
python -m pip install -r requirements.txt
```

### 2. Train the model

```bash
python train_model.py
```

### 3. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will open in the browser.

## Conclusion

The project demonstrates an end-to-end Machine Learning workflow for predicting used-car selling prices, from dataset preprocessing and model training to model evaluation, model saving, and deployment using Streamlit.