import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


st.title("House Prices Prediction using Linear Regression")
st.write("""
This app predicts the **House Prices** based on their square footage, number of bedrooms, and number of bathrooms.
""")

# Loading data
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')
    return df

df = load_data()

# Data preprocessing
df = df.dropna(subset=['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice'])
features = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
target = df['SalePrice']

# Displaying  raw data
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(df)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Training the model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write(f"Mean Squared Error: {mse}")
st.write(f"R-squared: {r2}")

# Visualizing the results
st.subheader('Actual vs Predicted Prices')
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, color='blue', label='Predicted Prices')
ax.scatter(y_test, y_test, color='red', alpha=0.5, label='Actual Prices')
ax.set_xlabel("Actual Prices")
ax.set_ylabel("Predicted Prices")
ax.set_title("Actual vs Predicted Prices")
ax.legend()
st.pyplot(fig)

st.subheader('Residuals Distribution')
residuals = y_test - y_pred
fig, ax = plt.subplots()
sns.histplot(residuals, kde=True, ax=ax)
ax.set_xlabel("Residuals")
ax.set_title("Residuals Distribution")
st.pyplot(fig)

# Interactive input for custom prediction
st.sidebar.header('User Input Features')
def user_input_features():
    GrLivArea = st.sidebar.slider('Living Area (sq ft)', int(df.GrLivArea.min()), int(df.GrLivArea.max()), int(df.GrLivArea.mean()))
    BedroomAbvGr = st.sidebar.slider('Number of Bedrooms', int(df.BedroomAbvGr.min()), int(df.BedroomAbvGr.max()), int(df.BedroomAbvGr.mean()))
    FullBath = st.sidebar.slider('Number of Bathrooms', int(df.FullBath.min()), int(df.FullBath.max()), int(df.FullBath.mean()))
    data = {'GrLivArea': GrLivArea,
            'BedroomAbvGr': BedroomAbvGr,
            'FullBath': FullBath}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Displaying the user input features
st.subheader('User Input features')
st.write(input_df)

# Predicting the price for user input
if st.button('Predict House Price'):
    prediction = model.predict(input_df)
    st.write(f'Predicted House Price: ${prediction[0]:,.2f}')
