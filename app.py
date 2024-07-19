# streamlit_app.py
import streamlit as st
import joblib
import pandas as pd

# Load the model and label encoders
model = joblib.load('sales_prediction_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Define user input form
st.title('BigMart Sales Prediction')
st.header('Enter Item and Outlet Details')

item_weight = st.number_input('Item Weight', min_value=0.0)
item_fat_content = st.selectbox('Item Fat Content', label_encoders['Item_Fat_Content'].classes_)
item_visibility = st.number_input('Item Visibility', min_value=0.0)
item_type = st.selectbox('Item Type', label_encoders['Item_Type'].classes_)
item_mrp = st.number_input('Item MRP', min_value=0.0)
outlet_identifier = st.selectbox('Outlet Identifier', label_encoders['Outlet_Identifier'].classes_)
outlet_establishment_year = st.number_input('Outlet Establishment Year', min_value=1900, max_value=2023, value=2000)
outlet_size = st.selectbox('Outlet Size', label_encoders['Outlet_Size'].classes_)
outlet_location_type = st.selectbox('Outlet Location Type', label_encoders['Outlet_Location_Type'].classes_)
outlet_type = st.selectbox('Outlet Type', label_encoders['Outlet_Type'].classes_)

# Preprocess user input
input_data = pd.DataFrame({
    'Item_Weight': [item_weight],
    'Item_Fat_Content': [item_fat_content],
    'Item_Visibility': [item_visibility],
    'Item_Type': [item_type],
    'Item_MRP': [item_mrp],
    'Outlet_Identifier': [outlet_identifier],
    'Outlet_Establishment_Year': [2023 - outlet_establishment_year],
    'Outlet_Size': [outlet_size],
    'Outlet_Location_Type': [outlet_location_type],
    'Outlet_Type': [outlet_type]
})

# Encode categorical variables
for column in label_encoders:
    input_data[column] = label_encoders[column].transform(input_data[column])

# Make prediction
if st.button('Predict Sales'):
    prediction = model.predict(input_data)
    st.write(f'Predicted Sales: {prediction[0]:.2f}')
