import turicreate as tc
import streamlit as st
import pandas as pd
import numpy as np

model = tc.load_model('wine_model')
st.set_option('deprecation.showfileUploaderEncoding', False)

def predict(model, input_df):
    predictions_df = model.predict(input_df)
    predictions = predictions_df[0]
    return predictions

def run():
    from PIL import Image
    image = Image.open('wine.jpg')
    image_hospital = Image.open('wine_tensor.jpg')
    st.image(image,use_column_width=True)
    add_selectbox = st.sidebar.selectbox("How would you like to predict?",("Individual", "Batch"))
    st.sidebar.info('This app is created to predict if a wine is good or not')
    st.sidebar.image(image_hospital,use_column_width=True)
    st.title("Predicting Wine Quality")
    if add_selectbox == 'Individual':
        fixed_acidity=st.number_input('fixed acidity' , min_value=0.1, max_value=16.0, value=0.1)
        volatile_acidity =st.number_input('volatile acidity',min_value=0.1, max_value=1.6, value=0.1)
        citric_acid = st.number_input('citric acid', min_value=0.0, max_value=1.0, value=0.1)
        residual_sugar = st.number_input('residual sugar', min_value=0.0, max_value=16.0, value=0.2)
        chlorides = st.number_input('chlorides',  min_value=0.0, max_value=1.0, value=0.0)
        free_sulfur_dioxide = st.number_input('free sulfur dioxide',  min_value=0, max_value=72, value=1)
        total_sulfur_dioxide = st.number_input('total sulfur dioxide', min_value=0, max_value=300, value=1)
        density = st.number_input('density', min_value=0.9900, max_value=1.0040, value=0.9900)
        pH = st.number_input('pH', min_value=2.0, max_value=5.0, value=2.0)
        sulphates = st.number_input('sulphates', min_value=0.0, max_value=2.0, value=0.1)
        alcohol = st.number_input('alcohol', min_value=8.0, max_value=15.0, value=8.0)
        output=""
        input_dict={'fixed acidity':fixed_acidity,'volatile acidity':volatile_acidity, 'citric acid':citric_acid, 'residual sugar':residual_sugar, 'chlorides':chlorides, 'free sulfur dioxide': free_sulfur_dioxide, 'total sulfur dioxide' :total_sulfur_dioxide, 'density':density, 'pH':pH, 'sulphates' : sulphates, 'alcohol' : alcohol}
        input_df = tc.SFrame([input_dict])
        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)
            if output == 0:
                output = 'Low'
            elif output == 1:
                output = 'High'
            st.success('It is a {} Quality Wine'.format(output))
            
    if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = tc.SFrame.read_csv(file_upload)
            predictions = model.predict(data)
            st.write(predictions[0])

run()
