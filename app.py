import streamlit as st
import numpy as np
import pickle

# Load the saved model
model = pickle.load(open('diabetes.pickle', 'rb'))

# Define the app


def main():
    # Set the app title
    st.title('Diabetes Prediction App')

    # Add a brief description
    st.write(
        'This app predicts whether a person has diabetes or not based on their medical attributes.')

    # Add inputs for the features
    Pregnancies = st.slider('Pregnancies', 1, 17, 5)
    Glucose = st.slider('Glucose', 44, 199, 100)
    BloodPressure = st.slider('BloodPressure', 24, 122, 50)
    SkinThickness = st.slider('SkinThickness', 7, 99, 40)
    Insulin = st.slider('Insulin', 14, 846, 300)
    BMI = st.slider('BMI', 18, 67, 35)
    DiabetesPedigreeFunction = st.slider(
        'DiabetesPedigreeFunction', 0.07, 2.42, 1.0)
    Age = st.slider('Age', 21, 81, 1)

    input_features = np.array(
        [[Pregnancies, Glucose, BloodPressure,	SkinThickness,	Insulin,	BMI,	DiabetesPedigreeFunction,	Age]])

    # Predict the wine quality

    prediction = model.predict(input_features)[0]

    if st.button("Predict"):

        if (prediction == 1):
            info = 'Diabetic'
            st.success('Prediction: {}'.format(info))

        else:
            info = 'Not Diabetic'
            st.error('Prediction: {}'.format(info))


# Run the app
if __name__ == '__main__':
    main()
