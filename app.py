import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# Set page configuration
st.set_page_config(
    page_title="Student Dropout Prediction with LIME",
    layout="centered",
    initial_sidebar_state="auto",
)

classes_dataset = ['Dropout', "Graduate"]

# Load the saved model and scaler with error handling
@st.cache_resource
def load_model():
    try:
        voting_clf = joblib.load('voting_classifier_model.pkl')  # Load the Voting Classifier
    except FileNotFoundError:
        st.error("Model file 'voting_classifier_model.pkl' not found.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

    try:
        scaler = joblib.load('min_max_scaler.pkl')  # Ensure this matches the scaler used during training
    except FileNotFoundError:
        st.error("Scaler file 'min_max_scaler.pkl' not found.")
        return None, None
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        return None, None

    return voting_clf, scaler


# Function to make predictions
def predict(input_data, voting_clf, scaler):
    # Transform the input data using the scaler
    scaled_data = scaler.transform(input_data)
    # Predict the probabilities and class
    prediction = voting_clf.predict(scaled_data)
    probability = voting_clf.predict_proba(scaled_data)
    return prediction, probability


# Function to generate LIME explanation
def lime_explanation(instance, X_train_scaled, voting_clf, feature_names, class_names):
    # Initialize LIME explainer with scaled training data
    explainer = LimeTabularExplainer(
        X_train_scaled,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification',
        discretize_continuous=True
    )

    # Explain the selected instance
    explanation = explainer.explain_instance(
        instance,
        voting_clf.predict_proba,
        num_features=4
    )

    # Generate and return the explanation figure
    fig = explanation.as_pyplot_figure()
    return fig


# Streamlit UI
st.title('Student Dropout Prediction with LIME Explanation')

# Load the model and scaler
voting_clf, scaler = load_model()

if voting_clf is not None and scaler is not None:
    # User input
    st.header('Enter Student Data')

    def get_float_input(label, min_value=None, max_value=None, default='0'):
        user_input = st.text_input(label, value=default)
        if user_input:
            try:
                value = float(user_input)
                if (min_value is not None and value < min_value) or (max_value is not None and value > max_value):
                    st.error(f"{label} must be between {min_value} and {max_value}.")
                    return None
                return value
            except ValueError:
                st.error(f"Please enter a valid number for {label}.")
                return None
        else:
            # If the user hasn't entered anything, return the default value
            try:
                return float(default)
            except ValueError:
                return None

    # Replace st.number_input with get_float_input and set initial value to '0'
    GDP = get_float_input('GDP', min_value=0.0, max_value=10.0, default='0.0')
    cu_2nd_sem_eval = get_float_input('Curricular units 2nd sem (evaluations)', min_value=0.0, max_value=20.0, default='0.0')
    cu_2nd_sem_approved = get_float_input('Curricular units 2nd sem (approved)', min_value=0.0, max_value=20.0, default='0.0')
    cu_1st_sem_eval = get_float_input('Curricular units 1st sem (evaluations)', min_value=0.0, max_value=20.0, default='0.0')
    cu_1st_sem_approved = get_float_input('Curricular units 1st sem (approved)', min_value=0.0, max_value=20.0, default='0.0')

    debtor = st.selectbox('Debtor', [0, 1])
    gender = st.selectbox('Gender', [0, 1])

    scholarship = get_float_input('Scholarship holder', min_value=0.0, max_value=20.0, default='0.0')
    special_needs = st.selectbox('Educational special needs', [0, 1])
    fees_up_to_date = st.selectbox('Tuition fees up to date', [0, 1])

    # Check if all float inputs are valid
    float_inputs = [GDP, cu_2nd_sem_eval, cu_2nd_sem_approved, cu_1st_sem_eval, cu_1st_sem_approved, scholarship]
    if all(input_val is not None for input_val in float_inputs):
        # Convert inputs into a DataFrame
        input_data = pd.DataFrame({
            'GDP': [GDP],
            'Curricular units 2nd sem (evaluations)': [cu_2nd_sem_eval],
            'Curricular units 2nd sem (approved)': [cu_2nd_sem_approved],
            'Curricular units 1st sem (evaluations)': [cu_1st_sem_eval],
            'Curricular units 1st sem (approved)': [cu_1st_sem_approved],
            'Debtor': [debtor],
            'Gender': [gender],
            'Scholarship holder': [scholarship],
            'Educational special needs': [special_needs],
            'Tuition fees up to date': [fees_up_to_date]
        })

        # Load training data and feature names for LIME
        @st.cache_data
        def load_training_data():
            try:
                X_train = pd.read_csv('Selected_Features.csv')  # Ensure this path is correct
                if 'Target' in X_train.columns:
                    X_train = X_train.drop('Target', axis=1)
                return X_train
            except FileNotFoundError:
                st.error("Training data file 'Selected_Features.csv' not found.")
                return None
            except Exception as e:
                st.error(f"Error loading training data: {e}")
                return None

        X_train = load_training_data()

        if X_train is not None:
            feature_names = X_train.columns.tolist()
            class_names = voting_clf.classes_.astype(str).tolist()  # Dynamically get class names

            # Make predictions and generate LIME explanation
            if st.button('Predict and Explain'):
                with st.spinner('Making prediction...'):
                    prediction, probability = predict(input_data, voting_clf, scaler)
                predicted_class = class_names[int(prediction[0])]
                
                # Convert predicted class (0 or 1) to "Dropout" or "Graduate"
                predicted_class_label = classes_dataset[int(prediction[0])]
                st.success(f'**Predicted Class:** {predicted_class_label}')
                st.info(f'**Prediction Probabilities:** {dict(zip(classes_dataset, probability[0]))}')

                # Generate LIME explanation
                instance = input_data.values[0]  # Select the instance to explain
                # Ensure the instance is scaled as per LIME's expectation
                try:
                    scaled_instance = scaler.transform([instance])[0]
                except Exception as e:
                    st.error(f"Error scaling input data: {e}")
                    scaled_instance = None

                if scaled_instance is not None:
                    # Scale the training data
                    try:
                        X_train_scaled = scaler.transform(X_train)
                    except Exception as e:
                        st.error(f"Error scaling training data: {e}")
                        X_train_scaled = None

                    if X_train_scaled is not None:
                        with st.spinner('Generating LIME explanation...'):
                            try:
                                fig = lime_explanation(scaled_instance, X_train_scaled, voting_clf, feature_names, class_names)
                                st.pyplot(fig)
                                st.info(f'LIME explanation for this code: Predicted Class: {predicted_class_label}')
                            except Exception as e:
                                st.error(f"Error generating LIME explanation: {e}")
        else:
            st.error("Unable to load training data. Please check the file path and format.")
    else:
        st.warning("Please fill in all the required numeric fields with valid numbers.")

# Add Footer with Copyright
st.markdown(""" 
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: #333333;
        text-align: center;
        padding: 10px;
        font-size: 12px;
    }
    </style>
    <div class="footer">
        &copy; 2024 Rakib Hossen, Sabbir Ahmed, MD. Zunead Abedin Eidmum
    </div>
    """, unsafe_allow_html=True)
