from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from datetime import datetime
import os

app = Flask(__name__)

# Function to train and save a fallback model
def train_and_save_model():
    # Generate example training data
    X, y = make_classification(n_samples=1000, n_features=22, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a DecisionTreeClassifier
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, 'fraud_detection_model.pkl')
    print("Fallback model trained and saved as 'fraud_detection_model.pkl'.")
    return model

# Load the trained model or train a fallback model
try:
    if os.path.exists('fraud_detection_model.pkl'):
        model = joblib.load('fraud_detection_model.pkl')
    else:
        print("Model file not found. Training a new model.")
        model = train_and_save_model()
except Exception as e:
    print(f"Error loading the model: {e}. Training a fallback model.")
    model = train_and_save_model()

# Load encoders for categorical columns
encoders = {}
encoder_files = ['merchant', 'category', 'first', 'last', 'gender', 'street', 'city', 'state', 'job']
for encoder_name in encoder_files:
    try:
        encoders[encoder_name] = joblib.load(f'encoders/{encoder_name}_encoder.pkl')
    except FileNotFoundError:
        print(f"Warning: Encoder file for {encoder_name} not found. Skipping.")

def preprocess_features(features_df):
    # Convert 'trans_date_trans_time' and 'dob' to datetime and extract features
    if 'trans_date_trans_time' in features_df:
        features_df['trans_date_trans_time'] = pd.to_datetime(features_df['trans_date_trans_time'], errors='coerce')
        features_df['year'] = features_df['trans_date_trans_time'].dt.year
        features_df['month'] = features_df['trans_date_trans_time'].dt.month
        features_df['day'] = features_df['trans_date_trans_time'].dt.day
        features_df['hour'] = features_df['trans_date_trans_time'].dt.hour
        features_df['weekday'] = features_df['trans_date_trans_time'].dt.weekday
        features_df.drop('trans_date_trans_time', axis=1, inplace=True)

    if 'dob' in features_df:
        features_df['dob'] = pd.to_datetime(features_df['dob'], errors='coerce')
        current_year = datetime.now().year
        features_df['age'] = current_year - features_df['dob'].dt.year
        features_df.drop('dob', axis=1, inplace=True)

    # Encode categorical features using pre-loaded LabelEncoders
    for col, encoder in encoders.items():
        if col in features_df.columns:
            try:
                features_df[col] = encoder.transform(features_df[col])
            except Exception as e:
                raise ValueError(f"Encoding failed for column '{col}': {e}")

    # Ensure features are in the correct order as the model expects
    feature_order = ['cc_num', 'merchant', 'category', 'amt', 'first', 'last', 'gender',
                     'street', 'city', 'state', 'zip', 'lat', 'long', 'city_pop', 'job',
                     'unix_time', 'year', 'month', 'day', 'hour', 'weekday', 'age']
    try:
        features_df = features_df[feature_order]
    except KeyError as e:
        missing_cols = list(set(feature_order) - set(features_df.columns))
        raise ValueError(f"Missing required columns: {missing_cols}")

    return features_df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        features_df = pd.DataFrame([data])

        if features_df.isnull().any().any():
            raise ValueError("All fields must be filled out.")

        processed_features = preprocess_features(features_df)

        # Adjust input features to match model's expected shape
        processed_features = processed_features.iloc[:, :22]  # Ensure it has 22 features

        prediction = model.predict(processed_features)
        prediction_proba = model.predict_proba(processed_features)
        fraud_probability = prediction_proba[0][1]
        return render_template('result.html', prediction=int(prediction[0]), probability=fraud_probability)
    except ValueError as ve:
        return render_template('index.html', data=request.form, error=str(ve))
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
    