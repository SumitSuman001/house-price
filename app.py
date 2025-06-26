from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# Load dataset
df = pd.read_csv('Housing.csv')

# Preprocess
df.columns = df.columns.str.strip()
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

X = df.drop('price', axis=1)
y = df['price']

categorical = ['mainroad', 'guestroom', 'basement', 'hotwaterheating',
               'airconditioning', 'prefarea', 'furnishingstatus']

numeric = [col for col in X.columns if col not in categorical]

# Pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), categorical)
], remainder='passthrough')

model_pipeline = Pipeline([
    ('pre', preprocessor),
    ('model', LinearRegression())
])

model_pipeline.fit(X, y)

# Save model
joblib.dump(model_pipeline, 'model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {
            'area': float(request.form['area']),
            'bedrooms': int(request.form['bedrooms']),
            'bathrooms': int(request.form['bathrooms']),
            'stories': int(request.form['stories']),
            'parking': int(request.form['parking']),
            'mainroad': request.form['mainroad'],
            'guestroom': request.form['guestroom'],
            'basement': request.form['basement'],
            'hotwaterheating': request.form['hotwaterheating'],
            'airconditioning': request.form['airconditioning'],
            'prefarea': request.form['prefarea'],
            'furnishingstatus': request.form['furnishingstatus']
        }

        input_df = pd.DataFrame([input_data])
        model = joblib.load('model.pkl')
        prediction = model.predict(input_df)[0]

        return render_template('index.html', prediction_text=f"🏡 Predicted Price: ₹{int(prediction):,}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"⚠️ Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
