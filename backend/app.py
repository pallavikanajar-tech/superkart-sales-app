
import os
import gdown
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
superkart_api = Flask("superkart_sales_api")
CORS(superkart_api)

# Google Drive public model link
GOOGLE_DRIVE_MODEL_URL = "https://drive.google.com/file/d/1DXtUE7A-OYzQf8rz4TG6_C4Izjdf6uAk/view?usp=drive_link"

MODEL_PATH = "superkart_random_forest_tuned.joblib"

# Download the model if it is not already available
if not os.path.exists(MODEL_PATH):
		gdown.download(GOOGLE_DRIVE_MODEL_URL,
				MODEL_PATH,
				quiet=False
				)

# Load the trained Tuned Random Forest model
model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")

# Health Check Route
@superkart_api.get('/')
def home():
    return " Welcome to the SuperKart Sales Forecasting API"

# Prediction Route
@superkart_api.post('/v1/predict')
def predict_sales():

    try:
        # Read Incoming JSON
        data = request.get_json()

        print("Incoming Request:")
        print(data)

        # Required Input Fields
        required_fields = [
            'Product_Weight',
            'Product_Sugar_Content',
            'Product_Allocated_Area',
            'Product_MRP',
            'Store_Size',
            'Store_Location_City_Type',
            'Store_Type',
            'Store_Age_Years',
            'Product_Type_Category'
        ]

        # Check for missing fields
        missing_fields = [
            field for field in required_fields
            if field not in data
        ]

        if missing_fields:
            return jsonify({
                "error": f"Missing fields: {missing_fields}"
            }), 400

        # Prepare Input Record
        sample = {
                'Product_Weight':float(data['Product_Weight']),
                'Product_Sugar_Content':data['Product_Sugar_Content'],
                'Product_Allocated_Area':float(data['Product_Allocated_Area']),
                'Product_MRP':float(data['Product_MRP']),
                'Store_Size':data['Store_Size'],
                'Store_Location_City_Type':data['Store_Location_City_Type'],
                'Store_Type':data['Store_Type'],
                'Store_Age_Years':int(data['Store_Age_Years']),
                'Product_Type_Category':data['Product_Type_Category']
                }


        # Convert to DataFrame
        input_df = pd.DataFrame([sample])

        print("Transformed Input:")
        print(input_df)

        # Generate Prediction
        prediction = model.predict(input_df)[0]

        # Return Prediction
        return jsonify({
            "Predicted_Sales": round(float(prediction), 2)
        })

    except Exception as e:

        print("Prediction Error:", str(e))

        return jsonify({
            "error": str(e)
        }), 500


# Run Locally
if __name__ == "__main__":
    superkart_api.run(
        host="0.0.0.0",
        port=7860,
        debug=True
    )
