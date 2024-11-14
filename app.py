from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model
model = joblib.load('stresslevelmodel.pkl')

value_mapping = {
    "snoring": {
        "Not at all": 5.0,
        "Sometimes": 30.0,
        "Often": 70.0,
        "Always": 100.0,
    },
    "respiration": {
        "Normal and steady": 16.0,
        "Slightly rapid": 20.0,
        "Often rapid": 25.0,
        "Very rapid": 30.0,
    },
    "body_temperature": {
        "Normal": 98.5,
        "A bit warm": 100.0,
        "Quite warm": 102.0,
        "Very warm": 110.0,
    },
    "limb_movement": {
        "Not at all": 0.0,
        "Slightly": 4.0,
        "Moderately": 7.0,
        "A lot": 10.0,
    },
    "blood_oxygen": {
        "Never": 98.0,
        "Rarely": 92.0,
        "Sometimes": 87.0,
        "Often": 82.0,
    },
    "eye_movement": {
        "Not at all": 7.0,
        "Sometimes": 20.0,
        "Often": 30.0,
        "Very often": 50.0,
    },
    "sleep": {
        "Less than 4 hours": 3.0,
        "4-6 hours": 5.0,
        "6-8 hours": 7.0,
        "8+ hours": 13.0,
    },
    "heart_rate": {
        "Very slow": 50.0,
        "Slow": 70.0,
        "Moderate": 90.0,
        "Fast": 130.0,
    },
}

@app.route('/')
def home():
    return render_template('index.html',prediction_text = False)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = []
        for feature, mapping in value_mapping.items():
            user_input = request.form[feature]
            features.append(mapping[user_input])

        data = np.array([features])

        prediction = model.predict(data)[0]
        
        stress_levels = {0: 'Low/Normal', 1: 'Medium Low', 2: 'Medium', 3: 'Medium - High', 4: 'High'}
        result = stress_levels.get(prediction, "Unknown")

        return render_template('index.html', prediction_text=f'Predicted Stress Level: {result}')
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
