import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

class ProjectSuccessPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def train(self, data_path):
        # Load and prepare data
        df = pd.read_csv(data_path)
        
        # Features and target
        features = ['completion_rate', 'hours_spent', 'team_size', 'complexity_score', 
                   'on_time_delivery', 'previous_similar_projects']
        X = df[features]
        y = df['success_rating']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = LogisticRegression(random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        return accuracy, report
    
    def predict(self, features_dict):
        features = pd.DataFrame([features_dict])
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)
        probability = self.model.predict_proba(features_scaled)
        return prediction[0], probability[0]
    
    def save_model(self, model_path='model.pkl'):
        with open(model_path, 'wb') as f:
            pickle.dump((self.model, self.scaler), f)
    
    def load_model(self, model_path='model.pkl'):
        with open(model_path, 'rb') as f:
            self.model, self.scaler = pickle.load(f)

# Initialize predictor
predictor = ProjectSuccessPredictor()

@app.route('/train', methods=['POST'])
def train_model():
    try:
        accuracy, report = predictor.train('data/user_project_history.csv')
        predictor.save_model()
        return jsonify({
            'status': 'success',
            'accuracy': accuracy,
            'report': report
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        prediction, probability = predictor.predict(data)
        return jsonify({
            'status': 'success',
            'prediction': int(prediction),
            'probability': probability.tolist()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    # Load model if exists
    if os.path.exists('model.pkl'):
        predictor.load_model()
    app.run(debug=True, port=5000)
