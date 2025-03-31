# Project Success Prediction Model

A machine learning service that predicts the likelihood of project success based on historical project data and various metrics.

## Features

### Input Features
- `completion_rate` (float): Historical completion rate (0.0-1.0)
- `hours_spent` (int): Total hours spent on the project
- `team_size` (int): Number of team members
- `complexity_score` (int): Project complexity rating (1-10)
- `on_time_delivery` (int): Binary indicator of on-time delivery (0 or 1)
- `previous_similar_projects` (int): Number of similar projects completed

### Model Details
- Algorithm: Logistic Regression
- Feature Scaling: StandardScaler
- Train/Test Split: 80/20
- Output: Binary classification (0: Failure, 1: Success) with probability scores

## Project Structure
```
.
├── README.md
├── requirements.txt
├── prediction_service.py
└── data
    └── user_project_history.csv
```

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip3 install -r requirements.txt
```

## Running the Service

Start the Flask server:
```bash
python3 prediction_service.py
```
The service will run on http://localhost:5000

## API Documentation

### 1. Train Model
Trains the model using the provided dataset.

**Endpoint:** `POST /train`

**Example Request:**
```bash
curl -X POST http://localhost:5000/train
```

**Example Response:**
```json
{
  "status": "success",
  "accuracy": 1.0,
  "report": "Classification report with precision, recall, and f1-score..."
}
```

### 2. Make Prediction
Predicts project success probability based on provided features.

**Endpoint:** `POST /predict`

**Request Body:**
```json
{
  "completion_rate": 0.85,
  "hours_spent": 100,
  "team_size": 3,
  "complexity_score": 6,
  "on_time_delivery": 1,
  "previous_similar_projects": 2
}
```

**Example Request:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "completion_rate": 0.85,
    "hours_spent": 100,
    "team_size": 3,
    "complexity_score": 6,
    "on_time_delivery": 1,
    "previous_similar_projects": 2
  }'
```

**Example Response:**
```json
{
  "status": "success",
  "prediction": 1,
  "probability": [0.105, 0.895]
}
```
- `prediction`: 0 (Failure) or 1 (Success)
- `probability`: Array of probabilities [P(Failure), P(Success)]

## Model Persistence
- The trained model is automatically saved to `model.pkl`
- The service automatically loads the existing model on startup if available
- Retrain the model using the `/train` endpoint when new data is available

## Testing the Service

1. Start the service:
```bash
python3 prediction_service.py
```

2. Train the model:
```bash
curl -X POST http://localhost:5000/train
```

3. Make predictions:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "completion_rate": 0.85,
    "hours_spent": 100,
    "team_size": 3,
    "complexity_score": 6,
    "on_time_delivery": 1,
    "previous_similar_projects": 2
  }'
```

## Sample Data Format
The training data (`user_project_history.csv`) should follow this format:
```csv
user_id,project_id,completion_rate,hours_spent,team_size,complexity_score,on_time_delivery,previous_similar_projects,success_rating
1,101,0.95,120,3,7,1,2,1
```

## Best Practices
1. Always train the model before making predictions
2. Ensure input features are within reasonable ranges:
   - completion_rate: 0.0 to 1.0
   - complexity_score: 1 to 10
   - on_time_delivery: 0 or 1
3. Regularly update the training data with new project outcomes
4. Monitor model performance metrics over time

## Production Deployment
For production deployment:
1. Use a production-grade WSGI server (e.g., Gunicorn)
2. Implement proper error handling and input validation
3. Add authentication and rate limiting
4. Set up monitoring and logging
5. Use environment variables for configuration
