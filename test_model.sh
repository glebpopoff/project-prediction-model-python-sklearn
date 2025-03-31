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