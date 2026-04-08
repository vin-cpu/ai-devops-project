from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load model
model = joblib.load("model.pkl")

# Define input format properly
class InputData(BaseModel):
    data: list

@app.get("/")
def home():
    return {"message": "AI DevOps API is running"}

class_labels = {
    0: "Class 1 (Setosa)",
    1: "Class 2 (Versicolor)",
    2: "Class 3 (Virginica)"
}
@app.post("/predict")
def predict(input: InputData):
    pred = model.predict([input.data])[0]
    return {
        "prediction_numeric": int(pred),
        "prediction_label": class_labels[pred]
    }

import os

port = int(os.environ.get("PORT", 8000))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
