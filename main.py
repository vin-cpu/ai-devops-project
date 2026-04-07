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

@app.post("/predict")
def predict(input: InputData):
    prediction = model.predict([input.data])
    return {"prediction": prediction.tolist()}

import os

port = int(os.environ.get("PORT", 8000))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
