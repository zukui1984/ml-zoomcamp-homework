import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Load model
with open('pipeline_v2.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

# Define the INPUT schema
class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

# Define the OUTPUT schema
class PredictResponse(BaseModel):
    conversion_probability: float
    will_convert: bool

# FastAPI app
app = FastAPI(title="Lead Scoring Service")

@app.post("/predict")
def predict(lead: Lead) -> PredictResponse:
    # Convert to the format 
    lead_dict = lead.model_dump()
    
    # Make prediction
    probability = float(pipeline.predict_proba([lead_dict])[0, 1])
    
    return PredictResponse(
        conversion_probability=probability,
        will_convert=probability >= 0.5
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
