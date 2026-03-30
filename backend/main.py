from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
try:
    from .wing_model import analyze_wing
except (ImportError, ValueError):
    from wing_model import analyze_wing
import uvicorn

app = FastAPI(title="Wing Analyzer API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class WingInput(BaseModel):
    wing_type: str = "General Aviation"
    span: float = Field(default=11.0, gt=0, lt=200)
    ar: float = Field(default=7.2, gt=1, lt=50)
    taper: float = Field(default=0.45, gt=0.1, lt=1.0)
    sweep_deg: float = Field(default=3.0, ge=-15, le=65)
    altitude: float = Field(default=2000.0, ge=0, lt=15000)
    velocity: float = Field(default=55.0, gt=5, lt=400)
    thickness: float = Field(default=0.12, gt=0.04, lt=0.25)
    camber: float = Field(default=0.02, ge=0, lt=0.10)



@app.get("/")
def health():
    return {"status": "Wing Analyzer API is running", "version": "2.0"}

@app.post("/analyze")
def analyze(wing: WingInput):
    try:
        result = analyze_wing(wing.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
