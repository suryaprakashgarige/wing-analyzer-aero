from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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
    span: float = 11.0
    ar: float = 7.2
    taper: float = 0.45
    sweep_deg: float = 3.0
    altitude: float = 2000.0
    velocity: float = 55.0
    thickness: float = 0.12
    camber: float = 0.02



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
