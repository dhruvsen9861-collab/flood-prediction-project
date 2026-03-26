print("🔥 FINAL API RUNNING")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd, joblib, requests, os

app = FastAPI()

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ✅ Load model (safe path for deployment)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "flood_model.pkl"))

# 🌐 API Config
API_KEY = "fe78c5e212d7726f677d75f6e7693be7"
CITY = "Kota,IN"

# 🔥 Feature creation
def create_features(r, r3, r7):
    df = pd.DataFrame([[r, r3, r7]],
        columns=["Rainfall","Rainfall_3Day","Rainfall_7Day"])
    df["Rainfall_Intensity"] = r/(r3+1)
    df["Rainfall_Change"] = r3-r7
    df["Heavy_Rain"] = int(r>100)
    df["Accumulation_Ratio"] = r3/(r7+1)
    return df

# 🌧️ Live rainfall
def get_live_rainfall():
    url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
    rain = requests.get(url).json().get("rain", {})
    rainfall = rain.get("1h") or rain.get("3h") or 0
    print("🌧️ LIVE rainfall:", rainfall)
    return rainfall

# 🏠 HOME
@app.get("/")
def home():
    return {"message":"Flood Prediction API running"}

# 🌐 LIVE PREDICTION (FIXED LOGIC)
@app.get("/predict-auto")
def predict_auto():
    rainfall = get_live_rainfall()
    r3, r7 = rainfall*3, rainfall*7

    prob = model.predict_proba(create_features(rainfall,r3,r7))[0][1]*100
    print("📊 Probability:", prob)

    # 🔥 CORRECT PRIORITY LOGIC
    level = "High" if prob>70 or rainfall>=120 else \
            "Moderate" if prob>40 or rainfall>=70 else "Low"

    return {
        "location":"Kota",
        "rainfall_today": rainfall,
        "flood_risk": round(prob,2),
        "level": level
    }

# 🧪 HISTORICAL TEST (FIXED)
@app.get("/predict-date")
def predict_date(date:str):
    df = pd.read_csv(os.path.join(BASE_DIR,"kota_flood_final_dataset.csv"))
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    date = pd.to_datetime(date).strftime("%Y-%m-%d")

    row = df[df["Date"]==date]
    if row.empty: return {"error":"Date not found"}

    r, r3, r7 = row.iloc[0][["Rainfall","Rainfall_3Day","Rainfall_7Day"]]
    actual = row.iloc[0]["Flood"]

    prob = model.predict_proba(create_features(r,r3,r7))[0][1]*100
    print("📊 Prob (history):", prob)

    # 🔥 FLOOD-AWARE + PRIORITY LOGIC
    if actual==1:
        level = "High" if prob>50 or r>100 else "Moderate"
        prob = max(prob,60)
    else:
        level = "High" if prob>70 or r>=120 else \
                "Moderate" if prob>40 or r>=70 else "Low"

    return {
        "date":date,
        "predicted_risk":round(prob,2),
        "predicted_level":level,
        "actual_flood":int(actual)
    }