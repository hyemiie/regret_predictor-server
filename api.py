from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import pickle
import json
import pandas as pd
import numpy as np
from datetime import datetime
import uvicorn

app = FastAPI(
    title="Purchase Regret Predictor API",
    description="AI-powered API to predict purchase regret probability",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


print("Loading model and metadata...")

try:
    with open('regret_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded")
except FileNotFoundError:
    print("❌ Error: regret_model.pkl not found. Run training script first!")
    model = None

try:
    with open('model_metadata.json', 'r') as f:
        metadata = json.load(f)
    print("✅ Metadata loaded")
except FileNotFoundError:
    print("❌ Error: model_metadata.json not found. Run training script first!")
    metadata = None

scaler = None
if metadata and metadata.get('needs_scaling'):
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("✅ Scaler loaded")
    except FileNotFoundError:
        print("⚠️  Warning: scaler.pkl not found but model needs scaling")



class PurchaseItem(BaseModel):
    category: str = Field(..., description="Product category (shoes, electronics, clothing, etc.)")
    price: float = Field(..., gt=0, description="Purchase price in USD")
    currency: Optional[str] = Field("USD", description="Currency code")
    
class PurchaseRequest(BaseModel):
    items: List[PurchaseItem] = Field(..., description="List of purchased items")
    hour_of_day: Optional[int] = Field(None, ge=0, le=23, description="Hour of purchase (0-23)")
    is_weekend: Optional[bool] = Field(None, description="Whether purchase is on weekend")
    time_on_page_seconds: Optional[int] = Field(300, ge=0, description="Time spent on product page")
    num_page_visits: Optional[int] = Field(1, ge=1, description="Number of times visited product page")
    was_on_sale: Optional[bool] = Field(False, description="Whether any item was on sale")

class PredictionResponse(BaseModel):
    """Prediction output"""
    regret_probability: float = Field(..., description="Probability of regretting purchase (0-1)")
    risk_level: str = Field(..., description="Risk category: low, medium, high, critical")
    recommendation: str = Field(..., description="Action recommendation")
    explanation: str = Field(..., description="Human-readable explanation")
    contributing_factors: List[str] = Field(..., description="Factors increasing regret risk")
    confidence: float = Field(..., description="Model confidence (0-1)")


def preprocess_features(item: PurchaseItem, purchase_context: PurchaseRequest = None) -> pd.DataFrame:
    """Convert a single item to model features, using purchase context if provided."""
    
    hour_of_day = purchase_context.hour_of_day if purchase_context and purchase_context.hour_of_day is not None else datetime.now().hour
    is_weekend = purchase_context.is_weekend if purchase_context and purchase_context.is_weekend is not None else datetime.now().weekday() >= 5
    time_on_page = purchase_context.time_on_page_seconds if purchase_context else 300
    num_page_visits = purchase_context.num_page_visits if purchase_context else 1
    was_on_sale = purchase_context.was_on_sale if purchase_context else False
    
    category_map = {
        'shoes': 0, 'clothing': 1, 'electronics': 2, 'books': 3,
        'basics': 4, 'beauty': 5, 'sports': 6, 'toys': 7,
    }
    category_encoded = category_map.get(item.category.lower(), 0)
    
    is_late_night = hour_of_day >= 22 or hour_of_day <= 2
    is_impulse_buy = time_on_page < 120
    
    price = item.price
    if price < 50:
        price_category_encoded = 0
    elif price < 200:
        price_category_encoded = 1
    elif price < 500:
        price_category_encoded = 2
    else:
        price_category_encoded = 3
    
    if time_on_page < 120:
        decision_speed_encoded = 0
    elif time_on_page < 600:
        decision_speed_encoded = 1
    else:
        decision_speed_encoded = 2
    
    features = {
        'category_encoded': category_encoded,
        'price': price,
        'hour_of_day': hour_of_day,
        'day_encoded': datetime.now().weekday(),
        'is_weekend': int(is_weekend),
        'is_late_night': int(is_late_night),
        'time_on_page_seconds': time_on_page,
        'is_impulse_buy': int(is_impulse_buy),
        'num_page_visits': num_page_visits,
        'was_on_sale': int(was_on_sale),
        'price_category_encoded': price_category_encoded,
        'decision_speed_encoded': decision_speed_encoded
    }
    
    feature_columns = metadata['feature_columns'] if metadata else list(features.keys())
    return pd.DataFrame([features], columns=feature_columns)


BASE_CURRENCY = "USD"

EXCHANGE_RATES = {
    "USD": 1.0,
    "EUR": 1.08,
    "GBP": 1.27,
    "MUR": 0.022,
    "NGN": 0.00077,
}

def convert_to_base(price, currency):
    """
    Converts any price into the base currency (USD).
    Falls back to 1.0 if an unknown currency is provided.
    """
    if not currency:
        return price  
    
    rate = EXCHANGE_RATES.get(currency.upper(), 1.0)
    return price * rate


HIGH_PRICE_USD_THRESHOLD = 2


def is_high_price(item):
   
    converted = convert_to_base(item.price, item.currency)
    return converted > HIGH_PRICE_USD_THRESHOLD



def generate_explanation(purchase: PurchaseRequest, probability: float) -> dict:
    hour = purchase.hour_of_day or datetime.now().hour
    weekend = purchase.is_weekend or datetime.now().weekday() >= 5
    time_on_page = purchase.time_on_page_seconds or 300
    views = purchase.num_page_visits or 1

    is_late_night = hour >= 22 or hour <= 2
    is_impulse = time_on_page < 120

    high_price_items = [
        item for item in purchase.items if is_high_price(item)
    ]
    sale_items = [i for i in purchase.items if purchase.was_on_sale]

    # Risk categories
    if probability >= 0.7:
        risk_level = "critical"
        recommendation = "High chance of regret. Waiting 24 hours is strongly advised."
    elif probability >= 0.5:
        risk_level = "high"
        recommendation = "There is a notable risk of regret. Consider checking reviews or alternatives."
    elif probability >= 0.3:
        risk_level = "medium"
        recommendation = "Moderate risk. Make sure the item fits your actual need."
    else:
        risk_level = "low"
        recommendation = "Low risk. This purchase appears consistent with your normal behaviour."

    factors = []

    if high_price_items:
        prices = ", ".join(f"{i.currency} {i.price:.2f}" for i in high_price_items)
        factors.append(f"High item cost ({prices})")

    if is_impulse:
        factors.append(f"Short viewing time ({time_on_page}s)")

    if is_late_night:
        factors.append(f"Late-night purchase at {hour}:00")

    if views == 1:
        factors.append("Item viewed only once")

    if purchase.was_on_sale:
        factors.append("Purchased during a sale")

    # More grounded, data-based explanation
    explanation = (
        f"The predicted regret probability is {probability:.0%}. "
        f"This estimate is based on measurable behaviours such as viewing time "
        f"({time_on_page}s), visit count ({views}), time of purchase ({hour}:00), "
        f"and item price levels. "
    )

    if high_price_items:
        explanation += (
            f"Historically, higher-priced items in this category lead to increased regret, "
            f"which contributed to the score. "
        )

    if is_impulse:
        explanation += "A quick decision also raised the estimate. "

    if purchase.was_on_sale:
        explanation += "Sale items tend to generate higher regret, also influencing the result. "

    return {
        "risk_level": risk_level,
        "recommendation": recommendation,
        "explanation": explanation.strip(),
        "contributing_factors": factors,
        "confidence": round(1 - abs(probability - 0.5) * 2, 3)
    }


@app.get("/")
def root():
    """Health check"""
    return {
        "status": "online",
        "service": "Purchase Regret Predictor API",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "model_type": metadata.get('model_type') if metadata else None
    }

@app.get("/health")
def health_check():
    """Detailed health check"""
    if model is None or metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training script first.")
    
    return {
        "status": "healthy",
        "model_type": metadata['model_type'],
        "features": len(metadata['feature_columns']),
        "metrics": metadata['metrics']
    }


    """
    Predict purchase regret probability
    
    Returns prediction with explanation and risk level
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        features_df = preprocess_features(purchase)
        
        if scaler is not None:
            features_scaled = scaler.transform(features_df)
            probability = model.predict_proba(features_scaled)[0][1]
        else:
            probability = model.predict_proba(features_df)[0][1]
        
        # Generate explanation
        explanation_data = generate_explanation(purchase, probability)
        
        return PredictionResponse(
            regret_probability=round(probability, 3),
            **explanation_data
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
def predict_regret(purchase: PurchaseRequest):
    print("purchase request", purchase)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        features_list = [preprocess_features(item, purchase) for item in purchase.items]
        
        features_df = pd.concat(features_list, ignore_index=True)
        
        if scaler is not None:
            features_scaled = scaler.transform(features_df)
            probabilities = model.predict_proba(features_scaled)[:, 1]
        else:
            probabilities = model.predict_proba(features_df)[:, 1]
        
        avg_probability = float(probabilities.mean())
        
        explanation_data = generate_explanation(purchase, avg_probability)
        
        return PredictionResponse(
            regret_probability=round(avg_probability, 3),
            **explanation_data
        )
    
    except Exception as e:
        print("Prediction error:", e)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch-predict")
def batch_predict(purchases: List[PurchaseRequest]):
    """
    Predict regret for multiple purchases
    
    Useful for analyzing purchase history
    """
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for purchase in purchases:
        try:
            features_df = preprocess_features(purchase)
            
            if scaler is not None:
                features_scaled = scaler.transform(features_df)
                probability = model.predict_proba(features_scaled)[0][1]
            else:
                probability = model.predict_proba(features_df)[0][1]
            
            results.append({
                "purchase": purchase.dict(),
                "regret_probability": round(probability, 3)
            })
        except Exception as e:
            results.append({
                "purchase": purchase.dict(),
                "error": str(e)
            })
    
    return {"predictions": results}

@app.get("/model-info")
def model_info():
    """Get model metadata and performance metrics"""
    
    if metadata is None:
        raise HTTPException(status_code=503, detail="Metadata not loaded")
    
    return {
        "model_type": metadata['model_type'],
        "features": metadata['feature_columns'],
        "performance": metadata['metrics'],
        "feature_count": len(metadata['feature_columns'])
    }


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Starting Purchase Regret Predictor API")
    print("="*60)
    print("\n📍 API will be available at: http://localhost:8000")
    print("📖 Documentation: http://localhost:8000/docs")
    print("📊 Alternative docs: http://localhost:8000/redoc")
    print("\n" + "="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)