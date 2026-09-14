import os
import pandas as pd
from contextlib import asynccontextmanager
from typing import Optional, List
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from pycaret.regression import load_model, predict_model
from openai import OpenAI
import instructor
from langfuse import observe, Langfuse

load_dotenv()

# --- 1. DANE POMOCNICZE I MODELE PYDANTIC ---
DZIELNICE = [
    'Mokotów', 'Wola', 'Ursynów', 'Praga-Południe', 'Bielany', 'Śródmieście', 
    'Targówek', 'Bemowo', 'Białołęka', 'Ochota', 'Wawer', 'Żoliborz', 
    'Ursus', 'Włochy', 'Praga-Północ', 'Wesoła', 'Rembertów', 'Wilanów', 'Inna'
]

# Schemat zapytania dla wyceny bezpośredniej
class PropertyInput(BaseModel):
    dzielnica: str = Field(..., example="Mokotów", description="Dzielnica Warszawy")
    metraz: float = Field(..., gt=10, lt=300, example=50.0, description="Metraż w m²")
    pokoje: int = Field(..., ge=1, le=10, example=2, description="Liczba pokoi")
    pietro: int = Field(default=2, ge=0, le=40, example=3, description="Piętro (domyślnie 2)")

# Schemat wyciągania danych przez LLM (instructor)
class PropertyExtractor(BaseModel):
    dzielnica: Optional[str] = Field(default=None, description="Dzielnica Warszawy")
    metraz: Optional[float] = Field(default=None, description="Powierzchnia w m2")
    pokoje: Optional[int] = Field(default=None, description="Liczba pokoi")
    pietro: Optional[int] = Field(default=2, description="Piętro")

# Schemat zapytania dla analizy tekstu
class TextInput(BaseModel):
    text: str = Field(..., example="Sprzedam bezpośrednio mieszkanie na Woli, 45 metrów, 2 pokoje, 3 piętro.")

# Schemat odpowiedzi z wyceną
class PredictionResponse(BaseModel):
    dzielnica: str
    metraz: float
    pokoje: int
    pietro: int
    szacowana_cena: float
    cena_za_m2: float

# --- 2. ZARZĄDZANIE CYKLEM ŻYCIA APLIKACJI (MODEL ML) ---
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Wczytujemy model PyCaret raz przy starcie serwera
    try:
        ml_models["pipeline"] = load_model("real_estate_model")
        print("✅ Model PyCaret pomyślnie załadowany do pamięci RAM.")
    except Exception as e:
        print(f"❌ Błąd podczas ładowania modelu ML: {e}")
        ml_models["pipeline"] = None
    yield
    ml_models.clear()

app = FastAPI(
    title="Kalkulator Cen Mieszkań w Warszawie - API",
    description="REST API do szacowania wartości nieruchomości za pomocą PyCaret i LLM (OpenAI GPT-4o-mini).",
    version="1.0.0",
    lifespan=lifespan
)

# --- 3. POMOCNICZA FUNKCJA WYCENIAJĄCA ---
def calculate_price(dzielnica: str, metraz: float, pokoje: int, pietro: int) -> tuple[float, float]:
    pipeline = ml_models.get("pipeline")
    if not pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Model ML nie jest dostępny."
        )
    
    df = pd.DataFrame([{
        'Dzielnica': dzielnica,
        'Metraż': metraz,
        'Pokoje': pokoje,
        'Piętro': pietro
    }])
    
    prediction = predict_model(pipeline, data=df)
    cena = float(prediction['prediction_label'].iloc[0])
    cena_m2 = cena / metraz
    return round(cena, 2), round(cena_m2, 2)

@observe(name="extract_property_data")
def extract_data_from_text(text: str) -> PropertyExtractor:
    client = instructor.from_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
    return client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=PropertyExtractor,
        messages=[
            {
                "role": "system",
                "content": "Wyciągnij parametry mieszkania w Warszawie z tekstu: dzielnicę, metraż (m²), liczbę pokoi i piętro."
            },
            {"role": "user", "content": text}
        ]
    )

# --- 4. ENDPOINTY REST API ---

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": ml_models.get("pipeline") is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict_from_features(payload: PropertyInput):
    """Szacuje cenę mieszkania na podstawie podanych parametrów ręcznych."""
    cena, cena_m2 = calculate_price(
        payload.dzielnica, 
        payload.metraz, 
        payload.pokoje, 
        payload.pietro
    )
    
    return PredictionResponse(
        dzielnica=payload.dzielnica,
        metraz=payload.metraz,
        pokoje=payload.pokoje,
        pietro=payload.pietro,
        szacowana_cena=cena,
        cena_za_m2=cena_m2
    )

@app.post("/predict-from-text", response_model=PredictionResponse)
def predict_from_text(payload: TextInput):
    """Analizuje surowy tekst ogłoszenia za pomocą GPT-4o-mini, po czym wycenia mieszkanie."""
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Tekst nie może być pusty.")
    
    # Ekstrakcja danych przez LLM
    extracted = extract_data_from_text(payload.text)
    
    # Walidacja wyciągniętych danych
    missing_fields = []
    if not extracted.dzielnica or extracted.dzielnica not in DZIELNICE:
        missing_fields.append("Dzielnica")
    if not extracted.metraz:
        missing_fields.append("Metraż")
    if not extracted.pokoje:
        missing_fields.append("Liczba pokoi")

    if missing_fields:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Nie udało się odczytać wymaganych parametrów z tekstu. Brakuje: {', '.join(missing_fields)}"
        )
    
    # Przeliczenie ceny
    cena, cena_m2 = calculate_price(
        extracted.dzielnica, 
        extracted.metraz, 
        extracted.pokoje, 
        extracted.pietro or 2
    )
    
    return PredictionResponse(
        dzielnica=extracted.dzielnica,
        metraz=extracted.metraz,
        pokoje=extracted.pokoje,
        pietro=extracted.pietro or 2,
        szacowana_cena=cena,
        cena_za_m2=cena_m2
    )
    
@app.get("/")
def main_redirect():
    """Przekierowanie z adresu głównego bezpośrednio do dokumentacji Swagger."""
    return RedirectResponse(url="/docs")