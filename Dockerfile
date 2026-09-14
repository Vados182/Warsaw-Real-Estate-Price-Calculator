# 1. Obraz bazowy z gotowym środowiskiem Python 3.11
FROM python:3.11-slim

# 2. Ustawienie katalogu roboczego wewnątrz kontenera
WORKDIR /app

# 3. Instalacja niezbędnych narzędzi systemowych (potrzebnych m.in. dla PyCaret / scikit-learn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Kopiujemy plik z listą bibliotek i je instalujemy
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Kopiujemy resztę kodu aplikacji i model .pkl do kontenera
COPY . .

# 6. Informujemy, że kontener nasłuchuje na portu 8000
EXPOSE 8000

# 7. Domyślne polecenie uruchamiające serwer FastAPI po starcie kontenera
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]