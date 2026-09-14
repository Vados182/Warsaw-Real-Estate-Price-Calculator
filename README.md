# 🏙️ Warsaw Real Estate Price Predictor (REST API)

A production-ready REST API designed to estimate apartment prices in Warsaw, Poland. Built using **FastAPI**, **PyCaret (Machine Learning)**, **OpenAI GPT-4o-mini**, and **Instructor**, this service supports both structured inputs and unstructured property listing descriptions.

---

## 📌 Features

- **Machine Learning Pricing Engine:** Predicts overall apartment prices and price per square meter ($m^2$) based on a trained PyCaret regression model.
- **Natural Language Parsing:** Accepts raw listing descriptions (text) and extracts structured parameters (`dzielnica`, `metraz`, `pokoje`, `pietro`) using OpenAI's `gpt-4o-mini` with strict Pydantic parsing via `instructor`.
- **High Performance:** Utilizes FastAPI's `lifespan` context manager for efficient, single-load model initialization on startup.
- **Automated API Documentation:** Fully interactive OpenAPI / Swagger UI provided natively by FastAPI.
- **Containerization Ready:** Includes a production-grade `Dockerfile` and `.dockerignore` for seamless cloud deployment.

---

## 🛠️ Tech Stack

- **Language:** Python 3.11
- **API Framework:** FastAPI, Uvicorn
- **Machine Learning:** PyCaret, Scikit-learn
- **LLM & Structured Extraction:** OpenAI API (`gpt-4o-mini`), Instructor, Pydantic
- **Containerization:** Docker

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- OpenAI API Key

### Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_GITHUB_USERNAME/warsaw-real-estate-api.git](https://github.com/YOUR_GITHUB_USERNAME/warsaw-real-estate-api.git)
   cd warsaw-real-estate-api