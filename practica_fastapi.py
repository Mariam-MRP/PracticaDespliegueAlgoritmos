
from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel
import pandas as pd


app = FastAPI()

class User(BaseModel):
    height: float
    weight: float

# Módulo 1: Calcular IMC
@app.post("/calcular-imc")
def calcular_imc(user: User):
    if user.height is not None and user.weight is not None:
        imc = user.weight / (user.height ** 2)
        return {"mensaje": f"Tu índice de masa corporal (IMC) es: {imc:.2f}"}
    return {"mensaje": "Por favor, proporciona tu altura y peso."}

# Módulo 2: Generación de texto usando Hugging Face
@app.post("/generate-text")
def generate_text(prompt: str):
    generator = pipeline("text-generation", model="gpt2")
    generated = generator(prompt, max_length=50, num_return_sequences=1)
    return {"texto_generado": generated[0]["generated_text"]}

# Módulo 3: Clasificación de texto usando Hugging Face
@app.get("/text-classification")
def text_classification(prompt: str):
    classifier = pipeline("zero-shot-classification")
    labels = ["sports", "politics", "technology", "health"]
    result = classifier(prompt, candidate_labels=labels)
    return {"clasificación": result["labels"][0], "confianza": result["scores"][0]}

# Módulo 4: Número par:
@app.get("/par-impar")
def par_impar(number: int):
    tipo = "par" if number % 2 == 0 else "impar"
    return {"Número": number, "Tipo": tipo}
    

# Módulo 5 Multiplicación de números:
@app.get("/multiplicacion")
def multiplicacion(number1: int, number2: int):
    resultado = number1 * number2
    return {"Resultado": resultado}
