# main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Union
import os, json
import dotenv

dotenv.load_dotenv()

# ---------- App & CORS ----------
app = FastAPI(title="Recomendador Mascotas (LLM-only)", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

# ---------- Modelos ----------
class Preferencias(BaseModel):
    tamanio: Optional[str] = None
    edad: Optional[Union[int, str]] = None
    es_jugueton: Optional[bool] = None
    es_tranquilo: Optional[bool] = None
    convive_otras_mascotas: Optional[bool] = None
    convive_ninos: Optional[bool] = None
    nivel_energia: Optional[str] = None
    class Config:
        extra = "allow"

class RecoRequest(BaseModel):
    preferencias: Preferencias
    mascotas: List[dict]

class RecoResponse(BaseModel):
    explicacion: str
    recomendadas: List[dict]

# ---------- OpenAI (SDK v1) ----------
from openai import OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Define OPENAI_API_KEY en el entorno.")
client = OpenAI(api_key=OPENAI_API_KEY)

def build_prompt(preferencias: dict, mascotas: List[dict], k: int = 3) -> str:
    return f"""
Eres un sistema de recomendación de adopción. A partir de las preferencias del usuario y un catálogo de mascotas,
elige como máximo {k} mascotas que mejor coincidan. Devuelve SIEMPRE SOLO JSON válido (sin texto extra) con este formato EXACTO:

{{
  "explicacion": "texto breve justificando por qué elegiste esas mascotas y empieza con 'Basado en tus preferencias...' y habla en plural "hemos seleccionado..."",
  "recomendadas": [
    {{"id": <id>, "nombre": "<nombre>", "tamanio": "<pequeño|mediano|grande>", "edad": "<cachorro|adulto|senior>", "personalidad": "<...>", "convive_con_otros": "<sí|no>"}}
  ]
}}

Preferencias del usuario:
{json.dumps(preferencias, ensure_ascii=False, indent=2)}

Catálogo de mascotas (elige máximo {k}):
{json.dumps(mascotas, ensure_ascii=False)[:20000]}

Instrucciones:
- Asegúrate de que las recomendaciones provienen del catálogo proporcionado (no inventes).
- Devuelve como mucho {k} elementos en "recomendadas".
- Responde SOLO el JSON, sin comentarios, sin backticks y sin texto adicional.
""".strip()

def call_openai(prompt: str) -> dict:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    content = resp.choices[0].message.content or ""

    # Limpia posibles fences accidentales
    if content.strip().startswith("```"):
        content = content.strip().strip("`")
        # también cubre casos como ```json ... ```
        if content.lstrip().startswith("json"):
            content = content.split("\n", 1)[1] if "\n" in content else ""

    try:
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("La respuesta no es un objeto JSON.")
        recs = data.get("recomendadas", [])
        if not isinstance(recs, list):
            recs = []
        data["recomendadas"] = recs[:3]
        data["explicacion"] = data.get("explicacion", "")
        return data
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Respuesta LLM inválida: {str(e)}")

# ---------- Endpoint ----------
@app.post("/recomendar", response_model=RecoResponse)
async def recomendar(req: RecoRequest, request: Request):
    if not req.mascotas or len(req.mascotas) == 0:
        raise HTTPException(status_code=400, detail="Debes enviar 'mascotas' como lista no vacía.")

    prefs = req.preferencias.dict(exclude_none=True)
    prompt = build_prompt(prefs, req.mascotas, k=3)
    data = call_openai(prompt)
    return RecoResponse(explicacion=data["explicacion"], recomendadas=data["recomendadas"])

# ---- main ----
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
