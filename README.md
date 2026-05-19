# HaciendaGPT

Asistente RAG en español especializado en consultas sobre la Agencia Tributaria de España.

## Requisitos

- Python `>=3.12,<3.14`
- Poetry
- (Opcional, para crawler web) Playwright/Chromium instalado

## Setup rápido

```bash
poetry install
cp .env.example .env  # si tienes uno; si no, exporta variables manualmente
```

Variables recomendadas de entorno:

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o-mini"
export OPENAI_TEMPERATURE="0"
export TOP_K="3"
export FAISS_INDEX_PATH="./data/faiss"
# Seguridad: solo activar si el índice FAISS proviene de una fuente 100% confiable
export FAISS_TRUSTED_INDEX="true"
export DECISION_DEBUG_MODE="false"
export DECISION_STATE_DB_PATH="./data/decision_state.sqlite3"
```

## 1) Crawling de contenidos

### HTML (sitio de la AEAT)

```bash
poetry run python -m hacienda_gpt.cli.crawler --crawler web --folder ./data/html --depth 1 --mode flat
```

### PDF

```bash
poetry run python -m hacienda_gpt.cli.crawler --crawler pdf --folder ./data/pdf --depth 1
```

> El crawler guarda por defecto en carpetas con `snapshot_date` (YYYY-MM-DD).

## 2) Construcción del índice FAISS

Con embeddings locales (GPT4All):

```bash
poetry run python -m hacienda_gpt.cli.processor --content-dir ./data/html --output-dir ./data/faiss --embedder gpt4all --overwrite-output
```

Con embeddings de OpenAI:

```bash
poetry run python -m hacienda_gpt.cli.processor --content-dir ./data/html --output-dir ./data/faiss --embedder openai --overwrite-output
```

## 3) Ejecutar la UI (Streamlit)

```bash
poetry run streamlit run hacienda_gpt/ui/app.py
```

## 4) Ejecutar evaluación

```bash
poetry run python -m hacienda_gpt.cli.eval --output ./eval_results.json
```

La evaluación genera:
- score global promedio
- métricas por dimensión (`keyword_score`, `citation_score`, `grounding_score`)
- detalle por pregunta

## Notas de seguridad

- El proyecto usa `FAISS.load_local(..., allow_dangerous_deserialization=True)` por compatibilidad con índices serializados por LangChain/FAISS.
- Para reducir riesgo, la carga ahora está protegida por `FAISS_TRUSTED_INDEX`; si no está en `true`, la app rechaza cargar el índice.
- Activa `FAISS_TRUSTED_INDEX=true` **solo** con índices creados por ti o por una fuente plenamente confiable en un entorno controlado.


## Modo debug de CaseState

Si activas `DECISION_DEBUG_MODE=true`, la UI mostrará en un expander los `facts` detectados y los `facts` faltantes por turno.
El estado de cada sesión se persiste en SQLite usando `DECISION_STATE_DB_PATH`.
