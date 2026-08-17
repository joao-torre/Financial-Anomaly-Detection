# Financial Anomaly Detection

Projeto de portfólio em Python para detecção de comportamentos anômalos em transações financeiras sintéticas.

> **Dados 100% fictícios.** O projeto não utiliza dados reais de clientes ou empresas.

## Objetivo

Construir um pipeline que gere dados, crie features comportamentais, treine um modelo de **Isolation Forest**, produza um **Risk Score de 0 a 100** e disponibilize o resultado por uma API FastAPI.

## Stack

Python · NumPy · Pandas · Scikit-learn · Isolation Forest · FastAPI · Pydantic · Pytest · Joblib

## Pipeline

```text
Dados sintéticos → Feature Engineering → Isolation Forest → Anomaly Score → Risk Score → API
```

## Features

- valor da transação
- relação do valor com a mediana do cliente
- Z-score do valor
- divergência de cidade
- transação na madrugada
- volume de transações em 7 dias
- renda estimada

## Risk Score

| Score | Nível |
|---:|---|
| 0–30 | LOW |
| 31–60 | ATTENTION |
| 61–80 | SUSPICIOUS |
| 81–100 | HIGH |

O score é demonstrativo e não representa decisão real de fraude ou crédito.

## Execução

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python main.py
pytest -q
uvicorn src.api:app --reload
```

API: `http://127.0.0.1:8000/docs`

## Endpoint

`POST /predict`

```json
{
  "transaction_id": 999999,
  "customer_id": 42,
  "timestamp": "2026-06-15T02:30:00",
  "amount": 4500,
  "category": "Electronics",
  "city": "Sao_Paulo",
  "usual_city": "Campinas",
  "payment_method": "Credit Card",
  "customer_income": 4500
}
```

## Próximas evoluções

- comparar Isolation Forest, LOF e One-Class SVM
- SHAP para explicabilidade
- monitoramento de drift
- Docker
- pipeline de treinamento automatizado
