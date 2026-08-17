from pathlib import Path
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from .features import add_features
from .model import load_model, score_transactions
app=FastAPI(title='Financial Anomaly Detection API',version='1.0.0')
MODEL_PATH=Path('models/isolation_forest.joblib')
class Transaction(BaseModel):
    transaction_id:int=1; customer_id:int; timestamp:str; amount:float=Field(gt=0); category:str='Online'; city:str='Sao_Paulo'; usual_city:str='Sao_Paulo'; payment_method:str='Credit Card'; customer_income:float=Field(gt=0)
@app.get('/health')
def health(): return {'status':'ok'}
@app.post('/predict')
def predict(transaction:Transaction):
    if not MODEL_PATH.exists(): return {'error':'Modelo não encontrado. Execute python main.py antes.'}
    model=load_model(str(MODEL_PATH)); row=pd.DataFrame([transaction.model_dump()]); features=add_features(row); scored=score_transactions(model,features).iloc[0]
    return {'transaction_id':int(scored.transaction_id),'risk_score':round(float(scored.risk_score),2),'risk_level':str(scored.risk_level),'anomaly':bool(scored.anomaly)}
