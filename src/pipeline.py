from pathlib import Path
from .data_generator import generate_data, save_raw_data
from .features import add_features
from .model import train_model, score_transactions, save_model
from .evaluation import evaluate, confusion

def run_pipeline():
    print('1/5 - Gerando dados sintéticos...'); raw=generate_data(); save_raw_data(raw)
    print('2/5 - Criando features comportamentais...'); data=add_features(raw)
    print('3/5 - Treinando Isolation Forest...'); model=train_model(data); save_model(model)
    print('4/5 - Calculando scores...'); scored=score_transactions(model,data)
    out=Path('data/processed'); out.mkdir(parents=True,exist_ok=True); scored.to_csv(out/'scored_transactions.csv',index=False)
    print('5/5 - Avaliando o modelo...'); metrics=evaluate(scored)
    for k,v in metrics.items(): print(f'{k:10}: {v:.4f}')
    print('\nMatriz de confusão:'); print(confusion(scored))
    cols=['transaction_id','customer_id','amount','city','risk_score','risk_level','anomaly','synthetic_anomaly','anomaly_type']
    print('\nTop 10 por risco:'); print(scored.sort_values('risk_score',ascending=False)[cols].head(10).to_string(index=False))
