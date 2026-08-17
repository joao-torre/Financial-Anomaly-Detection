from src.data_generator import generate_data
from src.features import add_features
from src.model import train_model, score_transactions

def test_model_returns_scores():
    raw=generate_data(n_customers=10,transactions_per_customer=20,seed=1); data=add_features(raw); model=train_model(data); scored=score_transactions(model,data)
    assert len(scored)==200; assert scored.risk_score.between(0,100).all(); assert scored.anomaly.dtype==bool
