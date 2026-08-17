from pathlib import Path
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .features import FEATURE_COLUMNS

def train_model(df, contamination=0.02, random_state=42):
    pipe=Pipeline([('scaler',StandardScaler()),('model',IsolationForest(n_estimators=250,contamination=contamination,random_state=random_state,n_jobs=-1))])
    pipe.fit(df[FEATURE_COLUMNS]); return pipe

def score_transactions(model, df):
    result=df.copy(); result['model_prediction']=model.predict(result[FEATURE_COLUMNS]); result['anomaly']=result['model_prediction'].eq(-1)
    decision=model.decision_function(result[FEATURE_COLUMNS]); result['anomaly_score']=-decision
    lo,hi=result['anomaly_score'].min(),result['anomaly_score'].max()
    result['risk_score']=0.0 if hi==lo else (result['anomaly_score']-lo)/(hi-lo)*100
    result['risk_level']=__import__('pandas').cut(result['risk_score'],[-.01,30,60,80,100.01],labels=['LOW','ATTENTION','SUSPICIOUS','HIGH'])
    return result

def save_model(model,path='models/isolation_forest.joblib'):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); joblib.dump(model,p)

def load_model(path='models/isolation_forest.joblib'): return joblib.load(path)
