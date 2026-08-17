import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def evaluate(df):
    y=df.synthetic_anomaly.astype(int); p=df.anomaly.astype(int)
    return {'precision':precision_score(y,p,zero_division=0),'recall':recall_score(y,p,zero_division=0),'f1':f1_score(y,p,zero_division=0),'roc_auc':roc_auc_score(y,df.risk_score)}

def confusion(df):
    m=confusion_matrix(df.synthetic_anomaly.astype(int),df.anomaly.astype(int))
    return pd.DataFrame(m,index=['Actual Normal','Actual Anomaly'],columns=['Predicted Normal','Predicted Anomaly'])
