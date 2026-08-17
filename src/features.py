from __future__ import annotations
import numpy as np
import pandas as pd

FEATURE_COLUMNS=['amount','amount_vs_median','amount_zscore','city_mismatch','night_transaction','rolling_7d_count','customer_income']

def add_features(df):
    data=df.copy(); data['timestamp']=pd.to_datetime(data['timestamp'])
    data['hour']=data['timestamp'].dt.hour; data['day_of_week']=data['timestamp'].dt.dayofweek
    data['is_weekend']=data['day_of_week'].isin([5,6]).astype(int)
    g=data.groupby('customer_id')
    data['customer_mean_amount']=g['amount'].transform('mean'); data['customer_median_amount']=g['amount'].transform('median')
    data['customer_std_amount']=g['amount'].transform('std').fillna(0)
    data['amount_vs_median']=(data['amount']/data['customer_median_amount'].replace(0,np.nan)).fillna(1)
    data['amount_zscore']=((data['amount']-data['customer_mean_amount'])/data['customer_std_amount'].replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).fillna(0)
    data['city_mismatch']=(data['city']!=data['usual_city']).astype(int)
    data['night_transaction']=data['hour'].isin([0,1,2,3,4,5]).astype(int)
    ordered=data.sort_values(['customer_id','timestamp'])
    rolling=ordered.set_index('timestamp').groupby('customer_id')['transaction_id'].rolling('7D').count()
    rolling=rolling.reset_index().set_index(['customer_id','timestamp'])['transaction_id']
    keys=pd.MultiIndex.from_frame(ordered[['customer_id','timestamp']])
    ordered['rolling_7d_count']=rolling.reindex(keys).to_numpy()
    data=ordered.sort_index()
    data[FEATURE_COLUMNS]=data[FEATURE_COLUMNS].replace([np.inf,-np.inf],np.nan).fillna(0)
    return data
