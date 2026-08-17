from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

CATEGORIES = ['Grocery','Fuel','Restaurant','Travel','Electronics','Online','Health']
CITIES = ['Sao_Paulo','Campinas','Sorocaba','Jundiai','Santos','Ribeirao_Preto']

def generate_data(n_customers=1000, transactions_per_customer=80, seed=42, anomaly_rate=0.02):
    rng = np.random.default_rng(seed)
    n = n_customers * transactions_per_customer
    customer_ids = np.repeat(np.arange(1, n_customers + 1), transactions_per_customer)
    base_income = rng.lognormal(np.log(4500), 0.55, n_customers)
    spend_factor = rng.lognormal(0, 0.35, n_customers)
    usual_city_idx = rng.integers(0, len(CITIES), n_customers)
    customer_income = np.repeat(base_income, transactions_per_customer)
    spend_factor = np.repeat(spend_factor, transactions_per_customer)
    usual_city = np.repeat(np.array(CITIES)[usual_city_idx], transactions_per_customer)
    timestamps = pd.Timestamp('2026-01-01') + pd.to_timedelta(rng.integers(0, 180*24*60, n), unit='m')
    categories = rng.choice(CATEGORIES, n, p=[.20,.15,.18,.08,.10,.20,.09])
    city = rng.choice(CITIES, n, p=[.45,.18,.12,.08,.10,.07])
    payment_method = rng.choice(['Credit Card','Debit Card','Pix'], n, p=[.55,.20,.25])
    bases = {'Grocery':140,'Fuel':90,'Restaurant':75,'Travel':600,'Electronics':850,'Online':180,'Health':220}
    amounts = np.array([bases[c] for c in categories]) * spend_factor * rng.lognormal(0,.45,n) + rng.normal(0,10,n)
    amounts = np.maximum(amounts, 5)
    is_anomaly = rng.random(n) < anomaly_rate
    anomaly_type = np.where(is_anomaly, rng.choice(['high_value','unusual_hour','location','combined'], n), 'none')
    high = is_anomaly & np.isin(anomaly_type, ['high_value','combined'])
    amounts[high] *= rng.uniform(6,15,high.sum())
    timestamps = pd.Series(timestamps)
    hours = timestamps.dt.hour.to_numpy()
    night = is_anomaly & np.isin(anomaly_type, ['unusual_hour','combined'])
    hours[night] = rng.choice([1,2,3,4], night.sum())
    timestamps = timestamps.dt.normalize() + pd.to_timedelta(hours, unit='h') + pd.to_timedelta(rng.integers(0,60,n), unit='m')
    loc = is_anomaly & np.isin(anomaly_type, ['location','combined'])
    current_city = city.copy(); current_city[loc] = rng.choice(CITIES, loc.sum())
    df = pd.DataFrame({'transaction_id':np.arange(1,n+1),'customer_id':customer_ids,'timestamp':timestamps,
        'amount':np.round(amounts,2),'category':categories,'city':current_city,'usual_city':usual_city,
        'payment_method':payment_method,'customer_income':np.round(customer_income,2),
        'synthetic_anomaly':is_anomaly,'anomaly_type':anomaly_type})
    return df.sort_values(['customer_id','timestamp']).reset_index(drop=True)

def save_raw_data(df, path='data/transactions.csv'):
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True); df.to_csv(p,index=False)
