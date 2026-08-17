import pandas as pd
from src.features import add_features

def test_feature_creation():
    df=pd.DataFrame({'transaction_id':[1,2,3],'customer_id':[1,1,1],'timestamp':pd.to_datetime(['2026-01-01 10:00','2026-01-02 10:00','2026-01-03 02:00']),'amount':[50.,60.,5000.],'category':['Grocery']*3,'city':['Campinas','Campinas','Sao_Paulo'],'usual_city':['Campinas']*3,'payment_method':['Pix']*3,'customer_income':[5000.]*3})
    r=add_features(df); assert 'amount_zscore' in r; assert r.loc[2,'city_mismatch']==1
