#Ce module gère le chargement initial, le filtrage par magasin et l'agrégation quotidienne.*

import pandas as pd

def clean_and_merge(train_path, trans_path, holiday_path, store_nbr=1):
    """
    Fusionne les datasets train, transactions et holidays pour un magasin donné.
    """
    # Chargement
    train = pd.read_csv(train_path, parse_dates=['date'])
    transactions = pd.read_csv(trans_path, parse_dates=['date'])
    holidays = pd.read_csv(holiday_path, parse_dates=['date'])
    
    # Filtrage par Magasin (Store 1 par défaut)
    df = train[train['store_nbr'] == store_nbr].copy()
    
    # Agrégation quotidienne
    df = df.groupby('date').agg({
        'sales': 'sum', 
        'onpromotion': 'sum'
    }).reset_index()
    
    # Fusion avec Transactions
    df = pd.merge(df, transactions[['date', 'transactions']], on='date', how='left')
    df['transactions'] = df['transactions'].fillna(0)
    
    # Fusion avec Holidays
    holidays = holidays[holidays['transferred'] == False]
    df['is_holiday'] = df['date'].isin(holidays['date']).astype(int)
    
    return df