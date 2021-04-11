import pandas as pd


def read_kraken_history(path: str) -> pd.DataFrame:
    """
    Import and format historical price price_data from the kraken exchange.

    Parameters
    ----------
    path : str
        Path to the historic price price_data

    Returns
    -------
    df : pd.DataFrame
        DataFrame that contains properly formatted historic price price_data
    """
    # import price_data
    df = pd.read_csv(path, names=['date', 'open', 'high', 'low', 'close', 'volume', 'amount'])
    df = df[['date', 'open', 'close', 'high', 'low', 'volume', 'amount']]

    # cast time to datetime
    df['date'] = pd.to_datetime(df['date'], unit='s')

    return df
