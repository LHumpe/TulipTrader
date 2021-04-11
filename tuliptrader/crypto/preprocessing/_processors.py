from typing import Optional, Tuple, Any, List

import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf


class CryptoHistoryTradingProcessor:
    """
    Instance providing preprocessing for price price_data and the merging of news price_data.

    Parameters
    ----------
    price_data : pd.DataFrame
        DataFrame containing price price_data.
        Columns must be ['date', 'open', 'high', 'low', 'close', 'volume', 'amount'].
    tech_indicators : List[str]
        List of names for technical indicators supported by the stockstats package.
    fall_quantile : float
        Quantile which marks the upper bound for the class "fall". E.g. when 0.25 then the lowest 25% of the
        observations regarding their value of percentage change will be considered as "fall".
    rise_quantile : float
        Quantile which marks the upper bound for the class "neutral". E.g. when 0.75  and fall quantile 0.25
        observations which fall into the range of 0.25% and 0.75% quantiles regarding their percentage change will be
        labeled as "neutral".
    """

    def __init__(self, price_data: pd.DataFrame, tech_indicators: List[str], fall_quantile: float,
                 rise_quantile: float):
        self.price_data = price_data

        self.tech_indicators = tech_indicators

        self.fall_quantile = fall_quantile
        self.rise_quantile = rise_quantile

        self._prep_data = self.price_data.copy()

    def preprocess_data(self):
        """Create Technical indicators and create the labels."""

        self._add_tech_indicators()

        self._create_label()

        return self

    def _create_label(self):
        """Create the label according to the quantiles specified and shift."""

        percentage_change = (self._prep_data['close'] - self._prep_data['open']) / self._prep_data['open']

        q1 = np.quantile(percentage_change, self.fall_quantile)
        q2 = np.quantile(percentage_change, self.rise_quantile)

        self._prep_data.loc[percentage_change <= q1, 'fall'] = 1
        self._prep_data.loc[percentage_change > q1, 'fall'] = 0
        self._prep_data['fall'] = self._prep_data['fall'].shift(-1)

        self._prep_data.loc[(percentage_change > q1) & (percentage_change <= q2), 'neutral'] = 1
        self._prep_data.loc[(percentage_change <= q1) | (percentage_change > q2), 'neutral'] = 0
        self._prep_data['neutral'] = self._prep_data['neutral'].shift(-1)

        self._prep_data.loc[percentage_change > q2, 'rise'] = 1
        self._prep_data.loc[percentage_change <= q2, 'rise'] = 0
        self._prep_data['rise'] = self._prep_data['rise'].shift(-1)

        return self

    def _add_tech_indicators(self):
        """Calculate and add technical indicators per currency pair."""
        temp_data = Sdf.retype(self._prep_data.copy())

        for indicator in self.tech_indicators:
            indicator_df = pd.DataFrame(temp_data[indicator])
            indicator_df.index = self._prep_data.index
            self._prep_data[indicator] = indicator_df

        self._prep_data.dropna(inplace=True)

        return self

    def slice_data(self, start: str, end: Optional[str] = None):
        """
        Slice data according to date. Creates a subset only containing observations from the specified time frame.

        Parameters
        ----------
        start : str
            String representing the first observation that should be used. Must be in the same DateTime format as
            the date column.
        end : str
            String representing the last observation that should be used. Must be in the same DateTime format as
            the date column.
        """
        if end:
            temp_data = self._prep_data[(self._prep_data.date >= start) & (self._prep_data.date < end)]
        else:
            temp_data = self._prep_data[self._prep_data.date >= start]

        self._prep_data = temp_data.sort_values(['date', 'tic'], ignore_index=True)
        self._prep_data.index = self._prep_data.date.factorize()[0]

        return self

    def make_subsets(self, train_size: float, val_size: float) -> Tuple[Any, Any, Any]:
        """
        Split the data into train, validation and trade sets and merge the textual data. The train and validation set
        will be sized in regards to the specified fractions. The trade set will contain the remainder of the data.

        Parameters
        ----------
        train_size : float
            Fraction of the data that should be used for the training set.
        val_size : float
            Fraction of the data that should be used as the validation set.

        Returns
        -------
        train, validation, trade : pd.DataFrame
            Three DataFrames containing the corresponding subsets.
        """

        temp_data = self._prep_data.copy()

        data_len = len(temp_data)

        # Split the data
        train = temp_data.iloc[0:int(train_size * data_len)]
        train.index = train.date.factorize()[0]

        validation = temp_data.iloc[int(train_size * data_len):int(data_len * (train_size + val_size))]
        validation.index = validation.date.factorize()[0]

        trade = temp_data.iloc[int(data_len * (train_size + val_size)):]
        trade.index = trade.date.factorize()[0]

        return train.dropna(), validation.dropna(), trade.dropna()

    @property
    def prep_data(self) -> pd.DataFrame:
        return self._prep_data.dropna()
