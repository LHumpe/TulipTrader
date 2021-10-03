import warnings
from typing import List, Optional

import gym
import numpy as np
import pandas as pd
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)


class CryptoTradingEnv(gym.Env):

    def __init__(self, data: pd.DataFrame, technical_indicators: List[str], add_features: List[str], state_space: int,
                 commission: float, initial_balance: int, inflation_rate: float):
        # data information
        self.full_data: pd.DataFrame = data
        self.technical_indicators: List[str] = technical_indicators
        self.add_features: List[str] = add_features

        # environment config
        self.state_space: int = state_space
        self.action_space: gym.spaces.MultiDiscrete = gym.spaces.MultiDiscrete([3, 10])
        self.observation_space: gym.spaces.Box = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space,),
                                                                dtype=np.float64)
        self.terminal: bool = False

        # state information
        self.current_step: int = 0
        self.step_data: Optional[pd.DataFrame] = None
        self.history: Optional[pd.DataFrame] = None
        self.state: Optional[list] = None

        # trading info
        self.commission: float = commission
        self.initial_balance: int = initial_balance
        self.inflation_rate: float = inflation_rate

    def reset(self):
        self.current_step = 0
        self.step_data = self.full_data.iloc[self.current_step, :].copy()
        self.history = self._reset_history()
        self.state = self._initialize_state()
        return self.state

    def _reset_history(self, step: bool = False) -> pd.DataFrame:
        if step:
            return pd.DataFrame({
                'Date': [self.step_data.date],
                'Actions': [np.nan],
                'Balance_Start': [self.history.Balance_End.iloc[-1]],
                'Balance_End': [np.nan],
                'NW_Start': [self.history.NW_End.iloc[-1]],
                'NW_End': [np.nan],
                'Coins_Start': [self.history.Coins_End.iloc[-1]],
                'Coins_End': [np.nan],
                'Total': [np.nan],
                'Reward': [np.nan],
            })
        else:
            return pd.DataFrame({
                'Date': [self.step_data.date],
                'Actions': [np.nan],
                'Balance_Start': [self.initial_balance],
                'Balance_End': [np.nan],
                'NW_Start': [self.initial_balance],
                'NW_End': [np.nan],
                'Coins_Start': [0],
                'Coins_End': [np.nan],
                'Total': [np.nan],
                'Reward': [np.nan],
            })

    def _initialize_state(self) -> list:
        state = [self.history.Balance_Start.iloc[-1]] + \
                [self.history.Coins_Start.iloc[-1]] + \
                sum([[self.step_data[tech]] for tech in self.technical_indicators], []) + \
                sum([[self.step_data[user]] for user in self.add_features], []) + \
                [self.step_data.open] + \
                [self.step_data.high] + \
                [self.step_data.low] + \
                [self.step_data.volume] + \
                [self.step_data.close] + \
                [self.step_data.amount]
        return state

    def step(self, action: list):
        # TODO: add penalty for unrealized profit
        action_type, amount = action[0], action[1] / 10
        real_cost_per_coin = self.step_data.close * (1 + self.commission)
        real_value_per_coin = self.step_data.close * (1 - self.commission)
        print(self.step_data.date, action)

        if action_type == 0:  # buy
            self.history.Total.iloc[-1] = -(self.history.Balance_Start.iloc[-1] * amount)
            self.history.Actions.iloc[-1] = abs(self.history.Total.iloc[-1]) / real_cost_per_coin
        elif action_type == 1:  # sell
            self.history.Actions.iloc[-1] = -(self.history.Coins_Start.iloc[-1] * amount)
            self.history.Total.iloc[-1] = abs(self.history.Actions.iloc[-1]) * real_value_per_coin
        else:  # hold
            self.history.Actions.iloc[-1] = 0
            self.history.Total.iloc[-1] = 0

        self.history.Coins_End.iloc[-1] = self.history.Coins_Start.iloc[-1] + self.history.Actions.iloc[-1]

        self.history.Balance_End.iloc[-1] = (self.history.Balance_Start.iloc[-1] + self.history.Total.iloc[-1]) * \
                                            (1 - self.inflation_rate)

        self.history.NW_End.iloc[-1] = self.history.Balance_End.iloc[-1] + \
                                       (self.history.Coins_End.iloc[-1] * self.step_data.close)

        self.history.Reward.iloc[-1] = self.history.NW_End.iloc[-1] - self.history.NW_Start.iloc[-1]

        self.current_step += 1

        self.terminal = self.history.NW_End.iloc[-1] <= 0 or self.current_step >= len(self.full_data.index.unique()) - 2

        if not self.terminal:
            self.step_data = self.full_data.loc[self.current_step, :]
            new_step = self._reset_history(step=True)
            self.history = pd.concat([self.history, new_step], axis=0).reset_index(drop=True)
            self.state = self._initialize_state()

        return self.state, self.history.Reward.iloc[-2], self.terminal, {'NW': self.history.values.tolist()}

    def render(self, mode='human'):
        return self.state
