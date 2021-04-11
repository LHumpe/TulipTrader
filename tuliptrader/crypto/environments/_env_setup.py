from typing import List

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


class CoinEnvSetup:
    def __init__(self, initial_balance: int, commission: float, technical_indicators: List[str],
                 add_features: List[str], state_space: int, annual_inflation: float, frequency: float):
        self.initial_balance = initial_balance
        self.commission = commission
        self.technical_indicators = technical_indicators
        self.user_feature_list = add_features
        self.state_space = state_space
        self.inflation_rate = annual_inflation / frequency

    def create_env(self, data, env_class, sense='train'):
        if sense == 'train':
            env = DummyVecEnv([
                lambda: Monitor(env_class(
                    data=data,
                    technical_indicators=self.technical_indicators,
                    add_features=self.user_feature_list,
                    state_space=self.state_space,
                    initial_balance=self.initial_balance,
                    commission=self.commission,
                    inflation_rate=self.inflation_rate
                ), filename='local_output/logs/monitor/train.csv', info_keywords=('NW',))
            ])
        else:
            env = DummyVecEnv([
                lambda: Monitor(env_class(
                    data=data,
                    technical_indicators=self.technical_indicators,
                    add_features=self.user_feature_list,
                    state_space=self.state_space,
                    initial_balance=self.initial_balance,
                    commission=self.commission,
                    inflation_rate=self.inflation_rate
                ), filename='local_output/logs/monitor/val.csv', info_keywords=('NW',))
            ])
        return env
