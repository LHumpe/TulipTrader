from stable_baselines3 import A2C

from tuliptrader.crypto.environments import CryptoTradingEnv, CoinEnvSetup
from tuliptrader.crypto.io import read_kraken_history
from tuliptrader.crypto.preprocessing import CryptoHistoryTradingProcessor

TECH_INDICATORS = ['close_10_sma', 'close_20_sma', 'close_30_sma', 'macd', 'macds', 'rsi_6', 'rsi_12', 'rsi_24']

data = read_kraken_history(
    '/mnt/ShareDrive/Development/Privat/TulipArena/singles/TulipTrader/local_input/XBTEUR_1440.csv'
)

engineer = CryptoHistoryTradingProcessor(price_data=data, tech_indicators=TECH_INDICATORS, fall_quantile=0.33,
                                         rise_quantile=0.66)
engineer.preprocess_data()

train, val, trade = engineer.make_subsets(0.8, 0.1)

env_setup = CoinEnvSetup(
    initial_balance=1000,
    commission=0.0026,
    technical_indicators=TECH_INDICATORS,
    add_features=['fall', 'neutral', 'rise'],
    state_space=len(engineer.prep_data.columns) + 1,
    annual_inflation=0.02,
    frequency=365
)
env_train = env_setup.create_env(data=trade, env_class=CryptoTradingEnv)

model = A2C.load('/mnt/ShareDrive/Development/Privat/TulipArena/singles/TulipTrader/local_output/models/best_model.zip')

obs = env_train.reset()
for _ in range(len(trade)-2):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env_train.step(action)

df = env_train.envs[0].env.history

