from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import EvalCallback

from tuliptrader.crypto.environments import CryptoTradingEnv, CoinEnvSetup
from tuliptrader.crypto.io import read_kraken_history
from tuliptrader.crypto.preprocessing import CryptoHistoryTradingProcessor

if __name__ == "__main__":
    TECH_INDICATORS = ['close_10_sma', 'close_20_sma', 'close_30_sma', 'macd', 'macds', 'rsi_6', 'rsi_12', 'rsi_24']

    data = read_kraken_history(
        '/mnt/ShareDrive/Development/Privat/TulipArena/singles/TulipTrader/local_input/XBTEUR_1440.csv'
    )

    engineer = CryptoHistoryTradingProcessor(price_data=data, tech_indicators=TECH_INDICATORS, fall_quantile=0.33,
                                             rise_quantile=0.66)
    engineer.preprocess_data()

    train, val, trade = engineer.make_subsets(0.8, 0.2)

    env_setup = CoinEnvSetup(
        initial_balance=1000,
        commission=0.02,
        technical_indicators=TECH_INDICATORS,
        add_features=['fall', 'neutral', 'rise'],
        state_space=len(engineer.prep_data.columns) + 1,
        annual_inflation=0.02,
        frequency=365
    )
    env_train = env_setup.create_env(data=train, env_class=CryptoTradingEnv)

    eval_env = env_setup.create_env(data=val, env_class=CryptoTradingEnv, sense='val')
    eval_callback = EvalCallback(
        eval_env,
        # log_path='local_output/logs/val',
        best_model_save_path='local_output/models/',
        n_eval_episodes=10,
        eval_freq=len(train),
        deterministic=False,
        render=True,
        verbose=1
    )

    model = A2C(
        'MlpPolicy',
        env_train,
        n_steps=50,
        ent_coef=0.005,
        learning_rate=0.0001,
        verbose=0,
        tensorboard_log='local_output/logs/train',
    )
    n_epochs = 8
    model.learn(
        total_timesteps=len(train)*n_epochs,
        log_interval=1,
        callback=eval_callback
    )

