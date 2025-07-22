# ✅ PPO Training Loop Template with Auto-Save + Optuna Tune Ready + Pruner + Loop + Performance Log
import os
import pandas as pd
import time
import numpy as np
import pickle
import json
import optuna
from optuna.pruners import MedianPruner
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
# from trading_env_v3_ultimate import TradingEnvV3Ultimate
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ✅ CONFIG
CSV_DATA = "xau_standardized.csv"
MODEL_BASE_NAME = "ppo_xauusd_model_"
REPLAY_BASE_NAME = "replay_buffer_"
START_ROUND = 0
END_ROUND = 500000
STEP_PER_ROUND = 100000
N_ENVS = 4
WINDOW_SIZE = 8000
N_TRIALS = 5
N_ROUNDS = 10  # ✅ จำนวนรอบเทรนวนซ้ำ
PERF_LOG_PATH = "performance_log_pro.csv"




class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def store(self, obs, action, reward):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((obs, action, reward))

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.buffer, f)

    def load(self, path):
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.buffer = pickle.load(f)

class TradingEnvV3Ultimate(Env):
    def __init__(self, df, initial_balance=10000, reward_factor=1.0, max_drawdown=0.2,
                 auto_tp=80, auto_sl=-50, log_path="trade_log.json", round_label="", replay_path="replay_buffer.pkl"):
        self.df = self._add_feature_columns(df)
        self.n_steps = len(self.df)
        self.initial_balance = initial_balance
        self.reward_factor = reward_factor
        self.max_drawdown_ratio = max_drawdown
        self.auto_tp = auto_tp
        self.auto_sl = auto_sl
        self.round_label = round_label
        if isinstance(log_path, str) and round_label:
            self.log_path = log_path.replace(".json", f"_{round_label}.json")
        else:
            self.log_path = log_path
        self.log_path = log_path.replace(".json", f"_{round_label}.json") if round_label else log_path
        self.replay_path = replay_path.replace(".pkl", f"_{round_label}.pkl") if round_label else replay_path
        self.replay_buffer = ReplayBuffer()
        self.replay_buffer.load(self.replay_path)

        self.action_space = Discrete(3)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32)
        self.reset()

    def _add_feature_columns(self, df):
        if 'time' not in df.columns and 'timestamp' in df.columns:
            df['time'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df.dropna(subset=['time'], inplace=True)
            df.reset_index(drop=True, inplace=True)

        if 'high' not in df.columns or 'low' not in df.columns:
            raise ValueError("❌ Training Data ขาดคอลัมน์จำเป็น: ['high', 'low']")

        df['zone_hit'] = ((df['close'] > df['ema_20']) & (df['close'] < df['ema_50'])).astype(int)

        df['confidence_score_advanced'] = (
          (df['trend_score'] > 0).astype(int) * 0.2 +                  # แค่เป็นบวกก็พอ
          (df['multi_tf_align'] == 1).astype(int) * 0.2 +
          (df['adx_14'] > 20).astype(int) * 0.2 +                       # ลดจาก 25 → 20
          (df['rsi_14'].between(45, 70)).astype(int) * 0.1 +            # ขยายขอบเขต RSI
          (df['trend_quality'] > 0.2).astype(int) * 0.3                 # ลดจาก 0.3 → 0.2
          )

        avg_vol = df['volume'].rolling(20).mean()
        df['vol_zone_alignment'] = ((df['volume'] > avg_vol) & (df['trend_score'] == 1)).astype(int)

        df['risk_to_reward_hint'] = df['adx_14'] * df['trend_quality']
        df['is_w_pattern_entry'] = 0  # 🔹 default = 0
        df['is_m_pattern_entry'] = 0
        df = self._detect_w_pattern_entry(df)  # ✅ เรียกฟังก์ชันใหม่
        df = self._detect_m_pattern_entries(df)

        df['trend_score_H1'] = df['trend_score'].rolling(window=10, min_periods=1).mean().shift(1)
        df['trend_quality_H1'] = df['trend_quality'].rolling(window=10, min_periods=1).mean().shift(1)

        df['volume_zone_match'] = ((df['volume'] > df['volume'].rolling(20).mean()) & (df['zone_hit'] == 1)).astype(int)

        df['momentum'] = df['close'].diff().rolling(window=3).mean().fillna(0)

        df['liquidity_zone'] = (df['high'].rolling(20).max() + df['low'].rolling(20).min()) / 2
        df['liquidity_zone_match'] = (abs(df['close'] - df['liquidity_zone']) / (df['close'] + 1e-6) < 0.01).astype(int)

        df['is_new_h1_bar'] = df['time'].dt.minute.apply(lambda m: 1 if m == 0 else 0)
        df['structure_buy'] = ((df['trend_score'] > 20) & (df['trend_quality'] > 0.3)).astype(int)
        df['structure_sell'] = ((df['trend_score'] < -20) & (df['trend_quality'] > 0.3)).astype(int)
        
        df.fillna(0, inplace=True)
        return df

    # ✅ เพิ่มใน _add_feature_columns()
    def _detect_w_pattern_entry(self, df):
        df['is_w_pattern_entry'] = 0
        for i in range(30, len(df)):
            window = df.iloc[i-30:i].copy()
            closes = window['close'].values
            times = window['time'].values

            A_idx = np.argmin(closes[:10])
            A_price = closes[A_idx]
            B_idx = A_idx + np.argmax(closes[A_idx+1:A_idx+11])
            B_price = closes[B_idx]

            if B_price <= A_price:
                continue  # B ต้องสูงกว่า A

            C_idx = B_idx + np.argmin(closes[B_idx+1:B_idx+11])
            C_price = closes[C_idx]

            if C_price < A_price * 0.995:
                continue  # ห้ามต่ำกว่า A มาก

            D_idx = C_idx + np.argmin(closes[C_idx+1:C_idx+11])
            D_price = closes[D_idx]

            E_idx = D_idx + np.argmax(closes[D_idx+1:D_idx+11])
            E_price = closes[E_idx]

            if E_price <= B_price:
                continue  # ต้องหลุดคอ B

            confirm_idx = E_idx + 1
            if confirm_idx >= len(window):
                continue

            confirm_close = closes[confirm_idx]
            confirm_open = window['open'].values[confirm_idx]

            if confirm_close > confirm_open:  # แท่งยืนยัน
                real_idx = window.index[confirm_idx]
                df.at[real_idx, 'is_w_pattern_entry'] = 1
        return df
    
    def _detect_m_pattern_entries(self, df):
        for i in range(30, len(df)):
            window = df.iloc[i-30:i].copy()
            closes = window['close'].values
            A_idx = np.argmax(closes[:10])
            A_price = closes[A_idx]
            B_idx = A_idx + np.argmin(closes[A_idx+1:A_idx+11])
            B_price = closes[B_idx]
            if B_price >= A_price:
                continue
            C_idx = B_idx + np.argmax(closes[B_idx+1:B_idx+11])
            C_price = closes[C_idx]
            if C_price > A_price * 1.005:
                continue
            D_idx = C_idx + np.argmax(closes[C_idx+1:C_idx+11])
            E_idx = D_idx + np.argmin(closes[D_idx+1:D_idx+11])
            E_price = closes[E_idx]
            if E_price >= B_price:
                continue
            confirm_idx = E_idx + 1
            if confirm_idx >= len(window):
                continue
            if closes[confirm_idx] < window['open'].values[confirm_idx]:
                real_idx = window.index[confirm_idx]
                df.at[real_idx, 'is_m_pattern_entry'] = 1
        return df


    def debug_plot(self, start=0, end=300):
        df = self.df.iloc[start:end].copy()
        plt.figure(figsize=(14, 6))
        plt.plot(df['close'].values, label='Close', color='black')
        plt.plot(df['ema_20'].values, label='EMA 20', linestyle='--')
        plt.plot(df['ema_50'].values, label='EMA 50', linestyle='--')

        pattern_idx = df[df['pattern_score'] > 0].index
        plt.scatter(pattern_idx - start, df.loc[pattern_idx, 'close'], color='red', label='Pattern Score > 0')

        if hasattr(self, 'trade_log') and len(self.trade_log) > 0:
            for trade in self.trade_log:
                if start <= trade['step'] <= end:
                    plt.axvline(trade['step'] - start, color='green', alpha=0.3, linestyle='--')

        plt.title(f"Debug View: Trades & Pattern Score [{start}-{end}]")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        return np.array([
            row['close'],
            row['volume'],
            row['ema_20'],
            row['ema_50'],
            row['adx_14'],
            row['rsi_14'],
            row['trend_score'],
            row['trend_score_H1'],
            row['trend_quality'],
            row['trend_quality_H1'],
            row['multi_tf_align'],
            row['is_new_h1_bar'],
            row['zone_hit'],
            row['confidence_score_advanced'],
            row['vol_zone_alignment'],
            row['risk_to_reward_hint'],
            row['momentum'],
            row['volume_zone_match'],
            row['liquidity_zone'],
            row['liquidity_zone_match'],
            row['structure_buy'],
            row['structure_sell']
        ], dtype=np.float32)

    def step(self, action):
        obs = self._get_obs()
        row = self.df.iloc[self.current_step]
        reward = compute_reward(row, action)
        terminated = False
        truncated = False
        info = {}

        self.replay_buffer.store(obs, action, reward)
        if self.current_step % 100 == 0:
          self.replay_buffer.save(self.replay_path)

        self.trade_log.append({
            "step": int(self.current_step),
            "action": int(action),
            "reward": float(reward),
            "close": float(row['close']),
            "trend_score": int(row['trend_score']),
            "confidence": float(row['confidence_score_advanced']),
            "momentum": float(row['momentum']),
            "structure_buy": int(row['structure_buy']),
            "structure_sell": int(row['structure_sell'])
        })

        self.current_step += 1
        if self.current_step >= self.n_steps - 1:
            terminated = True
            self._save_trade_log()

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.trade_log = []
        self.current_step = np.random.randint(100, len(self.df) - 1000)

        # ✅ เคลียร์ log เก่าก่อนเทรนใหม่
        if os.path.exists(self.log_path):
            os.remove(self.log_path)
        csv_path = self.log_path.replace(".json", ".csv")
        if os.path.exists(csv_path):
            os.remove(csv_path)

        # ✅ เคลียร์ replay buffer
        self.replay_buffer = ReplayBuffer()
        if os.path.exists(self.replay_path):
            os.remove(self.replay_path)

        return self._get_obs(), {}

    def _save_trade_log(self):
        if self.log_path:
            with open(self.log_path, "a") as f:
                for log in self.trade_log:
                    json.dump(log, f)
                    f.write("\n")

        # ✅ กำหนดชื่อ csv ที่ไม่มีรอบซ้ำซ้อน
        csv_path = f"trade_log_{self.round_label}.csv"

        # ✅ อ่านไฟล์เดิมอย่างปลอดภัย
        try:
            df_old = pd.read_csv(csv_path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, FileNotFoundError):
            # print(f"⚠️ CSV log เสียหรือว่าง หรือยังไม่มี → จะเริ่มใหม่ที่ {csv_path}")
            df_old = pd.DataFrame()

        df_log = pd.DataFrame(self.trade_log)
        df_all = pd.concat([df_old, df_log], ignore_index=True)
        df_all.to_csv(csv_path, index=False)

    def save_replay_buffer(self, replay_path=None):
        try:
            if replay_path is None:
                replay_path = self.replay_path.replace(".pkl", f"_{self.round_label}.pkl")
            self.replay_buffer.save(replay_path)
        except Exception as e:
            print(f"❌ Error saving replay buffer: {e}")

    def save_trade_log(self):
        try:
            self._save_trade_log()
        except Exception as e:
            print(f"❌ Error saving trade log: {e}")

def compute_reward(row, action):
    reward = 0
    trend_score = row['trend_score']
    trend_quality = row['trend_quality']

    trend_dir = "UP" if trend_score > 20 and trend_quality > 0.3 else \
                "DOWN" if trend_score < -20 and trend_quality > 0.3 else "SIDEWAY"

    if action == 1 and row.get("is_w_pattern_entry", 0) == 1:
        reward += 0.2
    elif action == 1 and trend_dir == "UP":
        reward += 0.2
    elif action == 2 and trend_dir == "DOWN" or row.get("is_w_pattern_entry", 0) == 1:
        reward += 0.2
    elif action == 0:
        reward += 0.01
    else:
        reward -= 0.05

    reward += row['confidence_score_advanced'] * 0.5
    reward += row['zone_hit'] * 0.1
    reward += row['risk_to_reward_hint'] * 0.001
    reward += row['momentum'] * 0.01

    if row['confidence_score_advanced'] > 0.6 and reward > 0:
        reward += 0.01
    elif row['confidence_score_advanced'] < 0.1 and action != 0:
        reward -= 0.005

    dynamic_threshold = 0.3

    # ✅ เพิ่ม reward หากเข้าไม้ถูกฝั่งตาม structure
    if action == 1 and row.get("structure_buy", 0) == 1:
        reward += 0.1
    if action == 2 and row.get("structure_sell", 0) == 1:
        reward += 0.1

    return round(float(reward), 4)

# ✅ Load Data
df = pd.read_csv(CSV_DATA)
if 'timestamp' in df.columns:
    df['time'] = pd.to_datetime(df['timestamp'], errors='coerce')
df.dropna(subset=['time'], inplace=True)
df.reset_index(drop=True, inplace=True)
for col in ['high', 'low']:
    if col not in df.columns:
        df[col] = df['close']

# ✅ Init performance log
performance_log = []

# ✅ Optuna Objective Function
def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3)
    gamma = trial.suggest_float("gamma", 0.90, 0.999)
    ent_coef = trial.suggest_float("ent_coef", 1e-5, 0.01)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.98)
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048])

    np.random.seed(trial.number)
    start_idx = np.random.randint(0, len(df) - WINDOW_SIZE)
    df_train = df.iloc[start_idx:start_idx + WINDOW_SIZE].copy()

    def make_env():
        def _init():
            replay_path = f"replay_buffer_trial_{trial.number}.pkl"
            env = TradingEnvV3Ultimate(df=df_train, round_label=f"optuna_trial_{trial.number}", replay_path=replay_path)
            env.reset(seed=int(time.time()))
            return env
        return _init

    env = DummyVecEnv([make_env()])

    model = PPO("MlpPolicy", env, verbose=0,
                learning_rate=learning_rate,
                gamma=gamma,
                ent_coef=ent_coef,
                n_steps=n_steps,
                batch_size=batch_size,
                clip_range=clip_range,
                gae_lambda=gae_lambda,
                device="cpu")

    eval_callback = EvalCallback(env, eval_freq=10000, n_eval_episodes=5, deterministic=True, verbose=0)
    model.learn(total_timesteps=15000, callback=eval_callback)

    trial.report(eval_callback.last_mean_reward, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return eval_callback.last_mean_reward

# ✅ Run Loop
import multiprocessing
if __name__ == '__main__':
    
    multiprocessing.freeze_support()
    current_round = START_ROUND

    for round_i in range(N_ROUNDS):
        current_round += STEP_PER_ROUND
        round_label = str(current_round)
        print(f"\n🔁 Round {round_label} | Optuna tuning...")

        # ✅ Optuna Study with Pruner
        pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=0)
        study = optuna.create_study(direction="maximize", pruner=pruner)
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_ENVS)

        print("\n✅ Best Trial:")
        print(study.best_trial)

        best_params = study.best_params
        best_reward = study.best_value

        # ✅ Final Training with Best Params
        start_idx = np.random.randint(0, len(df) - WINDOW_SIZE)
        df_train = df.iloc[start_idx:start_idx + WINDOW_SIZE].copy()

        def make_env_final(label, seed_offset=0):
            def _init():
                env = TradingEnvV3Ultimate(
                    df=df_train,
                    round_label=label,
                    log_path="trade_log.json",
                    replay_path="replay_buffer.pkl"
                )
                env.reset(seed=int(time.time()) + seed_offset)
                return env
            return _init

        env = SubprocVecEnv([make_env_final(round_label, seed_offset=i) for i in range(N_ENVS)])

        model = PPO("MlpPolicy", env, verbose=0, device="cpu", **best_params)
        model.learn(total_timesteps=STEP_PER_ROUND, reset_num_timesteps=False, progress_bar=True)
        model.save(MODEL_BASE_NAME + round_label + ".zip")
        print(f"✅ Final Model saved → {MODEL_BASE_NAME + round_label}.zip")

        try:
            replay_path = REPLAY_BASE_NAME + round_label + ".pkl"
            env.env_method("save_replay_buffer", replay_path)
            print(f"✅ Replay saved to {replay_path}")
        except Exception as e:
            print(f"⚠️ Replay save error: {e}")

        try:
            env.env_method("save_trade_log")
            print("📄 Trade log saved")
        except Exception as e:
            print(f"⚠️ Log save error: {e}")

        # ✅ บันทึก Performance
        performance_log.append({
            "round": round_label,
            "mean_reward": best_reward,
            **best_params
        })

    # ✅ Export Performance Log
    pd.DataFrame(performance_log).to_csv(PERF_LOG_PATH, index=False)
    print(f"\n📊 Performance log saved to {PERF_LOG_PATH}")
