import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import random
import time
import os
import json
import talib
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces, Env
import optuna
import pickle
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import requests
import traceback
from collections import deque

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# ✅ ตั้งค่าบัญชี MT5
MT5_USER = 123434
MT5_PASSWORD = ""
MT5_SERVER = ""
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

# ✅ ตั้งค่า Symbol และ Timeframe
SYMBOL = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M15
HIGHER_TIMEFRAME = mt5.TIMEFRAME_H1  # ✅ ใช้ H1 เป็น TF เสริม
ATR_PERIOD = 14
ADX_PERIOD = 14
RISK_PERCENT = 2  
CONFIDENCE_THRESHOLD = 0.75  # ✅ เทรดเฉพาะถ้ามั่นใจเกิน 80%
FAKE_BREAKOUT_THRESHOLD = 0.1  
ORDER_FLOW_WINDOW = 20  
TRAILING_STOP_MULTIPLIER = 1.5  # ✅ ใช้ ATR x 1.5 เป็น Trailing Stop
MAX_TRADES = 4  # ✅ จำกัดจำนวนออเดอร์ที่เปิดพร้อมกัน
TRADE_LOG_NAME = "100Log.json"  #แก้ชื่อไฟล์เก็บ LOG ที่นี่
REPLAY_NAME = "replay_buffer_400000_test.pkl"  #แก้ชื่อไฟล์เก็บข้อมูลการเทรนที่นี่
STEP_LEARN = 0

# ✅ โหลด / สร้างโมเดล PPO
MODEL_PATH = "ppo_xauusd_model_400000_test.zip"  #ชื่อไฟล์เก็บข้อมูลโมเดลที่นี่

# ✅ เชื่อมต่อ MT5
if not mt5.initialize():
    print("❌ MT5 Connect Failed!")
    mt5.shutdown()
if not mt5.login(MT5_USER, password=MT5_PASSWORD, server=MT5_SERVER):
    print("❌ Login Failed! ตรวจสอบ User/Pass/Server")
else:
    print(f"✅ Login Successful! Trading {SYMBOL}")

def get_mt5_data(symbol, timeframe, count=100):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None or len(rates) < count:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    return df  # ❌ อย่า enrich ตรงนี้ — ให้เรียกทีหลัง!

def telegram_notify(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            print("⚠️ Telegram Error:", response.text)
        else:
            print("📤 Telegram Sent!")
    except Exception as e:
        print("❌ Telegram Exception:", e)

# ✅ ต้องแน่ใจว่า momentum ถูกคำนวณไว้ก่อนเรียก build_observation_vector()
def enrich_features(df):
    df = calculate_indicators(df)
    if 'momentum' not in df.columns:
        df['momentum'] = df['close'].diff(4).fillna(0)  # หรือใช้ talib.MOM(df['close'], timeperiod=4)
    return df

def detect_market_structure(df, window=10):
    swing_highs = []
    swing_lows = []
    for i in range(window, len(df) - window):
        high = df['high'].iloc[i]
        low = df['low'].iloc[i]
        past_highs = df['high'].iloc[i-window:i+1]
        past_lows = df['low'].iloc[i-window:i+1]
        if high == past_highs.max():
            swing_highs.append((i, df['time'].iloc[i], high))
        if low == past_lows.min():
            swing_lows.append((i, df['time'].iloc[i], low))
    return swing_highs, swing_lows

def get_market_structure_signal(df_1h, window=10):
    swing_highs, swing_lows = detect_market_structure(df_1h, window)
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return None
    last_hh = swing_highs[-1][2]
    prev_hh = swing_highs[-2][2]
    last_hl = swing_lows[-1][2]
    last_ll = swing_lows[-1][2]
    prev_ll = swing_lows[-2][2]
    last_lh = swing_highs[-1][2]
    if last_hh > prev_hh and last_hl > prev_ll:
        return 'buy'
    if last_ll < prev_ll and last_lh < prev_hh:
        return 'sell'
    return None

# ✅ อัปเดต build_observation_vector ให้มี structure_buy/structure_sell
def build_observation_vector(row):
    required_keys = [
        'close', 'volume', 'ema_20', 'ema_50', 'adx_14', 'rsi_14',
        'trend_score', 'trend_score_H1', 'trend_quality', 'trend_quality_H1',
        'multi_tf_align', 'is_new_h1_bar', 'zone_hit', 'confidence_score_advanced',
        'vol_zone_alignment', 'risk_to_reward_hint', 'momentum',
        'volume_zone_match', 'liquidity_zone', 'liquidity_zone_match',
        'structure_buy', 'structure_sell'
    ]
    for key in required_keys:
        if key not in row:
            raise KeyError(f"Missing field: {key}")

    # 🔍 จัดกลุ่ม vector ตามหมวด เพื่อให้อ่านง่าย + แก้ไขง่าย
    price_features = [
        row['close'], row['volume'], row['ema_20'], row['ema_50']
    ]
    trend_features = [
        row['adx_14'], row['rsi_14'], row['trend_score'], row['trend_score_H1'],
        row['trend_quality'], row['trend_quality_H1']
    ]
    alignment_features = [
        row['multi_tf_align'], row['is_new_h1_bar'], row['zone_hit'],
        row['confidence_score_advanced'], row['vol_zone_alignment'], row['risk_to_reward_hint']
    ]
    market_features = [
        row['momentum'], row['volume_zone_match'], row['liquidity_zone'],
        row['liquidity_zone_match'], row['structure_buy'], row['structure_sell']
    ]

    return np.array(
        price_features + trend_features + alignment_features + market_features,
        dtype=np.float32
    )


# ✅ ฟังก์ชันบันทึก Replay Buffer ลงไฟล์
def save_replay_buffer():
    with open(REPLAY_NAME, "wb") as f:
        pickle.dump(replay_buffer.buffer, f)
    print(f"📁 Replay Buffer ถูกบันทึกลงไฟล์ ({len(replay_buffer.buffer)} records)!")

# ✅ ฟังก์ชันโหลด Replay Buffer จากไฟล์
def load_replay_buffer():
    global replay_buffer
    try:
        with open(REPLAY_NAME, "rb") as f:
            buffer_data  = pickle.load(f)
        replay_buffer = ReplayBuffer()
        replay_buffer.buffer = buffer_data  # ✅ เก็บข้อมูลใน buffer ให้ถูก
        print(f"📂 Loaded Replay Buffer ({len(replay_buffer.buffer)} records)")
    except FileNotFoundError:
        print("⚠️ ไม่พบไฟล์ Replay Buffer, เริ่มใหม่...")
        replay_buffer = ReplayBuffer(capacity=10000)
        with open(REPLAY_NAME, "wb") as f:
            pickle.dump(replay_buffer, f)

# ✅ Confidence Tuning Parameters
CONFIDENCE_FILTER_ENABLED = True
CONFIDENCE_TUNING_RULES = [
    {"min_conf": 0.00, "max_conf": 0.30, "action": "HOLD", "reason": "ความมั่นใจต่ำเกินไป"},
    {"min_conf": 0.30, "max_conf": 0.50, "action": "CAUTIOUS"},
    {"min_conf": 0.50, "max_conf": 0.70, "action": "NORMAL"},
    {"min_conf": 0.70, "max_conf": 1.00, "action": "CONFIDENT"}
]

# ✅ ฟังก์ชันปรับระดับ Confidence ให้เข้าไม้แบบชาญฉลาด
def confidence_tuning(confidence):
    for rule in CONFIDENCE_TUNING_RULES:
        if rule["min_conf"] <= confidence < rule["max_conf"]:
            return rule.get("action", "HOLD"), rule.get("reason", "")
    return "HOLD", "ไม่เข้าเงื่อนไขใด ๆ"

def get_multi_tf_alignment():
    df_m15 = get_mt5_data(SYMBOL, mt5.TIMEFRAME_M15, 100)
    df_h1 = get_mt5_data(SYMBOL, mt5.TIMEFRAME_H1, 100)

    if df_m15 is None or df_h1 is None:
        return None, 0

    score_m15, dir_m15 = calculate_trend_score_pro(df_m15)
    score_h1, dir_h1 = calculate_trend_score_pro(df_h1)

    multi_tf_align = int(dir_m15 == dir_h1)

    return df_m15, multi_tf_align

def calculate_indicators(df, multi_tf_align=1):
    if df is None or len(df) < 50:
        return df

    df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
    df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
    df['adx_14'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
    df['trend_quality'] = df['adx_14'] / 50
    
    df['trend_score'] = (df['ema_20'] - df['ema_50']) / df['close'] * 100
    avg_vol = df['volume'].rolling(20).mean()
    df['zone_hit'] = ((df['close'] > df['ema_20']) & (df['close'] < df['ema_50'])).astype(int)
    df['volume_zone_match'] = ((df['volume'] > avg_vol) & (df['zone_hit'] == 1)).astype(int)

    df["liquidity_zone"] = (df["high"].rolling(20).max() + df["low"].rolling(20).min()) / 2

    structure_signal = get_market_structure_signal(get_mt5_data(SYMBOL, HIGHER_TIMEFRAME, 100))
    df['structure_buy'] = 1 if structure_signal == "buy" else 0
    df['structure_sell'] = 1 if structure_signal == "sell" else 0
    
    pattern_detector = PatternDetector()
    df = pattern_detector._detect_w_pattern_entry(df)
    df = pattern_detector._detect_m_pattern_entries(df)

    trend_score, trend_dir = calculate_trend_score_pro(df)
    df.loc[df.index[-1], 'trend_score'] = trend_score
    df.loc[df.index[-1], 'trend_direction'] = trend_dir

    df['multi_tf_align'] = multi_tf_align

    df['confidence_score_advanced'] = (
        np.clip(abs(df['trend_score']) / 100.0, 0, 1) * 0.2 +      # ✅ เทรนด์แรง → ได้แต้ม
        (df['adx_14'] > 20).astype(int) * 0.2 +                    # ✅ เทรนด์ชัดเจน
        (df['rsi_14'].between(45, 65)).astype(int) * 0.1 +         # ✅ RSI กลาง (ไม่ overbought/oversold)
        (df['trend_quality'] > 0.25).astype(int) * 0.1 +           # ✅ เทรนด์มีคุณภาพ
        df['volume_zone_match'] * 0.1 +                            # ✅ อยู่ในโซน Volume สำคัญ
        df['structure_buy'] * 0.15 +                               # ✅ เจอโครงสร้าง Buy → เพิ่มแต้ม
        df['structure_sell'] * 0.15 +                              # ✅ เจอโครงสร้าง Sell → เพิ่มแต้ม
        df['is_w_pattern_entry'] * 0.1 +                           # ✅ เจอ W → เพิ่มแต้ม
        df['is_m_pattern_entry'] * 0.1                             # ✅ เจอ M → เพิ่มแต้ม
    )
    df['risk_to_reward_hint'] = df['adx_14'] * df['trend_quality']
    df['is_new_h1_bar'] = df['time'].dt.minute.apply(lambda m: 1 if m == 0 else 0)
    df['momentum'] = df['close'].diff(4).fillna(0)
    df['liquidity_zone_match'] = (abs(df['close'] - df['liquidity_zone']) / (df['close'] + 1e-6) < 0.01).astype(int)
    df['trend_score_H1'] = df['trend_score'].rolling(10).mean().shift(1)
    df['trend_quality_H1'] = df['trend_quality'].rolling(3).mean().shift(1)
    df['vol_zone_alignment'] = ((df['volume'] > df['volume'].rolling(20).mean()) & (df['trend_score'] == 1)).astype(int)
    pattern_detector = PatternDetector()
    df = pattern_detector._detect_w_pattern_entry(df)
    df = pattern_detector._detect_m_pattern_entries(df)
    df.fillna(0, inplace=True)
    return df

def calculate_trend_score_pro(df):
    """
    ✅ วิเคราะห์แนวโน้มแบบโปร: รวม EMA, ADX, Order Flow, Volume, Liquidity Zone (ไม่มี pattern แล้ว)
    ส่งออก: trend_score (-100 ถึง +100), trend_direction ("UP"/"DOWN"/"SIDEWAY")
    """
    if df is None or len(df) < 50:
        return 0, "SIDEWAY"

    ema_20 = talib.EMA(df['close'], timeperiod=20)
    ema_50 = talib.EMA(df['close'], timeperiod=50)
    ema_200 = talib.EMA(df['close'], timeperiod=200)
    adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)

    df['Buy_Vol'] = df['close'] - df['low']
    df['Sell_Vol'] = df['high'] - df['close']
    df['Order_Flow'] = df['Buy_Vol'] - df['Sell_Vol']
    df['Liquidity Zone'] = (df['high'].rolling(20).max() + df['low'].rolling(20).min()) / 2

    ema_20_val = ema_20.iloc[-1]
    ema_50_val = ema_50.iloc[-1]
    ema_200_val = ema_200.iloc[-1]
    adx_val = adx.iloc[-1]
    of_val = df['Order_Flow'].rolling(5).mean().iloc[-1]
    vol_val = df['volume'].iloc[-1]
    avg_vol_val = df['volume'].rolling(20).mean().iloc[-1]
    lz_val = df['Liquidity Zone'].iloc[-1]
    price = df['close'].iloc[-1]

    score = 0
    if ema_20_val > ema_50_val: score += 20
    if ema_50_val > ema_200_val: score += 20
    if ema_20_val < ema_50_val: score -= 20
    if ema_50_val < ema_200_val: score -= 20

    if adx_val > 20: score += 10 if score > 0 else -10
    if of_val > 0: score += 10
    elif of_val < 0: score -= 10

    if vol_val > avg_vol_val: score += 5
    else: score -= 5

    if price > lz_val: score += 5
    elif price < lz_val: score -= 5

    score = max(-100, min(score, 100))

    if score > 20:
        trend_dir = "UP"
    elif score < -20:
        trend_dir = "DOWN"
    else:
        trend_dir = "SIDEWAY"

    return score, trend_dir


# ✅ แทนที่  ด้วยระบบใหม่สำหรับ W/M
class PatternDetector:
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
                continue

            C_idx = B_idx + np.argmin(closes[B_idx+1:B_idx+11])
            C_price = closes[C_idx]

            if C_price < A_price * 0.995:
                continue

            D_idx = C_idx + np.argmin(closes[C_idx+1:C_idx+11])
            D_price = closes[D_idx]

            E_idx = D_idx + np.argmax(closes[D_idx+1:D_idx+11])
            E_price = closes[E_idx]

            if E_price <= B_price:
                continue

            confirm_idx = E_idx + 1
            if confirm_idx >= len(window):
                continue

            confirm_close = closes[confirm_idx]
            confirm_open = window['open'].values[confirm_idx]

            if confirm_close > confirm_open:
                real_idx = window.index[confirm_idx]
                df.at[real_idx, 'is_w_pattern_entry'] = 1
        return df

    def _detect_m_pattern_entries(self, df):
        df['is_m_pattern_entry'] = 0
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


# ✅ ฟังก์ชันเช็คคุณภาพแนวโน้ม (เวอร์ชันใหม่ + Pattern Matching + Retest คอ)
def check_trend_quality(df):
    if df is None or len(df) < 50:
        return {
            "trend_score": 0,
            "trend_direction": "NONE",
            "confidence_buy": 0.0,
            "confidence_sell": 0.0,
            "adx": 0,
            "fakeout_risk": 0.0,
            "liquidity_zone": 0.0,
            "order_flow": 0.0,
            "volume": 0.0
        }

    df['EMA_50'] = talib.EMA(df['close'], timeperiod=50)
    df['EMA_200'] = talib.EMA(df['close'], timeperiod=200)
    df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=ADX_PERIOD)

    ema_50, ema_200 = df['EMA_50'].iloc[-1], df['EMA_200'].iloc[-1]
    adx = df['ADX'].iloc[-1]

    order_flow = (df['close'] - df['low']) - (df['high'] - df['close'])
    df['Order_Flow'] = order_flow
    avg_of = df['Order_Flow'].rolling(5).mean().iloc[-1]

    df['Liquidity Zone'] = (df['high'].rolling(20).max() + df['low'].rolling(20).min()) / 2
    df.fillna(0, inplace=True)
    price = df['close'].iloc[-1]
    liquidity_zone = df['Liquidity Zone'].iloc[-1]

    score = 0
    if adx > 20:
        score += 30
    if ema_50 > ema_200:
        score += 25
    elif ema_50 < ema_200:
        score -= 25

    if avg_of > 0:
        score += 20
    elif avg_of < 0:
        score -= 20

    if price > liquidity_zone:
        score += 15
    elif price < liquidity_zone:
        score -= 15

    trend_score = max(-100, min(score, 100))
    trend_direction = "UP" if trend_score > 20 else "DOWN" if trend_score < -20 else "SIDEWAY"

    confidence_buy = min(1.0, max(0, (trend_score + 100) / 200)) if trend_direction == "UP" else 0.0
    confidence_sell = min(1.0, max(0, (100 - trend_score) / 200)) if trend_direction == "DOWN" else 0.0

    fakeout_risk = 1.0 - abs(avg_of) / max(abs(df['Order_Flow'].max()), 1e-5)

    return {
        "trend_score": trend_score,
        "trend_direction": trend_direction,
        "confidence_buy": round(confidence_buy, 3),
        "confidence_sell": round(confidence_sell, 3),
        "adx": round(adx, 2),
        "fakeout_risk": round(fakeout_risk, 3),
        "liquidity_zone": round(liquidity_zone, 2),
        "order_flow": round(avg_of, 2),
        "volume": float(df['volume'].iloc[-1])
    }

# ✅ วิเคราะห์ Multi-Timeframe แบบละเอียด
def analyze_multi_timeframe():
    df_m15 = get_mt5_data(SYMBOL, TIMEFRAME, 100)
    df_h1 = get_mt5_data(SYMBOL, HIGHER_TIMEFRAME, 100)

    if df_m15 is None or df_h1 is None:
        return "Sideway"

    trend_m15 = check_trend_quality(df_m15)
    trend_h1 = check_trend_quality(df_h1)

    if trend_m15["trend_direction"] == "UP" and trend_h1["trend_direction"] == "UP":
        return "Trend_UP"
    elif trend_m15["trend_direction"] == "DOWN" and trend_h1["trend_direction"] == "DOWN":
        return "Trend_DOWN"
    else:
        return "Sideway"

# ✅ ฟังก์ชันคำนวณ Dynamic Lot ตาม Balance
def get_dynamic_lot_size(sl_price_distance: float):
    acc_info = mt5.account_info()
    if acc_info is None or sl_price_distance <= 0:
        return 0.1

    balance = acc_info.balance
    risk_amount = balance * (RISK_PERCENT / 100.0)

    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None:
        return 0.1

    tick_size = symbol_info.trade_tick_size       # เช่น 0.01
    tick_value = symbol_info.trade_tick_value     # เช่น 1.00 ต่อ 0.01 จุด สำหรับ 1 lot

    # คำนวณจำนวน tick ที่ SL ห่างจากราคา
    tick_count = sl_price_distance / tick_size

    # คำนวณมูลค่าความเสี่ยงต่อ 1 lot ตาม SL
    loss_per_lot = tick_count * tick_value

    if loss_per_lot <= 0:
        return 0.1

    lot_size = risk_amount / loss_per_lot
    lot_size = max(0.01, min(round(lot_size, 2), 5.0))

    print(f"[LotSize] SL: {sl_price_distance:.2f} → TickCount: {tick_count:.1f} → Lot: {lot_size:.2f}")
    return lot_size

def calculate_tp_sl_structure_based(confidence: float, order_type: str):
    df = get_mt5_data(SYMBOL, TIMEFRAME, 100)
    df_m15 = get_mt5_data(SYMBOL, mt5.TIMEFRAME_M15, 100)
    df_h1 = get_mt5_data(SYMBOL, HIGHER_TIMEFRAME, 200)
    tick = mt5.symbol_info_tick(SYMBOL)

    if df is None or df_h1 is None or tick is None or len(df) < 20 or len(df_h1) < 30:
        return 100, 60, 90, 1.5

    atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=ATR_PERIOD).iloc[-1]
    price = tick.ask if order_type == "BUY" else tick.bid
    spread = tick.ask - tick.bid
    score_m15, dir_m15 = calculate_trend_score_pro(df_m15)
    if "trend_score" not in df.columns:
        df["trend_score"] = score_m15
    trend_score = df['trend_score'].iloc[-1]  # 🔍 ใช้ trend_score ล่าสุด

    # 🔍 หาจุด Swing H1
    swing_highs, swing_lows = detect_market_structure(df_h1, window=10)

    # ✅ คำนวณ TP ตาม swing
    tp = 0
    if order_type == "BUY" and swing_highs:
        swing_price = max([p[2] for p in swing_highs[-3:]])
        tp = max(1.0, swing_price - price - spread)
    elif order_type == "SELL" and swing_lows:
        swing_price = min([p[2] for p in swing_lows[-3:]])
        tp = max(1.0, price - swing_price - spread)

    # ❓ ถ้า TP ต่ำเกิน → fallback เป็น ATR
    if tp <= atr * 0.5:
        tp = np.interp(confidence, [0.0, 1.0], [1.0, 2.0]) * atr

    # ✅ Limit TP ตามระดับ Trend
    if abs(trend_score) < 15:        # Sideway → ไม่เกิน 7 pip
        max_tp = 5
    elif abs(trend_score) < 35:      # เทรนด์อ่อน → ไม่เกิน 100 จุด
        max_tp = 7
    else:                            # เทรนด์แรง → ไม่เกิน 150 จุด
        max_tp = 9

    tp = round(min(tp, max_tp), 2)

    # ✅ SL Adaptive + spread buffer
    sl = round(np.interp(confidence, [0.0, 1.0], [1.0, 0.6]) * atr + spread * 0.5, 2)
    trailing_stop = round(TRAILING_STOP_MULTIPLIER * atr, 2)
    rr_ratio = round(tp / sl, 2) if sl > 0 else 0.0

    print(f"[TP/SL] Trend: {trend_score} → TP: {tp:.2f}, SL: {sl:.2f}, RR: {rr_ratio}")
    return round(tp, 2), round(sl, 2), round(trailing_stop, 2), rr_ratio

# ✅ ฟังก์ชันตรวจจับแนวโน้มเปลี่ยนแปลง (เวอร์ชันใหม่)
def detect_trend_change(df):
    if df is None or 'ADX' not in df.columns:
        return False

    adx = df['ADX'].iloc[-1]
    ema_50_now, ema_50_prev = df['EMA_50'].iloc[-1], df['EMA_50'].iloc[-2]
    ema_200_now, ema_200_prev = df['EMA_200'].iloc[-1], df['EMA_200'].iloc[-2]
    order_flow_now, order_flow_prev = df['Order_Flow'].iloc[-1], df['Order_Flow'].iloc[-2]
    volume_now = df['volume'].iloc[-1]
    avg_volume = df['volume'].rolling(10).mean().iloc[-1]

    # ✅ เงื่อนไขยืนยันหลายอย่างร่วมกัน
    adx_drop = adx < 20
    ema_crossover = (ema_50_prev > ema_200_prev and ema_50_now < ema_200_now) or \
                    (ema_50_prev < ema_200_prev and ema_50_now > ema_200_now)
    of_flip = (order_flow_prev > 0 and order_flow_now < 0) or \
              (order_flow_prev < 0 and order_flow_now > 0)

    volume_fade = volume_now < (avg_volume * 0.6)  # ✅ ตลาดเบาบางผิดปกติ

    confirm_count = sum([adx_drop, ema_crossover, of_flip])

    if confirm_count >= 2 and volume_fade:
        print("⚠️ Trend Change Detected [Multi Confirmed + Low Vol]")
        return True

    return False

# ✅ ฟังก์ชันดึงกำไรล่าสุดจาก TRADE_LOG_NAME
def get_last_trade_profit():
    """ ✅ ดึงกำไรจาก TRADE_LOG_NAME (รายการล่าสุด) """
    try:
        with open(TRADE_LOG_NAME, "r") as f:
            trade_log = [json.loads(line) for line in f]  # ✅ โหลด trade log ทั้งหมดเป็น list ของ dict

        if not trade_log:  # ✅ ถ้าไม่มีข้อมูล
            return 0.0
        
        last_trade = trade_log[-1]  # ✅ ดึงรายการล่าสุด
        
        if isinstance(last_trade, dict):  # ✅ เช็คให้แน่ใจว่าเป็น dictionary
            return float(last_trade.get("profit", 0.0))  # ✅ คืนค่า profit ถ้ามี
        
    except (FileNotFoundError, json.JSONDecodeError):  # ✅ ป้องกัน error ถ้าไฟล์ไม่มีหรือ JSON เสีย
        return 0.0

    return 0.0  # ✅ ป้องกัน error เพิ่มเติม

def is_market_open(symbol=SYMBOL):
    info = mt5.symbol_info(symbol)
    if info is None:
        print("❌ ไม่พบข้อมูล Symbol")
        return False
    return info.visible and info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL

def modify_sl(position, new_sl):
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": position.ticket,
        "sl": new_sl,
        "tp": position.tp,
        "symbol": position.symbol,
        "magic": position.magic,
        "comment": "Modified SL by AI",
    }
    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"🔁 SL Modified to {new_sl:.2f} for Position {position.ticket}")
        telegram_notify(f"🔁 SL moved to {new_sl:.2f} | Pos: {position.ticket}")
        return True
    else:
        print(f"⚠️ Failed to modify SL: {result.retcode}")
        return False


# ✅ Smart Trailing SL: ถ้าราคาวิ่งถึง 75% ของ TP → ย้าย SL ไป 50%
def smart_trailing_sl(pos):
    if pos.tp == 0 or pos.sl == 0:
        return

    tp_distance = abs(pos.tp - pos.price_open)
    current_price = mt5.symbol_info_tick(SYMBOL).bid if pos.type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(SYMBOL).ask
    moved = False

    if pos.type == mt5.ORDER_TYPE_BUY:
        if current_price >= pos.price_open + 0.75 * tp_distance:
            new_sl = pos.price_open + 0.5 * tp_distance
            if new_sl > pos.sl:
                moved = True
    else:  # SELL
        if current_price <= pos.price_open - 0.75 * tp_distance:
            new_sl = pos.price_open - 0.5 * tp_distance
            if new_sl < pos.sl:
                moved = True

    if moved:
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": pos.ticket,
            "sl": new_sl,
            "tp": pos.tp,
            "symbol": pos.symbol,
            "magic": 202504,
            "comment": "Smart Trailing SL"
        }
        telegram_notify(f"🕒 {datetime.now().strftime('%H:%M:%S')} : ✅ ขยับ SL กันทุนที่ 50% ของ TP @ |  | New TP: {new_sl}")
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"🔁 Smart Trailing SL moved to: {new_sl:.2f}")
        else:
            print(f"⚠️ Failed to update SL: {result.retcode}")

def monitor_trades():
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is None:
        return

    for pos in positions:
        # เช็คว่าเป็นออเดอร์ของระบบ (magic number หรือ comment)
        if pos.magic != 202504:
            continue
        close_all_if_unrealized_profit_exceeds(threshold=5.0)  # ✅ เช็คกำไรรวม
        # ✅ เรียกใช้ Smart Lock Profit เมื่อเข้าใกล้ TP
        smart_trailing_sl(pos)

def close_all_if_unrealized_profit_exceeds(threshold=5.0):
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is None:
        print("❌ positions_get = None → อาจยังไม่ได้เชื่อมต่อ MT5")
        print(f"🧪 MT5 Error: {mt5.last_error()}")
        return
    if len(positions) == 0:
        return

    total_profit = sum(pos.profit for pos in positions)
    if total_profit >= threshold:
        print(f"💰 Total Unrealized Profit = {total_profit:.2f} → Close All!")
        telegram_notify(f"🕒 {datetime.now().strftime('%H:%M:%S')} : ✅ Close all profit sum > 5$ Profit = {total_profit:.2f} → Close All!")
        for pos in positions:
            price = mt5.symbol_info_tick(SYMBOL).ask if pos.type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(SYMBOL).bid
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": SYMBOL,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_BUY if pos.type == mt5.ORDER_TYPE_SELL else mt5.ORDER_TYPE_SELL,
                "position": pos.ticket,
                "price": price,
                "deviation": 20,
                "magic": 202504,
                "comment": "Auto Close All Profit",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC
            }
            result = mt5.order_send(request)
            if result is None:
                print("❌ order_send = None")
                print(f"🧪 Request: {request}")
                print(f"🧪 MT5 Error: {mt5.last_error()}")
            elif result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"✅ Closed position {pos.ticket} at price {price}")
                log_trade(
                    action="CLOSE",
                    symbol=pos.symbol,
                    order_type="SELL" if pos.type == mt5.ORDER_TYPE_BUY else "BUY",
                    lot_size=pos.volume,
                    entry_price=pos.price_open,
                    exit_price=price,
                    profit=pos.profit,
                    drawdown=abs(pos.price_open - price),
                    reward_risk_ratio=round(abs(price - pos.price_open) / (abs(pos.price_open - price) + 1e-6), 2),
                    closed=True,
                    market_condition="Profit Sum > 5$",
                    trend_score=0,
                    liquidity_zone=0,
                    volume_at_entry=0,
                    order_flow_at_entry=0,
                    spread=abs(mt5.symbol_info_tick(pos.symbol).ask - mt5.symbol_info_tick(pos.symbol).bid),
                    trend_direction="Unknown",
                    pattern_score=0,
                    pattern_type="NONE"
                )
            else:
                print(f"❌ Failed to close position {pos.ticket}: {result.retcode} ({result.comment})")

# ✅ ฟังก์ชันปิดออเดอร์อัตโนมัติ (เวอร์ชันใหม่)
def close_positions():
    df = get_mt5_data(SYMBOL, TIMEFRAME, 100)
    if df is None:
        return

    trend_info = check_trend_quality(df)
    if trend_info["trend_direction"] == "SIDEWAY":
        return

    open_positions = mt5.positions_get(symbol=SYMBOL)
    if not open_positions:
        return

    for pos in open_positions:
        close_price = mt5.symbol_info_tick(pos.symbol).bid if pos.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(pos.symbol).ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": pos.ticket,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "price": close_price,
            "deviation": 20,
            "magic": pos.magic,
            "comment": "AI Auto Close (Trend Change)",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"✅ ปิดออเดอร์สำเร็จ | {'BUY' if pos.type == 0 else 'SELL'} | Volume: {pos.volume} | Profit: {pos.profit:.2f}")
            telegram_notify(f"🕒 {datetime.now().strftime('%H:%M:%S')} : ✅ ปิดออเดอร์สำเร็จ | {'BUY' if pos.type == 0 else 'SELL'} | Volume: {pos.volume} | Profit: {pos.profit:.2f}")
            log_trade(
                action="CLOSE",
                symbol=pos.symbol,
                order_type="BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL",
                lot_size=pos.volume,
                entry_price=pos.price_open,
                exit_price=close_price,
                profit=pos.profit,
                drawdown=abs(pos.price_open - close_price),
                reward_risk_ratio=abs(pos.price_current - pos.price_open) / (abs(pos.price_open - close_price) + 1e-6),
                confidence_score=1.0,
                closed=True,
                market_condition="Trend Change Auto Close",
                trend_score=trend_info.get("trend_score", 0),
                liquidity_zone=trend_info.get("liquidity_zone", 0.0),
                volume_at_entry=trend_info.get("volume", 0),
                order_flow_at_entry=trend_info.get("order_flow", 0.0),
                spread=abs(mt5.symbol_info_tick(SYMBOL).ask - mt5.symbol_info_tick(SYMBOL).bid),
                trend_direction=trend_info.get("trend_direction", "Unknown"),
                pattern_score=0,
                pattern_type="NONE"
            )
        else:
            print(f"❌ ปิดออเดอร์ล้มเหลว | Error Code: {result.retcode if result else 'No response'}")

# ✅ ฟังก์ชันส่งคำสั่งซื้อขาย (เพิ่มความปลอดภัย: ตรวจสอบ trend ก่อนเปิด order)
def place_order(symbol, order_type, row):
    if not row.get('liquidity_zone_match', 0):
        print("⛔ อยู่ห่าง Liquidity Zone เกินกำหนด ไม่เข้าไม้")
        return
    if len(mt5.positions_get(symbol=symbol)) >= MAX_TRADES:
        print(f"⚠️ Max Trades Limit Reached ({MAX_TRADES})")
        return

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print("❌ ไม่สามารถดึงราคาได้")
        return

    price = tick.ask if order_type == "BUY" else tick.bid
    spread = tick.ask - tick.bid
    if spread > 2.5:  # ปรับตามกลยุทธ์ เช่น 25 points = 2.5 pips
        print(f"⚠️ Spread กว้างเกินไป: {spread:.2f} → ยกเลิกการเข้าไม้")
        return
    confidence = row.get("confidence_score_advanced", 0.5)
    tp, sl, trailing_stop, rr = calculate_tp_sl_structure_based(confidence, order_type)
    volume = get_dynamic_lot_size(sl)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": price - sl if order_type == "BUY" else price + sl,
        "tp": price + tp if order_type == "BUY" else price - tp,
        "deviation": 20,
        "magic": 202504,
        "comment": "AI V12 Entry",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    print(f"[TP/SL] Conf: {confidence:.2f} → TP: {tp:.2f}, SL: {sl:.2f}")
    positions = mt5.positions_get(symbol=symbol)
    if positions:
        for pos in positions:
            # ห้ามเปิดไม้สวน
            if (order_type == "BUY" and pos.type == mt5.ORDER_TYPE_SELL) or \
            (order_type == "SELL" and pos.type == mt5.ORDER_TYPE_BUY):
                print(f"⛔ มีไม้ฝั่งตรงข้ามอยู่แล้ว ({'SELL' if pos.type == 1 else 'BUY'}) → ยกเลิกคำสั่ง {order_type}")
                return
    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"✅ Order Sent: {order_type} @ {price:.2f}")
        obs = build_observation_vector(row)
        action = 1 if order_type == "BUY" else 2
        reward = 0
        profit = 0
        replay_buffer.add(obs, action, reward, profit)
        telegram_notify(f"🕒 {datetime.now().strftime('%H:%M:%S')} : ✅ {action_to_str(action)} @ {price:.2f} | Conf: {confidence:.2f} | TP: {tp:.2f} | SL: {sl:.2f}")

        df_1h = get_mt5_data(SYMBOL, HIGHER_TIMEFRAME, 100)
        structure_signal = get_market_structure_signal(df_1h)

        # ✅ Log OPEN พร้อม structure_signal
        log_trade(
            drawdown=0,
            action="OPEN",
            symbol=symbol,
            order_type=order_type,
            lot_size=volume,
            entry_price=price,
            take_profit=tp,
            stop_loss=sl,
            trailing_stop=trailing_stop,
            confidence_score=row["confidence_score_advanced"],
            trend_score=row["trend_score"],
            liquidity_zone=row.get("liquidity_zone", 0.0),
            volume_at_entry=row["volume"],
            order_flow_at_entry=row.get("order_flow", 0.0),
            spread=spread,
            trend_direction=row.get("trend_direction", "Unknown"),
            pattern_score=0.0,
            pattern_type="NONE",
            market_condition=row.get("market_condition", "Unknown"),
            structure_signal=structure_signal,
            reward_risk_ratio=rr
        )

        # ✅ ตั้ง Trailing Stop (Modify SL หลังเปิด)
        trailing_stop_price = price - trailing_stop if order_type == "BUY" else price + trailing_stop
        modify_request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": result.order,
            "sl": trailing_stop_price,
            "tp": request["tp"],
            "symbol": symbol,
            "magic": 202504,
            "comment": "Set Trailing Stop"
        }
        mod_result = mt5.order_send(modify_request)
        if mod_result.retcode == mt5.TRADE_RETCODE_DONE:
            print("🔁 Trailing Stop Set Successfully")
        else:
            print(f"⚠️ Failed to set Trailing Stop: {mod_result.retcode}")

    else:
        print(f"❌ Order Failed: {result.retcode}")

# ✅ ฟังก์ชันบันทึกข้อมูลเทรดพร้อมกำไร/ขาดทุนลง Log
def log_trade(**kwargs):
    log_data = {
        "time": str(pd.Timestamp.now()),
        "symbol": kwargs.get("symbol", SYMBOL),
        "action": kwargs.get("action", "OPEN"),
        "order_type": kwargs.get("order_type", ""),
        "lot_size": float(kwargs.get("lot_size", 0.0)),
        "entry_price": float(kwargs.get("entry_price", 0.0)),
        "exit_price": float(kwargs.get("exit_price", 0.0)),
        "take_profit": float(kwargs.get("take_profit", 0.0)),
        "stop_loss": float(kwargs.get("stop_loss", 0.0)),
        "trailing_stop": float(kwargs.get("trailing_stop", 0.0)),
        "confidence_score": float(kwargs.get("confidence_score", 0.0)),
        "holding_time": int(kwargs.get("holding_time", 0)),
        "profit": float(kwargs.get("profit", 0.0)),
        "drawdown": float(kwargs.get("drawdown", 0.0)),
        "reward_risk_ratio": float(kwargs.get("reward_risk_ratio", 0.0)),
        "closed": kwargs.get("closed", False),
        "tp_hit": kwargs.get("tp_hit", False),
        "sl_hit": kwargs.get("sl_hit", False),
        "market_condition": kwargs.get("market_condition", ""),
        "structure_signal": kwargs.get("structure_signal", ""),
        "trend_score": int(kwargs.get("trend_score", 0)),
        "liquidity_zone": float(kwargs.get("liquidity_zone", 0.0)),
        "volume_at_entry": float(kwargs.get("volume_at_entry", 0.0)),
        "order_flow_at_entry": float(kwargs.get("order_flow_at_entry", 0.0)),
        "spread": float(kwargs.get("spread", 0.0)),
        "trend_direction": kwargs.get("trend_direction", ""),
        "pattern_score": float(kwargs.get("pattern_score", 0.0)),
        "pattern_type": kwargs.get("pattern_type", "")
    }
    with open(TRADE_LOG_NAME, "a", encoding="utf-8") as f:
        json.dump(log_data, f)
        f.write("\n")
    print(f"📘 Logged: {log_data['action']} | {log_data['order_type']} @ {log_data['entry_price']} → {log_data['exit_price']} | Profit: {log_data['profit']:.2f}")

# ✅ ฟังก์ชันเก็บ transition
transition_buffer = []
def store_transition(obs_prev, action, obs_next, reward):
    if obs_prev is not None and obs_next is not None:
        transition_buffer.append((obs_prev, action, obs_next, reward))
        with open(REPLAY_NAME, "wb") as f:
            pickle.dump(transition_buffer, f)
        print(f"🧠 Stored transition ({len(transition_buffer)} records) → {REPLAY_NAME}")

# ✅ compute_reward ไม่มี pattern score แล้ว
def compute_reward(action, df):
    profit = get_last_trade_profit()
    trend = check_trend_quality(df)
    market_tf = analyze_multi_timeframe()
    momentum = df['close'].diff(4).iloc[-1]

    is_w_entry = int(df['is_w_pattern_entry'].iloc[-1]) if 'is_w_pattern_entry' in df.columns else 0
    is_m_entry = int(df['is_m_pattern_entry'].iloc[-1]) if 'is_m_pattern_entry' in df.columns else 0

    df_1h = get_mt5_data(SYMBOL, HIGHER_TIMEFRAME, 100)
    structure_signal = get_market_structure_signal(df_1h)

    reward = 0.0
    if action == 0:
        reward = 0.01 if abs(profit) < 1.0 else -0.02
    elif action == 1:
        if trend["trend_direction"] == "UP" and market_tf == "Trend_UP":
            reward += profit * 0.1
            reward += trend["confidence_buy"] * 0.4
            reward += 0.1 if trend["liquidity_zone"] < df['close'].iloc[-1] else -0.1
        else:
            reward -= 0.2
        reward += 0.05 if momentum > 0 else -0.05
        if structure_signal == "buy":
            reward += 0.1
        elif structure_signal == "sell":
            reward -= 0.1
        if is_w_entry == 1:
            reward += 0.2  # ✅ Bonus เมื่อเข้าไม้ BUY ตรง W
    elif action == 2:
        if trend["trend_direction"] == "DOWN" and market_tf == "Trend_DOWN":
            reward += profit * 0.1
            reward += trend["confidence_sell"] * 0.4
            reward += 0.1 if trend["liquidity_zone"] > df['close'].iloc[-1] else -0.1
        else:
            reward -= 0.2
        reward += 0.05 if momentum < 0 else -0.05
        if structure_signal == "sell":
            reward += 0.1
        elif structure_signal == "buy":
            reward -= 0.1
        if is_m_entry == 1:
            reward += 0.2  # ✅ Bonus เมื่อเข้าไม้ SELL ตรง M

    return round(reward, 4)


# ✅ ฟังก์ชันคำนวณ Confidence Score ด้วยข้อมูล trend
def calculate_confidence_score(df):
    trend = check_trend_quality(df)
    market_tf = analyze_multi_timeframe()

    if trend["trend_direction"] == "UP" and market_tf == "Trend_UP":
        return trend["confidence_buy"]
    elif trend["trend_direction"] == "DOWN" and market_tf == "Trend_DOWN":
        return trend["confidence_sell"]
    else:
        return 0.0

def order_type_based_on_trend(trend_score):
    if trend_score == 1:
        return "BUY"
    elif trend_score == -1:
        return "SELL"
    else:
        return "HOLD"

# ✅ decision_engine ใหม่ ไม่มี pattern แล้ว
def decision_engine(row, df_full, action=None):
    row = row.copy()
    momentum = row.get("momentum", 0.0)
    trend_dir = row.get("trend_direction", "SIDEWAY")
    trend_score = row.get("trend_score", 0.0)
    df_1h = get_mt5_data(SYMBOL, HIGHER_TIMEFRAME, 100)
    if df_1h is None or len(df_1h) < 30:
        return "HOLD", "❌ ไม่มีข้อมูล H1 เพียงพอ"

    structure_signal = get_market_structure_signal(df_1h)

    # 🧠 เพิ่ม structure entry flag
    row["structure_buy"] = 1 if structure_signal == "buy" else 0
    row["structure_sell"] = 1 if structure_signal == "sell" else 0

    # ✅ ตรวจ Liquidity Zone Match
    close_price = row['close']
    liquidity_zone = row.get('liquidity_zone', 0.0)
    zone_match = abs(close_price - liquidity_zone) / (close_price + 1e-6) < 0.015
    row['liquidity_zone_match'] = int(zone_match)

    # ✅ คำนวณ confidence จาก row
    confidence = row.get("confidence_score_advanced", 0.0)
    reasons = []

    # ✅ เพิ่ม confidence จาก pattern/structure
    if row.get("structure_buy", 0) == 1:
        confidence += 0.05
        reasons.append("Structure BUY")
    if row.get("structure_sell", 0) == 1:
        confidence += 0.05
        reasons.append("Structure SELL")
    if row.get("is_w_pattern_entry", 0) == 1:
        confidence += 0.1
        reasons.append("W Pattern")
    if row.get("is_m_pattern_entry", 0) == 1:
        confidence += 0.1
        reasons.append("M Pattern")

    # ✅ MAIN LOGIC: หากมั่นใจพอ ให้ตัดสินใจจาก Signal ก่อน แล้ว fallback ไป PPO
    if round(confidence, 2) >= 0.40:
        # 🧠 ตรวจ wick ก่อน
        candle_range = row['high'] - row['low'] + 1e-6
        wick_top = row['high'] - max(row['close'], row['open'])
        wick_bottom = min(row['close'], row['open']) - row['low']
        wick_top_ratio = wick_top / candle_range
        wick_bottom_ratio = wick_bottom / candle_range
        
        # 🎯 เข้าตาม Pattern ก่อน
        if row.get("is_w_pattern_entry", 0) == 1 and wick_top_ratio <= 0.5:
            return "BUY", "W Pattern Retest"
        if row.get("is_m_pattern_entry", 0) == 1 and wick_bottom_ratio <= 0.5:
            return "SELL", "M Pattern Retest"

        # 🔁 เทรนด์ตรง + structure ตรง
        if trend_score > 20 and trend_dir == "UP" and structure_signal == 'buy' and wick_top_ratio <= 0.5:
            return "BUY", "Trend + Structure"
        if trend_score < -20 and trend_dir == "DOWN" and structure_signal == 'sell' and wick_bottom_ratio <= 0.5:
            return "SELL", "Trend + Structure"

        # 🤖 Fallback → ให้ PPO ตัดสินใจ
        if action == 1 and wick_top_ratio <= 0.5:
            return "BUY", "PPO Action"
        elif action == 2 and wick_bottom_ratio <= 0.5:
            return "SELL", "PPO Action"
        
        # 🚫 ถ้าผิดฝั่ง wick → ปฏิเสธ
        if action in [1, 2]:
            reasons.append("🚫 ปลายใส้ผิดฝั่ง (wick)")

    # ❌ confidence ไม่พอ
    reasons.append(f"Confidence ไม่พอ: {confidence:.2f}")
    return "HOLD", " | ".join(reasons)


# ✅ สร้าง Custom Gym Environment ให้ RL Model ใช้งาน
class TradingEnv(Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32)
        
        df_raw, multi_tf_align = get_multi_tf_alignment()
        if df_raw is not None and len(df_raw) >= 50:
            self.df = calculate_indicators(df_raw, multi_tf_align=multi_tf_align)
        else:
            self.df = pd.DataFrame()  # fallback

    def reset(self, seed=None, options=None):
        self.df = get_mt5_data(SYMBOL, TIMEFRAME, 100)
        self.df = enrich_features(self.df)
         # ✅ แก้ตรงนี้! เพิ่ม structure_signal แล้วสร้าง structure_buy/sell
        structure_signal = get_market_structure_signal(get_mt5_data(SYMBOL, HIGHER_TIMEFRAME, 100))
        self.df['structure_buy'] = 1 if structure_signal == "buy" else 0
        self.df['structure_sell'] = 1 if structure_signal == "sell" else 0
        self.current_step = 0
        return build_observation_vector(self.df.iloc[self.current_step]), {}

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        structure_signal = get_market_structure_signal(get_mt5_data(SYMBOL, HIGHER_TIMEFRAME, 100))
        self.df['structure_buy'] = 1 if structure_signal == "buy" else 0
        self.df['structure_sell'] = 1 if structure_signal == "sell" else 0
        obs = build_observation_vector(self.df.iloc[self.current_step])
        reward = 0.0
        return obs, reward, done, False, {}

# ✅ เพิ่ม Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def add(self, obs, action, reward, profit):
        """เพิ่มข้อมูลใหม่เข้า buffer"""
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)  # ลบอันเก่าสุด
        self.buffer.append((obs, action, reward, profit))

    def store(self, obs, action, reward, profit_loss):
        """ เก็บข้อมูล Trade Log ลง Buffer """
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)  # ลบอันเก่าออก
        self.buffer.append((obs, action, reward, profit_loss))

    def sample(self, batch_size=64):
        """ สุ่มข้อมูลจาก Buffer มา Train AI """
        if len(self.buffer) < batch_size:
            return None
        return random.sample(self.buffer, batch_size)

# ✅ โหลด Log Trade ที่มี Profit/Loss
def load_trade_log():
    trade_log = []
    log_file = TRADE_LOG_NAME

    # ✅ ถ้าไฟล์ยังไม่เคยมี ให้สร้างไฟล์ว่าง (แบบ line-by-line JSON)
    if not os.path.exists(log_file):
        print("⚠️ TRADE_LOG_NAME ไม่พบ! กำลังสร้างไฟล์ใหม่...")
        with open(log_file, "w", encoding="utf-8") as f:
            pass  # ❌ อย่าเขียน json.dump([]) เพราะเราจะเก็บแบบบรรทัดละ JSON

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                trade = safe_load_trade_log_entry(line)
                if trade is not None:
                    trade_log.append(trade)

        # ✅ ล้าง trade log เก่าเกิน 1000 รายการ
        if len(trade_log) > 1000:
            print(f"♻️ ลดขนาด trade log จาก {len(trade_log)} → 1000 records")
            trade_log = trade_log[-1000:]

    except Exception as e:
        print(f"❌ ERROR loading TRADE_LOG_NAME: {e}")
        return []

    print(f"📂 Loaded Trade Log ({len(trade_log)} records from {log_file}))")
    return trade_log

# ✅ แก้ convert_old_logs_to_observation ให้รองรับ structure_signal

def convert_old_logs_to_observation(trade):
    trend_dir = {"UP": 1, "DOWN": -1, "SIDEWAY": 0}.get(trade.get("trend_direction", "SIDEWAY"), 0)

    structure_buy = 1 if trade.get("structure_signal", "") == "buy" else 0
    structure_sell = 1 if trade.get("structure_signal", "") == "sell" else 0

    return np.array([
        float(trade.get("entry_price", 0)),
        float(trade.get("volume_at_entry", 0)),
        float(trade.get("ema_20", 0)),
        float(trade.get("ema_50", 0)),
        float(trade.get("adx", 20)),
        float(trade.get("rsi_14", 55)),
        float(trade.get("trend_score", 0)),
        float(trade.get("trend_score", 0)) / 50.0,
        int(trade.get("multi_tf_align", 1)),
        0,
        1,
        1,
        float(trade.get("confidence_score", 0.5)),
        float(trade.get("reward_risk_ratio", 1.5)),
        float(trade.get("liquidity_zone", 0)),
        trend_dir,
        1,
        float(trade.get("momentum", 0.1)),
        float(trade.get("spread", 20.0)),
        structure_buy,
        structure_sell
    ], dtype=np.float32)

# ✅ แก้ bug การโหลด trade log ที่บางบรรทัดเป็น int
def safe_load_trade_log_entry(line):
    try:
        trade = json.loads(line)
        if isinstance(trade, dict):
            return trade
        else:
            print("⚠️ Invalid log entry (not dict):", trade)
            return None
    except json.JSONDecodeError:
        print("⚠️ Skipping invalid JSON entry")
        return None

# ✅ สร้าง Replay Buffer
load_replay_buffer()
trade_log_data = load_trade_log()
for trade in trade_log_data:
    if not isinstance(trade, dict):
        print("⚠️ Skipping invalid trade log:", trade)
        continue

    raw_action = trade.get("action", "")
    if isinstance(raw_action, str):
        action_str = raw_action.upper()
    elif isinstance(raw_action, int):
        action_str = {1: "BUY", 2: "SELL", 0: "HOLD"}.get(raw_action, "HOLD")
    else:
        action_str = "HOLD"

    if action_str == "BUY":
        action = 1
    elif action_str == "SELL":
        action = 2
    else:
        action = 0

    obs = convert_old_logs_to_observation(trade)
    reward = float(trade.get("profit", 0))
    replay_buffer.store(obs, action, reward, reward)

print(f"✅ Replay Buffer ถูกสร้างแล้ว ({len(replay_buffer.buffer)} records)")
save_replay_buffer()

# ✅ เทรดอัตโนมัติ + RL Online Learning
env = DummyVecEnv([lambda: TradingEnv()])
# ✅ โหลด / สร้างโมเดล PPO
if os.path.exists(MODEL_PATH):
    model = PPO.load(MODEL_PATH, env=env)
    print("📥 Loaded existing model")
else:
    model = PPO("MlpPolicy", env, verbose=1)
    print("🧠 Created new PPO model")

# 🎯 แปลง action → string เพื่อให้เข้าใจง่าย
def action_to_str(action):
    """ 🔄 แปลง action (int, list, np.ndarray) → string แบบปลอดภัย """
    if isinstance(action, np.ndarray):
        action = action.item()  # ✅ รองรับ array(1), array([1]), array([[1]]) ก็ได้
    elif isinstance(action, list):
        action = action[0]
    return {
        0: "⏸️ HOLD",
        1: "🟢 BUY",
        2: "🔴 SELL"
    }.get(int(action), "❓ UNKNOWN")

# ✅ ฟังก์ชันปรับ Hyperparameter ด้วย Optuna (เวอร์ชันเร็ว+ฉลาด)
def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 3e-4)
    gamma = trial.suggest_float('gamma', 0.85, 0.99)
    entropy_coef = trial.suggest_float('ent_coef', 0.01, 0.2)
    n_steps = trial.suggest_int('n_steps', 1024, 4096, step=512)
    clip_range = trial.suggest_float('clip_range', 0.2, 0.4)
    gae_lambda = trial.suggest_float('gae_lambda', 0.9, 1.0)

    model = PPO("MlpPolicy", env,
                learning_rate=learning_rate,
                gamma=gamma,
                ent_coef=entropy_coef,
                n_steps=n_steps,
                clip_range=clip_range,
                gae_lambda=gae_lambda,
                device="cpu",
                verbose=0)

    # ✅ Train แค่สั้น ๆ แล้ววัด reward เฉลี่ยเร็วขึ้น
    model.learn(total_timesteps=3000)
    df = get_mt5_data(SYMBOL, TIMEFRAME, 100)
    if df is None or len(df) < 50:
        return -9999

    rewards = [compute_reward(random.choice([0, 1, 2]), df) for _ in range(20)]
    return np.mean(rewards)

# ✅ รัน Optuna หา Hyperparameter ที่ดีที่สุด
study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(),
    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2)
)
study.optimize(objective, n_trials=5, n_jobs=1)

best_params = study.best_params
print("🔥 Best Hyperparameters:", best_params)

# === Performance Tracker ===
performance_history = deque(maxlen=20)
STEP_LEARN = 0

def update_performance(reward):
    performance_history.append(reward)
    return np.mean(performance_history)

# === Auto Learn Trigger ===
def auto_learn_if_needed():
    global STEP_LEARN
    STEP_LEARN += 1
    if len(replay_buffer) > 1000 and STEP_LEARN % 5 == 0:
        mean_reward = np.mean(performance_history)
        print(f"📊 Avg Reward (last 20): {mean_reward:.4f}")
        model.learn(total_timesteps=512, reset_num_timesteps=False)
        model.save(MODEL_PATH)
        save_replay_buffer()
        print("🤖 Auto-learn complete!")

model = PPO("MlpPolicy", env, **best_params, verbose=0)
model.set_env(env)

while True:
    try:
        if not is_market_open():
            print("⏸ ตลาดปิดอยู่ รอรอบต่อไป...")
            time.sleep(60)
            continue
        
        monitor_trades()
        time.sleep(30)  # เช็คทุก 30 วินาที
        
        df_raw, multi_tf_align = get_multi_tf_alignment()
        if df_raw is None or len(df_raw) < 50:
            print("⚠️ No data, รอข้อมูลใหม่...")
            time.sleep(60)
            continue

        # ✅ STEP 2: คำนวณ indicators พร้อมส่ง multi_tf_align เข้าไป
        df = calculate_indicators(df_raw, multi_tf_align=multi_tf_align)
        row = df.iloc[-1].copy()
        # ✅ เพิ่ม structure_buy / structure_sell ก่อนสร้าง obs
        structure_signal = get_market_structure_signal(get_mt5_data(SYMBOL, HIGHER_TIMEFRAME, 100))
        row['structure_buy'] = 1 if structure_signal == "buy" else 0
        row['structure_sell'] = 1 if structure_signal == "sell" else 0
        
        obs_next = build_observation_vector(row)

        # ✅ ใช้ model ทำนาย action ตาม observation
        if 'obs_prev' in globals():
            obs_prev = obs_next
        else:
            obs_prev = obs_next  # ตั้งค่าเริ่มต้น
        obs = obs_prev  # ใช้ obs ก่อนตัดสินใจ
        action, _ = model.predict(obs_prev)
        reward = compute_reward(action, df)
        signal, reason = decision_engine(row, df[-30:], action)

        # ✅ แสดงผลรอบนี้แบบละเอียด
        print("───────────────")
        print(f"🕒 {datetime.now().strftime('%H:%M:%S')} | Action: {action_to_str(action)} ({action}) | Signal: {signal}")
        print(f"🔎 Reason: {reason}")
        print(f"📊 Conf: {row['confidence_score_advanced']:.2f} | Trend Score: {row['trend_score']} | Liquidity Match: {row['liquidity_zone_match']}")

        # ✅ แสดง Pattern Entry
        if row.get("is_w_pattern_entry", 0) == 1:
            print("🔷 พบ W Pattern Entry (Retest) ✅")
        if row.get("is_m_pattern_entry", 0) == 1:
            print("🔶 พบ M Pattern Entry (Retest) ✅")

        # ✅ ตัดสินใจเข้าไม้
        if signal == "BUY":
            place_order(SYMBOL, "BUY", row)
        elif signal == "SELL":
            place_order(SYMBOL, "SELL", row)
        else:
            print("⏸ No trade signal.")

        reward = compute_reward(action, df)
        profit = row.get("profit", 0)
        replay_buffer.add(obs, action, reward, profit)

        if reward is not None:
            mean_reward = update_performance(reward)
            auto_learn_if_needed()
        time.sleep(60)

    except Exception as e:
        error_msg = f"❌ ERROR: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        telegram_notify(f"🕒 {datetime.now().strftime('%H:%M:%S')} : {error_msg}")
        try:
            save_replay_buffer()
            print("📦 Saved replay buffer before crash.")
        except:
            print("❌ Failed to save buffer on crash.")
        time.sleep(30)
