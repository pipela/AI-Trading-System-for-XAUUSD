# AI-Trading-System-for-XAUUSD
แจกระบบ AI เทรดทองคำด้วย RL (Reinforcement Learning) เป็นหลัก เทรดอัตโนมัติด้วย AI ที่สามารถวิเคราะห์แนวโน้มทองคำ (XAUUSD) และตัดสินใจเข้าออเดอร์แบบ Real-time
ใช้บน MetaTrader 5 โดยใช้ Reinforcement Learning (PPO) + Technical Analysis

#🧠 Gym Environment RL Model 22 shape
จะเทรนต้องใช้ shape ให้เท่ากับ shape ตอนเทรดเพราะไม่อย่างนั้นจะผิดกับโครงสร้างที่มีการเทรนมา

REPLAY_NAME = "replay_buffer_test.pkl"  #แก้ชื่อไฟล์เก็บข้อมูลการเทรนที่นี่ 
MODEL_PATH = "ppo_xauusd_model_test.zip"  #ชื่อไฟล์เก็บข้อมูลโมเดลที่นี่ เปรียบเสมือนมันสมองของ ai

.pkl เปรียบเสมือนประสบการการเรียนรู้การเทรด
.zip  เปรียบเสมือนมันสมองของ ai

## 🔥 Features
- ใช้โมเดล PPO RL สำหรับเข้าออเดอร์แบบ Real-time
- วิเคราะห์แนวโน้มด้วย Multi-Timeframe + EMA + ADX + Order Flow
- ตรวจจับ Pattern เช่น W, M, H&S และ Retest คอ
- Adaptive TP/SL ตาม Confidence Score
- ระบบ Trailing Stop + No-Loss System
- บันทึก Log รายไม้, ใช้ Replay Buffer สำหรับ Online Learning
- พร้อมระบบแจ้งเตือนเข้าออเดอร์ผ่าน Telegram

## 📦 License
MIT – แจกฟรี ใช้ได้ทุกที่ แต่ไม่รับผิดชอบหากใช้งานแล้วขาดทุน
#ใครเอาโค้ดผมไป custom แล้วสำเร็จทำกำไรได้ดี ผมดีใจด้วยครับ ขอแค่ให้เครดิตผมหน่อย
#ส่วนใครอยากขอบคุณด้วยกับเลี้ยงกาแฟก็ยินดีครับ
[👉☕ Buy Me a Coffee](https://buymeacoffee.com/siratchakorn.k)

==============================================================================

# 🤖 AI Trading System for XAUUSD
This project provides an AI-powered gold trading bot using Reinforcement Learning (RL). 
The system analyzes market trends in real-time and makes trading decisions on MetaTrader 5, powered by PPO (Proximal Policy Optimization) and advanced technical analysis.

# 🧠 Gym Environment – RL Model (Shape v22)
⚠️ When training your model, make sure the observation shape matches the one used in real trading. Mismatched shapes will cause errors due to incompatible model structure.

REPLAY_NAME = "replay_buffer_test.pkl"   # Your experience data (learning history)
MODEL_PATH = "ppo_xauusd_model_test.zip" # Your trained model (AI brain)
.pkl → like the experience the AI has learned
.zip → like the brain of the AI (the actual model used in trading)


## 🔥 Features
- Real-time order execution using PPO Reinforcement Learning
- Multi-Timeframe trend analysis with EMA, ADX, and Order Flow
- Smart pattern detection: W, M, Head & Shoulders, Retest Neckline
- Adaptive Take-Profit / Stop-Loss based on Confidence Score
- Auto Trailing Stop and No-Loss System
- Trade logging per entry, with Replay Buffer for online learning
- Instant Telegram notifications for every trade

## 📦 License
MIT License – Free to use anywhere.
⚠️ However, please note: I am not responsible for any losses that may occur from using this bot.

🙏 Giving Credit / Support
If you customize this code and find success with it, I'm truly happy for you!
All I ask is — please give me credit somewhere in your project 🙇

And if you'd like to say thanks with a cup of coffee,
you're more than welcome to do so here:
[👉☕Buy Me a Coffee](https://buymeacoffee.com/siratchakorn.k)





