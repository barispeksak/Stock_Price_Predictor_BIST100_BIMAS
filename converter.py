import ta
import pandas as pd

df = pd.read_csv('bimas_historical_price.csv')

df.rename(columns=
          {"Tarih": "date",
          "Kapanış(TL)": "close",
          "Min(TL)": "low",
          "Max(TL)": "high",
          "Hacim(TL)":"volume"
          }, inplace=True)
df['date'] = pd.to_datetime(df['date'])

df["rsi"] = ta.momentum.RSIIndicator(df['close']).rsi()
df["macd"] = ta.trend.MACD(df['close']).macd()
df["macd_signal"] = ta.trend.MACD(df['close']).macd_signal()
df["macd_diff"] = ta.trend.MACD(df['close']).macd_diff()
df["bb_upper"] = ta.volatility.BollingerBands(df['close']).bollinger_hband()
df["bb_lower"] = ta.volatility.BollingerBands(df['close']).bollinger_lband()
df["ema_5"] = ta.trend.EMAIndicator(df['close'], window=5).ema_indicator()
df["ema_50"] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
df["ema_200"] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
df["roc"] = ta.momentum.ROCIndicator(close=df["close"]).roc()

df = df[["date", "close", "high", "low", "volume", "rsi", "macd", "macd_signal", "macd_diff",
         "bb_upper", "bb_lower", "ema_5", "ema_50", "ema_200", "roc"]]

df.to_csv("bimas_historical_technical_price.csv", index=False)
