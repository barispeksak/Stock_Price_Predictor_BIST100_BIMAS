"""
Created on Sun Jul 30 2025

@author: Barış Pekska
"""
"inspired by Coursera Course Machine Learning Specialization by Andrew Ng"

"importing useful libraries"
# data analysis
import pandas as pd
import numpy as np
# deep learning
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
# preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
# visualization
import matplotlib.pyplot as plt

"Stock Movement Prediction on Turkish Stock BIMAS"
# load historical technical price data
df = pd.read_csv('bimas_historical_technical_price.csv')

"clean dataset by removing rows with missing technical indicators"
df_clean = df.dropna(subset=['rsi', 'macd', 'macd_signal', 'macd_diff', 'bb_upper', 'bb_lower', 'ema_5', 'ema_50', 'ema_200', 'roc'])

df_clean = df_clean.copy()

"extract date components for feature engineering"
# split date string into year, month, day components
splitted = df_clean['date'].str.split('-', expand=True)
df_clean['year'] = splitted[0].astype(int)
df_clean['month'] = splitted[1].astype(int)
df_clean['day'] = splitted[2].astype(int)

"create binary target: 1 if next day price goes up, 0 if down"
df_clean['target'] = np.where(df_clean['close'].shift(-1) > df_clean['close'], 1, 0)

"define feature set including price, volume, technical indicators and date components"
features = ['close', 'high', 'low', 'volume', 'rsi', 'macd', 'macd_signal',
            'macd_diff', 'bb_upper', 'bb_lower', 'ema_5',
            'ema_50', 'ema_200', 'roc', 'year', 'month', 'day']

"remove last row since it has no target value"
df_clean = df_clean[:-1]

"prepare features and target variables"
X = df_clean[features]
Y = df_clean['target']

"apply feature scaling for neural network training"
# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

"split data into training and validation sets"
X_train, X_valid, Y_train, Y_valid = train_test_split(X_scaled, Y, test_size=0.1, random_state=2022)

input_size = X_train.shape[1]

#Building the Neural Network
"Sequential model with dropout regularization"
model = Sequential([
    Dense(units=64, activation='relu', input_shape=(input_size,), name="L1_Hidden"),
    tf.keras.layers.Dropout(0.2),
    Dense(units=32, activation='relu', name="L2_hidded"),
    tf.keras.layers.Dropout(0.15),
    Dense(units=16, activation='relu', name="L3_hidded"),
    Dense(units=1, activation='sigmoid', name="L4_Output")
])

"compile model with binary crossentropy loss for classification"
# Compiling the Neural Network
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

model.summary()

"setup early stopping to prevent overfitting"
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    mode='min'
)

print("\nTraining the Neural Network")

"train the neural network model"
# Fitting the Neural Network to the Training set
history = model.fit(
    x=X_train,
    y=Y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_valid, Y_valid),
    callbacks=[early_stopping]
)

print("\nTraining Complete")
"end neural network model"

# Getting the predicted stock movements
"make predictions on validation set"
Y_pred_proba = model.predict(X_valid)
Y_pred_class = (Y_pred_proba > 0.5).astype(int)

"get corresponding dates for validation predictions"
val_dates = df_clean['date'].loc[Y_valid.index].values

"create results dataframe for visualization"
results_df = pd.DataFrame({
    'Date': val_dates,
    'Actual': Y_valid.to_numpy().flatten(),
    'Predicted_Prob': Y_pred_proba.flatten(),
    'Predicted_Class': Y_pred_class.flatten()
})

"prepare data for plotting"
results_df['Date'] = pd.to_datetime(results_df['Date'])
results_df = results_df.sort_values(by='Date').reset_index(drop=True)

# Visualize the results
"visualize predictions vs actual movements"
plt.figure(figsize=(15, 6))

plt.plot(results_df['Date'], results_df['Actual'], label='Actual (1=Up, 0=Down)', marker='o', linestyle='None', alpha=0.6, color='blue')

plt.plot(results_df['Date'], results_df['Predicted_Prob'], label='Predicted Probability (Up)', color='red', alpha=0.7)

plt.title('Neural Network Predictions vs. Actual Movements')
plt.xlabel('Date')
plt.ylabel('Value (0, 1 or Probability)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()