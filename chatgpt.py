import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt

# 1) Cargar y preprocesar dataset

def load_and_preprocess():
    candidates = [
        os.path.expanduser('~/Descargas/unclean_cpu_metrics.csv'),
        os.path.expanduser('~/Downloads/unclean_cpu_metrics.csv')
    ]
    for path in candidates:
        if os.path.isfile(path):
            print(f"Cargando dataset desde: {path}")
            df = pd.read_csv(path)
            break
    else:
        raise FileNotFoundError("No se encontró 'unclean_cpu_metrics.csv' en Descargas o Downloads.")

    # Selección automática de target (última columna)
    target = df.columns[-1]
    print(f"Columna objetivo: '{target}'")

    # Imputar NaN en target con la media
    df[target] = pd.to_numeric(df[target], errors='coerce')
    df[target].fillna(df[target].mean(), inplace=True)

    # Extraer features y target
    X_df = df.drop(columns=[target]).copy()
    y = df[target].values.reshape(-1, 1)

    # Procesar Timestamp (si existe)
    if 'Timestamp' in X_df.columns:
        ts = pd.to_datetime(X_df['Timestamp'], errors='coerce')
        X_df['hour'] = ts.dt.hour.fillna(0).astype(int)
        X_df['dayofweek'] = ts.dt.dayofweek.fillna(0).astype(int)
        X_df.drop(columns=['Timestamp'], inplace=True)
        print("Extraídas variables de fecha: 'hour', 'dayofweek'.")

    # Convertir todo a numérico e imputar con mediana
    for col in X_df.columns:
        X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
    medians = X_df.median()
    X_df = X_df.fillna(medians)
    print(f"Imputadas columnas con mediana donde fue necesario.")

    feature_names = X_df.columns.tolist()
    X = X_df.values
    return X, y, feature_names

# 2) División y normalización
def split_and_scale(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)
    return (scaler_X.transform(X_train), scaler_X.transform(X_test),
            scaler_y.transform(y_train), scaler_y.transform(y_test),
            scaler_X, scaler_y)

# 3) Crear modelo con Keras: 3 capas ocultas + dropout
def build_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(input_shape*4, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(input_shape*2, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(input_shape, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss='mse', metrics=['mse'])
    return model

# --- Ejecución principal ---
if __name__ == '__main__':
    # Carga y preprocesamiento
    X, y, features = load_and_preprocess()
    X_train, X_test, y_train, y_test, scX, scY = split_and_scale(X, y)

    # Construcción del modelo
    model = build_model(X_train.shape[1])
    model.summary()

    # Callbacks
    es = callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    rlrop = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20)

    # Entrenamiento
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=1000,
        batch_size=16,
        callbacks=[es, rlrop],
        verbose=2
    )

    # Predicción y evaluación
    y_pred_s = model.predict(X_test)
    y_pred = scY.inverse_transform(y_pred_s)
    y_true = scY.inverse_transform(y_test)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\nMSE final: {mse:.3f}")
    print(f"R² final: {r2:.3f}")

    # Componentes de una muestra
    idx = 0
    x_s = X_test[idx:idx+1]
    y_s_true = y_true[idx][0]
    y_s_pred = y_pred[idx][0]
    print("\n== Componentes muestra ==")
    for name, val in zip(features, x_s.flatten()):
        print(f"{name}: {val:.3f}")
    print(f"Predicción: {y_s_pred:.3f}, Real: {y_s_true:.3f}, Error: {abs(y_s_pred-y_s_true):.3f}")

    # Visualizar curva de pérdida
    plt.figure(figsize=(8,5))
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Curva de pérdida')
    plt.xlabel('Época')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()
