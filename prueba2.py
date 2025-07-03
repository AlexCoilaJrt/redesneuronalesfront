import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuración para gráficos
plt.style.use('default')
np.random.seed(42)

class RedNeuronalKaggle:
    def __init__(self, input_size, hidden_size, output_size, task_type='regression'):
        """
        Red neuronal adaptable para datasets de Kaggle
        
        Parámetros:
        - input_size: número de features de entrada
        - hidden_size: número de neuronas en la capa oculta
        - output_size: número de salidas (1 para regresión, n_clases para clasificación)
        - task_type: 'regression' o 'classification'
        """
        self.task_type = task_type
        
        # Inicialización de pesos y sesgos (Xavier/Glorot initialization)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2/input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2/hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        # Almacenar valores intermedios
        self.z1 = None  # Pre-activación capa oculta
        self.a1 = None  # Post-activación capa oculta (valores intermedios)
        self.z2 = None  # Pre-activación capa salida
        self.a2 = None  # Salida final
        
        # Historial de entrenamiento
        self.loss_history = []
        self.accuracy_history = [] if task_type == 'classification' else None
    
    def relu(self, z):
        """Función de activación ReLU"""
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """Derivada de ReLU"""
        return (z > 0).astype(float)
    
    def sigmoid(self, z):
        """Función de activación Sigmoid"""
        # Evitar overflow
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        """Derivada de Sigmoid"""
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def softmax(self, z):
        """Función de activación Softmax para clasificación multiclase"""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, X):
        """Forward pass - Propagación hacia adelante"""
        # Capa oculta
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)  # Valores intermedios con ReLU
        
        # Capa de salida
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        
        if self.task_type == 'regression':
            self.a2 = self.z2  # Sin activación para regresión
        elif self.task_type == 'binary_classification':
            self.a2 = self.sigmoid(self.z2)  # Sigmoid para clasificación binaria
        else:  # multiclass_classification
            self.a2 = self.softmax(self.z2)  # Softmax para clasificación multiclase
        
        return self.a2
    
    def compute_loss(self, y_true, y_pred):
        """Calcular pérdida según el tipo de tarea"""
        if self.task_type == 'regression':
            return np.mean((y_true - y_pred)**2)  # MSE
        elif self.task_type == 'binary_classification':
            # Binary cross-entropy
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:  # multiclass_classification
            # Categorical cross-entropy
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    def backward(self, X, y, learning_rate):
        """Backward pass - Retropropagación"""
        m = X.shape[0]
        
        # Gradientes capa de salida
        if self.task_type == 'regression':
            dz2 = self.a2 - y
        else:  # clasificación
            dz2 = self.a2 - y
        
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # Gradientes capa oculta
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Actualizar pesos y sesgos
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train(self, X, y, epochs, learning_rate, verbose=True):
        """Entrenar la red neuronal"""
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Calcular pérdida
            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)
            
            # Calcular accuracy si es clasificación
            if self.task_type != 'regression':
                if self.task_type == 'binary_classification':
                    y_pred_class = (y_pred > 0.5).astype(int)
                    accuracy = np.mean(y_pred_class == y)
                else:  # multiclass
                    y_pred_class = np.argmax(y_pred, axis=1)
                    y_true_class = np.argmax(y, axis=1)
                    accuracy = np.mean(y_pred_class == y_true_class)
                self.accuracy_history.append(accuracy)
            
            # Backward pass
            self.backward(X, y, learning_rate)
            
            # Mostrar progreso
            if verbose and epoch % 100 == 0:
                if self.task_type == 'regression':
                    print(f"Época {epoch}, Pérdida: {loss:.6f}")
                else:
                    print(f"Época {epoch}, Pérdida: {loss:.6f}, Accuracy: {accuracy:.4f}")
    
    def predict(self, X):
        """Hacer predicciones"""
        return self.forward(X)
    
    def mostrar_componentes(self, X_sample, feature_names=None, y_true=None):
        """Mostrar todos los componentes de la red para una muestra"""
        print("="*80)
        print("🧠 COMPONENTES DE LA RED NEURONAL - CPU METRICS DATASET")
        print("="*80)
        
        # Forward pass para obtener valores intermedios
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(1, -1)
        
        output = self.forward(X_sample)
        
        print(f"📥 INPUTS (CPU Metrics Features):")
        if feature_names:
            for i, (feature, value) in enumerate(zip(feature_names, X_sample[0])):
                print(f"   {feature}: {value:.4f}")
        else:
            print(f"   {X_sample[0]}")
        print()
        
        print(f"⚖️  PESOS W1 (Entrada → Capa Oculta):")
        print(f"   Forma: {self.W1.shape}")
        print(f"   Muestra (primeras 3x3): \n{self.W1[:3, :3]}")
        print()
        
        print(f"⚖️  PESOS W2 (Capa Oculta → Salida):")
        print(f"   Forma: {self.W2.shape}")
        print(f"   Valores: {self.W2.flatten()[:5]}")
        print()
        
        print(f"📊 SESGOS b1 (Capa Oculta):")
        print(f"   {self.b1.flatten()[:5]}")
        print()
        
        print(f"📊 SESGOS b2 (Capa Salida):")
        print(f"   {self.b2.flatten()}")
        print()
        
        print(f"🔄 VALORES PRE-ACTIVACIÓN z1:")
        print(f"   {self.z1.flatten()[:8]}")
        print()
        
        print(f"✨ FUNCIÓN DE ACTIVACIÓN ReLU:")
        print(f"   Input z1:  {self.z1.flatten()[:5]}")
        print(f"   Output a1: {self.a1.flatten()[:5]} (valores intermedios)")
        print()
        
        print(f"🔄 VALORES PRE-ACTIVACIÓN z2 (Salida):")
        print(f"   {self.z2.flatten()}")
        print()
        
        if self.task_type == 'regression':
            print(f"🎯 OUTPUT FINAL (Regresión):")
            print(f"   Predicción: {self.a2.flatten()[0]:.6f}")
            if y_true is not None:
                print(f"   Valor real: {y_true:.6f}")
                print(f"   Error: {abs(self.a2.flatten()[0] - y_true):.6f}")
        elif self.task_type == 'binary_classification':
            print(f"🎯 OUTPUT FINAL (Clasificación Binaria):")
            print(f"   Probabilidad: {self.a2.flatten()[0]:.6f}")
            print(f"   Predicción: {'Clase 1' if self.a2.flatten()[0] > 0.5 else 'Clase 0'}")
        else:
            print(f"🎯 OUTPUT FINAL (Clasificación Multiclase):")
            print(f"   Probabilidades: {self.a2.flatten()}")
            print(f"   Clase predicha: {np.argmax(self.a2.flatten())}")
        
        print("="*80)

class ProcesadorDatasetKaggle:
    """Clase para procesar automáticamente datasets de Kaggle"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = None
        self.target_name = None
        self.task_type = None
    
    def detectar_tipo_tarea(self, y):
        """Detectar automáticamente si es regresión o clasificación"""
        unique_values = len(np.unique(y))
        
        if y.dtype in ['float64', 'float32'] and unique_values > 10:
            return 'regression'
        elif unique_values == 2:
            return 'binary_classification'
        else:
            return 'multiclass_classification'
    
    def procesar_dataset(self, df, target_column, test_size=0.2):
        """Procesar dataset completo de Kaggle"""
        print(f"📊 Procesando dataset CPU Metrics...")
        print(f"   Forma original: {df.shape}")
        
        # Separar features y target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        self.feature_names = X.columns.tolist()
        self.target_name = target_column
        
        # Detectar tipo de tarea
        self.task_type = self.detectar_tipo_tarea(y)
        print(f"   Tipo de tarea detectado: {self.task_type}")
        
        # Procesar features categóricas
        X_processed = X.copy()
        for col in X.columns:
            if X[col].dtype == 'object':
                print(f"   Codificando columna categórica: {col}")
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X[col].astype(str))
                self.encoders[col] = le
        
        # Manejar valores faltantes
        print(f"   Valores faltantes encontrados: {X_processed.isnull().sum().sum()}")
        X_processed = X_processed.fillna(X_processed.mean())
        
        # Procesar target según tipo de tarea
        if self.task_type == 'regression':
            y_processed = y.values.reshape(-1, 1)
        elif self.task_type == 'binary_classification':
            if y.dtype == 'object':
                le_target = LabelEncoder()
                y_processed = le_target.fit_transform(y).reshape(-1, 1)
                self.encoders['target'] = le_target
            else:
                y_processed = y.values.reshape(-1, 1)
        else:  # multiclass
            if y.dtype == 'object':
                le_target = LabelEncoder()
                y_encoded = le_target.fit_transform(y)
                self.encoders['target'] = le_target
            else:
                y_encoded = y.values
            
            # One-hot encoding para multiclase
            n_classes = len(np.unique(y_encoded))
            y_processed = np.eye(n_classes)[y_encoded]
        
        # Normalizar features
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X_processed)
        self.scalers['X'] = scaler_X
        
        # Normalizar target solo para regresión
        if self.task_type == 'regression':
            scaler_y = StandardScaler()
            y_scaled = scaler_y.fit_transform(y_processed)
            self.scalers['y'] = scaler_y
        else:
            y_scaled = y_processed
        
        # Dividir en train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=test_size, random_state=42
        )
        
        print(f"   Features procesadas: {X_train.shape[1]}")
        print(f"   Muestras entrenamiento: {X_train.shape[0]}")
        print(f"   Muestras prueba: {X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test
    
    def evaluar_modelo(self, red, X_test, y_test, y_pred):
        """Evaluar modelo según tipo de tarea"""
        print(f"\n📊 EVALUACIÓN DEL MODELO ({self.task_type.upper()}):")
        print("-" * 50)
        
        if self.task_type == 'regression':
            if hasattr(self.scalers, 'y') and 'y' in self.scalers:
                y_test_orig = self.scalers['y'].inverse_transform(y_test)
                y_pred_orig = self.scalers['y'].inverse_transform(y_pred)
            else:
                y_test_orig = y_test
                y_pred_orig = y_pred
            
            mse = mean_squared_error(y_test_orig, y_pred_orig)
            r2 = r2_score(y_test_orig, y_pred_orig)
            
            print(f"MSE: {mse:.6f}")
            print(f"RMSE: {np.sqrt(mse):.6f}")
            print(f"R²: {r2:.6f}")
            
            return {'mse': mse, 'rmse': np.sqrt(mse), 'r2': r2}
            
        else:  # clasificación
            if self.task_type == 'binary_classification':
                y_pred_class = (y_pred > 0.5).astype(int)
                y_test_class = y_test.astype(int)
            else:  # multiclass
                y_pred_class = np.argmax(y_pred, axis=1)
                y_test_class = np.argmax(y_test, axis=1)
            
            accuracy = accuracy_score(y_test_class, y_pred_class)
            print(f"Accuracy: {accuracy:.6f}")
            print(f"\nReporte de clasificación:")
            print(classification_report(y_test_class, y_pred_class))
            
            return {'accuracy': accuracy}

def cargar_dataset_cpu_metrics():
    """Cargar y explorar el dataset unclean_cpu_metrics.csv"""
    try:
        print("📂 Cargando dataset unclean_cpu_metrics.csv...")
        df = pd.read_csv('unclean_cpu_metrics.csv')
        print(f"✅ Dataset cargado exitosamente!")
        return df
    except FileNotFoundError:
        print("❌ Error: No se encontró el archivo 'unclean_cpu_metrics.csv'")
        print("   Asegúrate de que el archivo esté en el mismo directorio que este script")
        return None
    except Exception as e:
        print(f"❌ Error al cargar el dataset: {e}")
        return None

def analizar_dataset(df):
    """Analizar la estructura del dataset para determinar automáticamente la columna objetivo"""
    print(f"\n🔍 ANÁLISIS AUTOMÁTICO DEL DATASET:")
    print(f"Forma: {df.shape}")
    print(f"Columnas disponibles: {list(df.columns)}")
    print(f"\nTipos de datos:")
    print(df.dtypes)
    
    print(f"\nValores faltantes por columna:")
    missing_values = df.isnull().sum()
    for col, missing in missing_values.items():
        if missing > 0:
            print(f"   {col}: {missing} ({missing/len(df)*100:.1f}%)")
    
    print(f"\nEstadísticas descriptivas:")
    print(df.describe())
    
    # Sugerir posibles columnas objetivo basándose en nombres comunes
    possible_targets = []
    target_keywords = ['cpu_usage', 'usage', 'utilization', 'load', 'performance', 'score', 'target', 'label', 'class']
    
    for col in df.columns:
        col_lower = col.lower()
        for keyword in target_keywords:
            if keyword in col_lower:
                possible_targets.append(col)
                break
    
    print(f"\n🎯 Posibles columnas objetivo detectadas:")
    if possible_targets:
        for i, target in enumerate(possible_targets, 1):
            print(f"   {i}. {target}")
        return possible_targets[0]  # Retornar la primera sugerencia
    else:
        print("   No se detectaron automáticamente. Columnas disponibles:")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i}. {col}")
        return df.columns[-1]  # Retornar la última columna por defecto

def main():
    """Función principal para procesar el dataset unclean_cpu_metrics.csv"""
    
    print("🚀 RED NEURONAL PARA CPU METRICS DATASET")
    print("="*80)
    
    # Cargar el dataset local
    df = cargar_dataset_cpu_metrics()
    if df is None:
        print("⚠️  No se pudo cargar el dataset. Saliendo...")
        return
    
    # Analizar automáticamente el dataset
    suggested_target = analizar_dataset(df)
    
    # Permitir al usuario seleccionar la columna objetivo
    print(f"\n🎯 SELECCIÓN DE COLUMNA OBJETIVO:")
    print(f"Sugerencia automática: '{suggested_target}'")
    
    user_input = input(f"\n¿Usar '{suggested_target}' como columna objetivo? (s/n) o escribe el nombre de otra columna: ").strip()
    
    if user_input.lower() in ['', 's', 'si', 'yes', 'y']:
        target_column = suggested_target
    elif user_input.lower() in ['n', 'no']:
        print("Columnas disponibles:")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i}. {col}")
        col_index = int(input("Selecciona el número de la columna objetivo: ")) - 1
        target_column = df.columns[col_index]
    else:
        if user_input in df.columns:
            target_column = user_input
        else:
            print(f"❌ Columna '{user_input}' no encontrada. Usando sugerencia automática.")
            target_column = suggested_target
    
    print(f"✅ Columna objetivo seleccionada: '{target_column}'")
    
    # Mostrar información detallada del dataset
    print(f"\n📋 INFORMACIÓN DETALLADA DEL DATASET:")
    print(f"Muestra de datos:")
    print(df.head())
    
    # Procesar dataset
    procesador = ProcesadorDatasetKaggle()
    X_train, X_test, y_train, y_test = procesador.procesar_dataset(df, target_column)
    
    # Crear red neuronal adaptativa
    input_size = X_train.shape[1]
    hidden_size = max(10, min(input_size * 2, 50))  # Tamaño adaptativo con límite
    
    if procesador.task_type == 'regression':
        output_size = 1
    elif procesador.task_type == 'binary_classification':
        output_size = 1
    else:  # multiclass
        output_size = y_train.shape[1]
    
    print(f"\n🧠 CONFIGURACIÓN DE LA RED NEURONAL:")
    print(f"   Input size: {input_size} (features)")
    print(f"   Hidden size: {hidden_size} (neuronas ocultas)")
    print(f"   Output size: {output_size} (salidas)")
    print(f"   Tipo de tarea: {procesador.task_type}")
    
    red = RedNeuronalKaggle(input_size, hidden_size, output_size, procesador.task_type)
    
    # Entrenar modelo con configuración adaptativa
    epochs = 1500 if procesador.task_type == 'regression' else 1000
    learning_rate = 0.01 if input_size < 20 else 0.005
    
    print(f"\n🏋️‍♂️ ENTRENANDO RED NEURONAL...")
    print(f"   Épocas: {epochs}")
    print(f"   Learning rate: {learning_rate}")
    
    red.train(X_train, y_train, epochs=epochs, learning_rate=learning_rate)
    
    # Mostrar componentes detallados de la red
    print(f"\n" + "="*80)
    y_true_sample = None
    if procesador.task_type == 'regression':
        if 'y' in procesador.scalers:
            y_true_sample = procesador.scalers['y'].inverse_transform(y_test[:1])[0][0]
        else:
            y_true_sample = y_test[0][0]
    
    red.mostrar_componentes(X_test[0], procesador.feature_names, y_true_sample)
    
    # Hacer predicciones
    y_pred_train = red.predict(X_train)
    y_pred_test = red.predict(X_test)
    
    # Evaluar modelo
    resultados = procesador.evaluar_modelo(red, X_test, y_test, y_pred_test)
    
    # Crear visualizaciones comprehensivas
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Curva de pérdida
    axes[0, 0].plot(red.loss_history, 'b-', linewidth=2)
    axes[0, 0].set_title('Curva de Pérdida Durante Entrenamiento', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Pérdida')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Accuracy o Predicciones vs Reales
    if procesador.task_type != 'regression' and red.accuracy_history:
        axes[0, 1].plot(red.accuracy_history, 'g-', linewidth=2)
        axes[0, 1].set_title('Accuracy Durante Entrenamiento', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Época')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
    else:
        # Para regresión: Predicciones vs Reales
        if procesador.task_type == 'regression':
            if 'y' in procesador.scalers:
                y_test_plot = procesador.scalers['y'].inverse_transform(y_test)
                y_pred_plot = procesador.scalers['y'].inverse_transform(y_pred_test)
            else:
                y_test_plot = y_test
                y_pred_plot = y_pred_test
            
            axes[0, 1].scatter(y_test_plot, y_pred_plot, alpha=0.6, color='blue', s=30)
            min_val = min(y_test_plot.min(), y_pred_plot.min())
            max_val = max(y_test_plot.max(), y_pred_plot.max())
            axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            axes[0, 1].set_title('Predicciones vs Valores Reales', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('Valores Reales')
            axes[0, 1].set_ylabel('Predicciones')
            axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Importancia de features
    feature_importance = np.abs(red.W1).mean(axis=1)
    top_features_idx = np.argsort(feature_importance)[-8:]  # Top 8 features
    
    axes[0, 2].barh(range(len(top_features_idx)), feature_importance[top_features_idx], color='skyblue')
    axes[0, 2].set_yticks(range(len(top_features_idx)))
    if procesador.feature_names:
        feature_labels = [procesador.feature_names[i] for i in top_features_idx]
        axes[0, 2].set_yticklabels(feature_labels, fontsize=9)
    axes[0, 2].set_title('Top Features por Importancia', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Importancia Promedio')
    
    # 4. Matriz de correlación
    correlation_matrix = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, 
                ax=axes[1, 0], square=True)
    axes[1, 0].set_title('Matriz de Correlación del Dataset', fontsize=12, fontweight='bold')
    
    # 5. Distribución del target
    if procesador.task_type == 'regression':
        axes[1, 1].hist(df[target_column], bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].set_title(f'Distribución de {target_column}', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel(target_column)
        axes[1, 1].set_ylabel('Frecuencia')
    else:
        df[target_column].value_counts().plot(kind='bar', ax=axes[1, 1], color='orange')
        axes[1, 1].set_title(f'Distribución de Clases - {target_column}', fontsize=12, fontweight='bold')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 6. Residuos (para regresión) o Matriz de confusión (para clasificación)
    if procesador.task_type == 'regression':
        if 'y' in procesador.scalers:
            y_test_orig = procesador.scalers['y'].inverse_transform(y_test)
            y_pred_orig = procesador.scalers['y'].inverse_transform(y_pred_test)
        else:
            y_test_orig = y_test
            y_pred_orig = y_pred_test
        
        residuos = y_test_orig.flatten() - y_pred_orig.flatten()
        axes[1, 2].scatter(y_pred_orig, residuos, alpha=0.6, color='red', s=30)
        axes[1, 2].axhline(y=0, color='black', linestyle='--')
        axes[1, 2].set_title('Gráfico de Residuos', fontsize=12, fontweight='bold')
        axes[1, 2].set_xlabel('Predicciones')
        axes[1, 2].set_ylabel('Residuos')
        axes[1, 2].grid(True, alpha=0.3)
    else:
        # Matriz de confusión simplificada para clasificación
        if procesador.task_type == 'binary_classification':
            y_pred_class = (y_pred_test > 0.5).astype(int)
            y_test_class = y_test.astype(int)
        else:  # multiclass
            y_pred_class = np.argmax(y_pred_test, axis=1)
            y_test_class = np.argmax(y_test, axis=1)
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test_class, y_pred_class)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 2])
        axes[1, 2].set_title('Matriz de Confusión', fontsize=12, fontweight='bold')
        axes[1, 2].set_xlabel('Predicciones')
        axes[1, 2].set_ylabel('Valores Reales')
    
    plt.tight_layout()
    plt.suptitle(f'Análisis Completo - Dataset: {target_column}', fontsize=16, fontweight='bold', y=1.02)
    plt.show()
    
    # Mostrar resumen final
    print(f"\n✅ ANÁLISIS COMPLETO FINALIZADO!")
    print("="*80)
    print(f"📊 RESUMEN DEL MODELO:")
    print(f"   • Dataset: unclean_cpu_metrics.csv")
    print(f"   • Columna objetivo: {target_column}")
    print(f"   • Tipo de tarea: {procesador.task_type}")
    print(f"   • Features utilizadas: {len(procesador.feature_names)}")
    print(f"   • Muestras de entrenamiento: {X_train.shape[0]}")
    print(f"   • Muestras de prueba: {X_test.shape[0]}")
    
    if procesador.task_type == 'regression':
        print(f"   • MSE: {resultados['mse']:.6f}")
        print(f"   • RMSE: {resultados['rmse']:.6f}")
        print(f"   • R²: {resultados['r2']:.6f}")
    else:
        print(f"   • Accuracy: {resultados['accuracy']:.6f}")
    
    print(f"\n🎯 TOP 5 FEATURES MÁS IMPORTANTES:")
    top_5_idx = np.argsort(feature_importance)[-5:]
    for i, idx in enumerate(reversed(top_5_idx), 1):
        feature_name = procesador.feature_names[idx]
        importance = feature_importance[idx]
        print(f"   {i}. {feature_name}: {importance:.4f}")
    
    print(f"\n💡 CONSEJOS PARA MEJORAR EL MODELO:")
    if procesador.task_type == 'regression':
        if resultados['r2'] < 0.7:
            print("   • Considera agregar más features o usar feature engineering")
            print("   • Prueba con diferentes arquitecturas (más capas ocultas)")
            print("   • Ajusta el learning rate o número de épocas")
        else:
            print("   • ¡Excelente modelo! R² > 0.7 indica buen ajuste")
    else:
        if resultados['accuracy'] < 0.8:
            print("   • Considera balancear las clases si están desbalanceadas")
            print("   • Prueba con diferentes funciones de activación")
            print("   • Aumenta el tamaño de la capa oculta")
        else:
            print("   • ¡Buen modelo! Accuracy > 0.8 es satisfactorio")
    
    print(f"\n🔧 PARÁMETROS UTILIZADOS:")
    print(f"   • Arquitectura: {input_size} → {hidden_size} → {output_size}")
    print(f"   • Épocas: {epochs}")
    print(f"   • Learning rate: {learning_rate}")
    print(f"   • Función de activación: ReLU (capa oculta)")
    if procesador.task_type == 'regression':
        print(f"   • Función de salida: Lineal")
    elif procesador.task_type == 'binary_classification':
        print(f"   • Función de salida: Sigmoid")
    else:
        print(f"   • Función de salida: Softmax")
    
    print("="*80)

if __name__ == "__main__":
    main()