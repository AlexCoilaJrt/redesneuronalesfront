import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configuración de la página
st.set_page_config(
    page_title="🧠 MNIST Neural Network Trainer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar el diseño
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: #2c3e50;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(255,255,255,0.8);
    }
    
    .main-header p {
        color: #34495e;
        font-size: 1.2rem;
        margin: 0;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: #f8f9fa;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: #2c3e50;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: #2c3e50;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    .step-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Header principal
st.markdown("""
<div class="main-header">
    <h1>🧠 MNIST Neural Network Trainer</h1>
    <p>Entrenamiento de Red Neuronal para Reconocimiento de Dígitos Manuscritos</p>
</div>
""", unsafe_allow_html=True)

# Sidebar con controles
st.sidebar.markdown("## 🎛️ Controles de Entrenamiento")
epochs = st.sidebar.slider("📈 Número de Épocas", 5, 50, 10)
batch_size = st.sidebar.selectbox("📦 Tamaño del Batch", [32, 64, 128, 256], index=2)
validation_split = st.sidebar.slider("✅ División de Validación", 0.1, 0.3, 0.1)
dropout_rate = st.sidebar.slider("🎯 Tasa de Dropout", 0.1, 0.5, 0.2)

st.sidebar.markdown("---")
st.sidebar.markdown("## 📊 Información del Modelo")
st.sidebar.info("**Arquitectura:**\n- Entrada: 784 neuronas\n- Oculta: 128 neuronas (ReLU)\n- Salida: 10 neuronas (Softmax)")

# Función para mostrar información del sistema
def show_system_info():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card" style="color: black;">
            <h3>🔧 TensorFlow</h3>
            <p><strong>Versión:</strong> {}</p>
        </div>
        """.format(tf.__version__), unsafe_allow_html=True)

    
    with col2:
        st.markdown("""
        <div class="metric-card"style="color: black;">
            <h3>📱 Dispositivo</h3>
            <p><strong>GPU:</strong> {}</p>
        </div>
        """.format("Disponible" if tf.config.list_physical_devices('GPU') else "No disponible"), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card"style="color: black;">
            <h3>🎯 Dataset</h3>
            <p><strong>MNIST:</strong> 70,000 imágenes</p>
        </div>
        """, unsafe_allow_html=True)

# Función para visualizar la arquitectura
def visualizar_arquitectura_red():
    """Función para visualizar la arquitectura de la red neuronal"""
    fig = go.Figure()
    
    # Definir posiciones de las capas
    layers_info = [
        {"name": "Entrada", "neurons": 784, "x": 0, "color": "#FFE5B4", "activation": "Entrada"},
        {"name": "Oculta", "neurons": 128, "x": 1, "color": "#ADD8E6", "activation": "ReLU"},
        {"name": "Salida", "neurons": 10, "x": 2, "color": "#FFB6C1", "activation": "Softmax"}
    ]
    
    # Crear visualización con círculos para cada capa
    for i, layer in enumerate(layers_info):
        # Mostrar solo algunas neuronas para simplificar
        display_neurons = min(8, layer["neurons"])
        y_positions = np.linspace(0, 1, display_neurons)
        
        for j, y in enumerate(y_positions):
            fig.add_trace(go.Scatter(
                x=[layer["x"]],
                y=[y],
                mode='markers',
                marker=dict(
                    size=20,
                    color=layer["color"],
                    line=dict(width=2, color='black')
                ),
                showlegend=False,
                hovertemplate=f"<b>{layer['name']}</b><br>" +
                            f"Neuronas: {layer['neurons']}<br>" +
                            f"Activación: {layer['activation']}<extra></extra>"
            ))
        
        # Añadir etiquetas
        fig.add_annotation(
            x=layer["x"],
            y=-0.2,
            text=f"<b>{layer['name']}</b><br>{layer['neurons']} neuronas<br>{layer['activation']}",
            showarrow=False,
            font=dict(size=12),
            bgcolor=layer["color"],
            bordercolor="black",
            borderwidth=1
        )
    
    # Configurar layout
    fig.update_layout(
        title="🏗️ Arquitectura de la Red Neuronal",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

# Función para crear gráficos de entrenamiento
def create_training_plots(history):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('📉 Pérdida del Modelo', '📈 Precisión del Modelo'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    epochs_range = range(1, len(history.history['loss']) + 1)
    
    # Gráfico de pérdida
    fig.add_trace(
        go.Scatter(x=list(epochs_range), y=history.history['loss'], 
                  name='Entrenamiento', line=dict(color='#667eea', width=3)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=list(epochs_range), y=history.history['val_loss'], 
                  name='Validación', line=dict(color='#f093fb', width=3)),
        row=1, col=1
    )
    
    # Gráfico de precisión
    fig.add_trace(
        go.Scatter(x=list(epochs_range), y=history.history['accuracy'], 
                  name='Entrenamiento', line=dict(color='#11998e', width=3),
                  showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=list(epochs_range), y=history.history['val_accuracy'], 
                  name='Validación', line=dict(color='#38ef7d', width=3),
                  showlegend=False),
        row=1, col=2
    )
    
    fig.update_layout(height=400, title_text="📊 Progreso del Entrenamiento")
    fig.update_xaxes(title_text="Época")
    fig.update_yaxes(title_text="Pérdida", row=1, col=1)
    fig.update_yaxes(title_text="Precisión", row=1, col=2)
    
    return fig

# Función principal
def main():
    # Mostrar información del sistema
    show_system_info()
    
    # Paso 1: Cargar datos
    with st.expander("📁 Paso 1: Cargar Dataset MNIST", expanded=True):
        st.markdown("""
        <div class="step-container" style="color: black;">
            <h4>🔄 Cargando dataset MNIST...</h4>
            <p>El dataset MNIST contiene 70,000 imágenes de dígitos manuscritos (0-9) en escala de grises de 28x28 píxeles.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Cargar Dataset"):
            with st.spinner("Cargando datos..."):
                # Cargar el dataset MNIST
                (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
                
                # Guardar en session state
                st.session_state.x_train = x_train
                st.session_state.y_train = y_train
                st.session_state.x_test = x_test
                st.session_state.y_test = y_test
                
                st.success("✅ Dataset cargado exitosamente!")
                
                # Mostrar estadísticas
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🎯 Datos de Entrenamiento", f"{x_train.shape[0]:,}")
                with col2:
                    st.metric("🧪 Datos de Prueba", f"{x_test.shape[0]:,}")
                with col3:
                    st.metric("📏 Dimensiones", f"{x_train.shape[1]}x{x_train.shape[2]}")
                with col4:
                    st.metric("🔢 Clases", len(np.unique(y_train)))
                
                # Visualizar algunas imágenes
                st.subheader("🖼️ Ejemplos del Dataset")
                fig, axes = plt.subplots(2, 6, figsize=(12, 4))
                for i in range(12):
                    row = i // 6
                    col = i % 6
                    axes[row, col].imshow(x_train[i], cmap='gray')
                    axes[row, col].set_title(f'Etiqueta: {y_train[i]}')
                    axes[row, col].axis('off')
                plt.tight_layout()
                st.pyplot(fig)
    
    # Paso 2: Arquitectura del modelo
    if 'x_train' in st.session_state:
        with st.expander("🏗️ Paso 2: Arquitectura del Modelo", expanded=True):
            st.markdown("""
            <div class="info-box">
                <h4>🧠 Arquitectura de la Red Neuronal</h4>
                <p>Esta red neuronal utiliza una arquitectura simple pero efectiva para el reconocimiento de dígitos:</p>
                <ul>
                    <li><strong>Capa de Entrada:</strong> 784 neuronas (28x28 píxeles aplanados)</li>
                    <li><strong>Capa Oculta:</strong> 128 neuronas con activación ReLU</li>
                    <li><strong>Dropout:</strong> 20% para prevenir sobreajuste</li>
                    <li><strong>Capa de Salida:</strong> 10 neuronas con activación Softmax</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Mostrar arquitectura
            st.plotly_chart(visualizar_arquitectura_red(), use_container_width=True)
            
            # Explicación del funcionamiento
            st.markdown("""
            <div class="step-container" style="color: black;">
                <h4>🔍 Funcionamiento de la Red</h4>
                <p><strong>1. Entrada:</strong> Cada imagen se convierte en un vector de 784 números (píxeles)</p>
                <p><strong>2. Capa Oculta:</strong> 128 neuronas procesan los datos con activación ReLU</p>
                <p><strong>3. Dropout:</strong> Desactiva aleatoriamente 20% de neuronas para evitar memorización</p>
                <p><strong>4. Salida:</strong> 10 neuronas representan la probabilidad de cada dígito (0-9)</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Paso 3: Entrenamiento
    if 'x_train' in st.session_state:
        with st.expander("🎯 Paso 3: Entrenamiento del Modelo", expanded=True):
            st.markdown("""
            <div class="step-container" style="color: black;">
                <h4>⚙️ Configuración del Entrenamiento</h4>
                <p>Los parámetros seleccionados en la barra lateral determinarán el comportamiento del entrenamiento.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Mostrar configuración actual
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📈 Épocas", epochs)
            with col2:
                st.metric("📦 Batch Size", batch_size)
            with col3:
                st.metric("✅ Validación", f"{validation_split*100:.0f}%")
            with col4:
                st.metric("🎯 Dropout", f"{dropout_rate*100:.0f}%")
            
            if st.button("🚀 Iniciar Entrenamiento"):
                # Preparar datos
                x_train = st.session_state.x_train.astype('float32') / 255.0
                x_test = st.session_state.x_test.astype('float32') / 255.0
                x_train_flat = x_train.reshape(-1, 28 * 28)
                x_test_flat = x_test.reshape(-1, 28 * 28)
                
                # Crear modelo
                model = models.Sequential([
                    layers.Dense(128, activation='relu', input_shape=(784,), name='capa_oculta'),
                    layers.Dropout(dropout_rate, name='dropout'),
                    layers.Dense(10, activation='softmax', name='capa_salida')
                ])
                
                model.compile(
                    optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                # Mostrar resumen del modelo
                st.subheader("📋 Resumen del Modelo")
                model_summary = []
                model.summary(print_fn=lambda x: model_summary.append(x))
                st.text('\n'.join(model_summary))
                
                # Entrenar con barra de progreso
                st.subheader("🏃‍♂️ Entrenamiento en Progreso")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Crear contenedores para métricas en tiempo real
                col1, col2 = st.columns(2)
                with col1:
                    loss_chart = st.empty()
                with col2:
                    acc_chart = st.empty()
                
                # Entrenar el modelo
                start_time = time.time()
                
                history = model.fit(
                    x_train_flat, st.session_state.y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split,
                    verbose=0
                )
                
                training_time = time.time() - start_time
                
                # Guardar modelo y resultados
                st.session_state.model = model
                st.session_state.history = history
                st.session_state.x_test_flat = x_test_flat
                
                # Actualizar barra de progreso
                progress_bar.progress(1.0)
                status_text.success(f"✅ Entrenamiento completado en {training_time:.2f} segundos")
                
                # Mostrar gráficos de entrenamiento
                st.subheader("📊 Progreso del Entrenamiento")
                st.plotly_chart(create_training_plots(history), use_container_width=True)
                
                # Evaluar modelo
                st.subheader("🎯 Evaluación del Modelo")
                test_loss, test_acc = model.evaluate(x_test_flat, st.session_state.y_test, verbose=0)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🎯 Precisión Test", f"{test_acc:.4f}", f"{test_acc*100:.2f}%")
                with col2:
                    st.metric("📉 Pérdida Test", f"{test_loss:.4f}")
                with col3:
                    st.metric("⏱️ Tiempo", f"{training_time:.2f}s")
                
                # Mensaje de éxito basado en precisión
                if test_acc > 0.97:
                    st.markdown("""
                    <div class="success-box">
                        <h4>🎉 ¡Excelente Rendimiento!</h4>
                        <p>El modelo ha alcanzado una precisión superior al 97%. ¡Fantástico trabajo!</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif test_acc > 0.95:
                    st.markdown("""
                    <div class="info-box">
                        <h4>👍 Buen Rendimiento</h4>
                        <p>El modelo tiene un buen rendimiento con más del 95% de precisión.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="warning-box">
                        <h4>⚠️ Rendimiento Mejorable</h4>
                        <p>El modelo podría mejorarse. Considera aumentar las épocas o ajustar parámetros.</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Paso 4: Predicciones
    if 'model' in st.session_state:
        with st.expander("🔮 Paso 4: Predicciones del Modelo", expanded=True):
            st.subheader("🎯 Predicciones en Imágenes de Prueba")
            
            # Hacer predicciones
            predictions = st.session_state.model.predict(st.session_state.x_test_flat[:10])
            predicted_classes = np.argmax(predictions, axis=1)
            
            # Mostrar predicciones
            cols = st.columns(5)
            for i in range(10):
                with cols[i % 5]:
                    fig, ax = plt.subplots(figsize=(2, 2))
                    ax.imshow(st.session_state.x_test[i], cmap='gray')
                    ax.set_title(f'Real: {st.session_state.y_test[i]}\nPred: {predicted_classes[i]}')
                    ax.axis('off')
                    st.pyplot(fig)
                    
                    # Mostrar confianza
                    confidence = np.max(predictions[i]) * 100
                    if st.session_state.y_test[i] == predicted_classes[i]:
                        st.success(f"✅ Correcto ({confidence:.1f}%)")
                    else:
                        st.error(f"❌ Incorrecto ({confidence:.1f}%)")
    
    # Paso 5: Análisis avanzado
    if 'model' in st.session_state:
        with st.expander("📊 Paso 5: Análisis Avanzado", expanded=True):
            st.subheader("🔍 Matriz de Confusión")
            
            # Calcular matriz de confusión
            all_predictions = st.session_state.model.predict(st.session_state.x_test_flat)
            all_predicted_classes = np.argmax(all_predictions, axis=1)
            cm = confusion_matrix(st.session_state.y_test, all_predicted_classes)
            
            # Crear heatmap
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Matriz de Confusión')
            ax.set_ylabel('Etiqueta Real')
            ax.set_xlabel('Predicción')
            st.pyplot(fig)
            
            # Reporte de clasificación
            st.subheader("📋 Reporte de Clasificación")
            report = classification_report(st.session_state.y_test, all_predicted_classes, output_dict=True)
            
            # Mostrar métricas por clase
            classes = [str(i) for i in range(10)]
            metrics_df = []
            for cls in classes:
                metrics_df.append({
                    'Dígito': cls,
                    'Precisión': report[cls]['precision'],
                    'Recall': report[cls]['recall'],
                    'F1-Score': report[cls]['f1-score'],
                    'Soporte': report[cls]['support']
                })
            
            import pandas as pd
            df = pd.DataFrame(metrics_df)
            st.dataframe(df, use_container_width=True)
            
            # Resumen final
            st.markdown("""
            <div class="success-box">
                <h4>🎯 Resumen Final</h4>
                <p><strong>✅ Proceso Completado:</strong> La red neuronal ha sido entrenada exitosamente</p>
                <p><strong>🧠 Aprendizaje:</strong> El modelo puede reconocer dígitos manuscritos con alta precisión</p>
                <p><strong>🔄 Funcionamiento:</strong> Entrada → Capa Oculta → Dropout → Salida → Predicción</p>
                <p><strong>📊 Análisis:</strong> La matriz de confusión muestra el rendimiento por cada dígito</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()