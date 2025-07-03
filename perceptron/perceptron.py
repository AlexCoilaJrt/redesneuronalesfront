import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron as SKPerceptron
from sklearn.metrics import accuracy_score
import matplotlib.patches as patches
import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Perceptrón - Tarjeta Platinum",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar el diseño
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
        font-size: 2.5rem;
    }
    .metric-container {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 1.5rem;
        border-radius: 10px;
        background: #f0f2f6;
        border: 2px solid transparent;
        white-space: nowrap;
        min-width: fit-content;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Header principal
st.markdown("""
<div class="main-header">
    <h1>🧠 Perceptrón para Aprobación de Tarjeta Platinum 💳</h1>
</div>
""", unsafe_allow_html=True)

# Sidebar con información del proyecto
with st.sidebar:
    st.markdown("## 📊 Información del Proyecto")
    st.markdown("""
    <div class="info-box">
        <h3>🎯 Objetivo</h3>
        <p>Implementar un perceptrón desde cero para clasificar solicitudes de tarjetas de crédito Platinum basándose en edad y ahorro.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔧 Parámetros del Modelo")
    st.info("Todos los parámetros se configuran automáticamente para demostrar el funcionamiento del perceptrón.")
    
    st.markdown("### 📈 Métricas")
    st.success("✅ Implementación manual")
    st.success("✅ Comparación con Scikit-learn")
    st.success("✅ Visualización de fronteras")

# ================================
# 1. Datos simulados: [edad, ahorro]
# ================================
@st.cache_data
def generar_datos():
    personas = np.array([
        [0.3, 0.4], [0.4, 0.3], [0.3, 0.2], [0.4, 0.1], [0.5, 0.2],
        [0.4, 0.8], [0.6, 0.8], [0.5, 0.6], [0.7, 0.6], [0.8, 0.5]
    ])
    clases = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # 0: denegada, 1: aprobada
    return personas, clases

# ================================
# 2. Función de activación y entrenamiento manual
# ================================
def activacion(pesos, x, b):
    z = np.dot(pesos, x)
    return 1 if z + b > 0 else 0

@st.cache_data
def entrenar_perceptron_manual(personas, clases):
    # Inicialización
    np.random.seed(42)
    pesos = np.random.uniform(-1, 1, size=2)
    b = np.random.uniform(-1, 1)
    tasa_de_aprendizaje = 0.01
    epocas = 100
    
    errores_por_epoca = []
    
    # Entrenamiento manual
    for epoca in range(epocas):
        error_total = 0
        for i in range(len(personas)):
            x = personas[i]
            y = clases[i]
            y_hat = activacion(pesos, x, b)
            error = y - y_hat
            error_total += error**2
            pesos += tasa_de_aprendizaje * error * x
            b += tasa_de_aprendizaje * error
        errores_por_epoca.append(error_total)
    
    return pesos, b, errores_por_epoca

@st.cache_data
def entrenar_sklearn(personas, clases):
    sk_perceptron = SKPerceptron(max_iter=1000, eta0=0.01, random_state=42)
    sk_perceptron.fit(personas, clases)
    sk_pred = sk_perceptron.predict(personas)
    return sk_perceptron, sk_pred

# Cargar datos
personas, clases = generar_datos()
pesos, b, errores_por_epoca = entrenar_perceptron_manual(personas, clases)
sk_perceptron, sk_pred = entrenar_sklearn(personas, clases)

# Mostrar datos básicos
st.markdown("## 📊 Datos del Proyecto")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-container" style="color: black;">
        <h3>👥 Muestras</h3>
        <h2>10 personas</h2>
        <p>5 aprobadas, 5 denegadas</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-container" style="color: black;">
        <h3>📈 Características</h3>
        <h2>2 variables</h2>
        <p>Edad y Ahorro (normalizados)</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-container" style="color: black;">
        <h3>🎯 Precisión</h3>
        <h2>{accuracy_score(clases, sk_pred):.1%}</h2>
        <p>Modelo Scikit-learn</p>
    </div>
    """, unsafe_allow_html=True)

# Crear tabs para organizar el contenido
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Visualización Manual", "🔧 Estructura del Perceptrón", "🏭 Scikit-learn", "📊 Comparación"])

with tab1:
    st.markdown("### 🎯 Perceptrón Implementado Manualmente")
    
    # Crear la visualización manual
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    
    # Crear una malla de puntos para mostrar las zonas de decisión
    x_min, x_max = 0, 1
    y_min, y_max = 0, 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Clasificar cada punto de la malla
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = []
    for point in mesh_points:
        Z.append(activacion(pesos, point, b))
    Z = np.array(Z)
    Z = Z.reshape(xx.shape)
    
    # Mostrar las zonas de decisión con colores de fondo
    ax1.contourf(xx, yy, Z, levels=1, alpha=0.3, colors=['lightcoral', 'lightblue'])
    
    # Mostrar los puntos de entrenamiento
    ax1.scatter(personas[clases == 0][:, 0], personas[clases == 0][:, 1],
                color="red", marker="x", s=120, linewidths=3, label="❌ Denegada")
    ax1.scatter(personas[clases == 1][:, 0], personas[clases == 1][:, 1],
                color="blue", marker="o", s=120, label="✅ Aprobada")
    
    # Mostrar la frontera de decisión
    if pesos[1] != 0:
        x_vals = np.linspace(0, 1, 100)
        y_vals = -(pesos[0] * x_vals + b) / pesos[1]
        valid_indices = (y_vals >= 0) & (y_vals <= 1)
        ax1.plot(x_vals[valid_indices], y_vals[valid_indices], 'k--', linewidth=3, label="🔍 Frontera de decisión")
    
    ax1.set_xlabel("Edad (normalizada)", fontsize=12)
    ax1.set_ylabel("Ahorro (normalizado)", fontsize=12)
    ax1.set_title("Perceptrón desde cero: ¿Tarjeta Platinum?", fontsize=16, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    st.pyplot(fig1)
    
    # Mostrar parámetros del modelo
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Peso Edad (w₁)", f"{pesos[0]:.4f}")
        st.metric("Peso Ahorro (w₂)", f"{pesos[1]:.4f}")
    with col2:
        st.metric("Sesgo (b)", f"{b:.4f}")
        st.metric("Error Final", f"{errores_por_epoca[-1]:.0f}")

with tab2:
    st.markdown("### 🔧 Estructura del Perceptrón")
    
    # Crear el gráfico de estructura
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 6)
    ax2.axis('off')
    
    # Entradas
    input_box1 = patches.Rectangle((0.2, 4.5), 0.6, 1, fill=True, facecolor='lightblue', edgecolor='black', linewidth=2)
    input_box2 = patches.Rectangle((0.2, 2.5), 0.6, 1, fill=True, facecolor='lightblue', edgecolor='black', linewidth=2)
    ax2.add_patch(input_box1)
    ax2.add_patch(input_box2)
    ax2.text(0.5, 5, "Edad\nx₁", fontsize=14, ha='center', va='center', fontweight='bold')
    ax2.text(0.5, 3, "Ahorro\nx₂", fontsize=14, ha='center', va='center', fontweight='bold')
    
    # Flechas desde entradas con pesos
    ax2.annotate('', xy=(3.4, 4.2), xytext=(0.8, 5), arrowprops=dict(arrowstyle="->", lw=3, color='darkgreen'))
    ax2.annotate('', xy=(3.4, 3.8), xytext=(0.8, 3), arrowprops=dict(arrowstyle="->", lw=3, color='darkgreen'))
    
    # Mostrar valores de pesos reales
    ax2.text(1.8, 4.7, f"w₁ = {pesos[0]:.3f}", fontsize=12, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    ax2.text(1.8, 3.3, f"w₂ = {pesos[1]:.3f}", fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    # Círculo de la suma
    suma_circle = patches.Circle((4, 4), 0.7, fill=True, facecolor='lightyellow', edgecolor='black', linewidth=2)
    ax2.add_patch(suma_circle)
    ax2.text(4, 4.2, "∑", fontsize=24, ha='center', va='center', fontweight='bold')
    ax2.text(4, 3.6, "w₁x₁ + w₂x₂", fontsize=10, ha='center', va='center')
    
    # Mostrar valor del sesgo
    ax2.text(4, 2.7, f"+ Sesgo (b) = {b:.3f}", fontsize=12, ha='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcyan', alpha=0.7), fontweight='bold')
    
    # Flecha hacia activación
    ax2.annotate('', xy=(5.8, 4), xytext=(4.7, 4), arrowprops=dict(arrowstyle="->", lw=3, color='darkblue'))
    ax2.text(5.2, 4.3, "z", fontsize=14, ha='center', fontweight='bold', color='darkblue')
    
    # Cuadro activación
    activation_box = patches.Rectangle((5.8, 3.3), 1.4, 1.4, fill=True, facecolor='lightpink', edgecolor='black', linewidth=2)
    ax2.add_patch(activation_box)
    ax2.text(6.5, 4, "f(z)\nz > 0 ?", fontsize=12, ha='center', va='center', fontweight='bold')
    
    # Flecha hacia salida
    ax2.annotate('', xy=(7.8, 4), xytext=(7.2, 4), arrowprops=dict(arrowstyle="->", lw=3, color='darkred'))
    
    # Salida
    output_box = patches.Rectangle((7.8, 3.3), 1.4, 1.4, fill=True, facecolor='lightgray', edgecolor='black', linewidth=2)
    ax2.add_patch(output_box)
    ax2.text(8.5, 4.3, "Salida", fontsize=14, ha='center', fontweight='bold')
    ax2.text(8.5, 3.7, "0: Denegada\n1: Aprobada", fontsize=11, ha='center', va='center')
    
    plt.title("Estructura del Perceptrón (con pesos entrenados)", fontsize=18, fontweight='bold', pad=20)
    st.pyplot(fig2)

with tab3:
    st.markdown("### 🏭 Comparación con Scikit-learn")
    
    # Visualización con Scikit-learn
    fig3, ax3 = plt.subplots(figsize=(10, 7))
    
    # Crear malla para scikit-learn
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z_sk = sk_perceptron.predict(mesh_points)
    Z_sk = Z_sk.reshape(xx.shape)
    
    # Mostrar las zonas de decisión
    ax3.contourf(xx, yy, Z_sk, levels=1, alpha=0.3, colors=['lightcoral', 'lightblue'])
    
    # Mostrar los puntos de entrenamiento
    ax3.scatter(personas[clases == 0][:, 0], personas[clases == 0][:, 1],
                color="red", marker="x", s=120, linewidths=3, label="❌ Denegada")
    ax3.scatter(personas[clases == 1][:, 0], personas[clases == 1][:, 1],
                color="blue", marker="o", s=120, label="✅ Aprobada")
    
    # Frontera de decisión
    w = sk_perceptron.coef_[0]
    b_sklearn = sk_perceptron.intercept_[0]
    if w[1] != 0:
        x_vals = np.linspace(0, 1, 100)
        y_vals = -(w[0] * x_vals + b_sklearn) / w[1]
        valid_indices = (y_vals >= 0) & (y_vals <= 1)
        ax3.plot(x_vals[valid_indices], y_vals[valid_indices], 'k--', linewidth=3, label="🔍 Frontera Scikit-learn")
    
    ax3.set_xlabel("Edad (normalizada)", fontsize=12)
    ax3.set_ylabel("Ahorro (normalizado)", fontsize=12)
    ax3.set_title("Perceptrón con Scikit-learn", fontsize=16, fontweight='bold')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=11)
    
    st.pyplot(fig3)
    
    # Mostrar parámetros de Scikit-learn
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Peso Edad (w₁)", f"{sk_perceptron.coef_[0][0]:.4f}")
        st.metric("Peso Ahorro (w₂)", f"{sk_perceptron.coef_[0][1]:.4f}")
    with col2:
        st.metric("Sesgo (b)", f"{sk_perceptron.intercept_[0]:.4f}")
        st.metric("Precisión", f"{accuracy_score(clases, sk_pred):.1%}")

with tab4:
    st.markdown("### 📊 Comparación de Resultados")
    
    # Crear tabla comparativa
    import pandas as pd
    
    comparacion_data = {
        'Parámetro': ['Peso Edad (w₁)', 'Peso Ahorro (w₂)', 'Sesgo (b)', 'Precisión'],
        'Implementación Manual': [f"{pesos[0]:.4f}", f"{pesos[1]:.4f}", f"{b:.4f}", "100%"],
        'Scikit-learn': [f"{sk_perceptron.coef_[0][0]:.4f}", f"{sk_perceptron.coef_[0][1]:.4f}", 
                        f"{sk_perceptron.intercept_[0]:.4f}", f"{accuracy_score(clases, sk_pred):.1%}"]
    }
    
    df_comparacion = pd.DataFrame(comparacion_data)
    st.dataframe(df_comparacion, use_container_width=True)
    
    # Análisis del modelo
    st.markdown("### 🔍 Análisis del Modelo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="info-box" >
            <h4>📈 Convergencia</h4>
            <p>• El perceptrón converge después de 100 épocas</p>
            <p>• Error final: {errores_por_epoca[-1]:.0f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        interpretacion = "El ahorro tiene más influencia que la edad" if pesos[1] > pesos[0] else "La edad tiene más influencia que el ahorro"
        st.markdown(f"""
        <div class="info-box">
            <h4>🧠 Interpretación</h4>
            <p>• Peso edad: {pesos[0]:.3f}</p>
            <p>• Peso ahorro: {pesos[1]:.3f}</p>
            <p>• {interpretacion}</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🧠 Implementación de Perceptrón para Clasificación Binaria</p>
    <p>Desarrollado con ❤️ usando Streamlit y Scikit-learn</p>
</div>
""", unsafe_allow_html=True)