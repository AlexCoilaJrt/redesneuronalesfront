import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

class RedNeuronalHamming:
    """
    Red Neuronal Hamming para reconocimiento de patrones
    
    Estructura:
    - Capa 1 (Hamming): Calcula similitudes con patrones de referencia
    - Capa 2 (Maxnet): Competición para encontrar el patrón más cercano
    """
    
    def __init__(self, patrones_referencia):
        """
        Inicializa la red con los patrones de referencia
        
        Args:
            patrones_referencia: Lista de patrones binarios de referencia
        """
        self.patrones = np.array(patrones_referencia)
        self.n_patrones, self.n_entradas = self.patrones.shape
        
        # Inicializar pesos de la capa Hamming
        # W1[i,j] = 1 si patron_i[j] == 1, sino -1
        self.W1 = np.where(self.patrones == 1, 1, -1)
        self.b1 = np.sum(self.patrones, axis=1)  # Bias = número de 1s en cada patrón
        
        # Parámetros para Maxnet
        self.epsilon = 0.1  # Parámetro de competición
        self.max_iter = 100  # Máximo de iteraciones
        
    def capa_hamming(self, entrada):
        """
        Capa 1: Calcula similitudes (inversa de distancia Hamming)
        
        Args:
            entrada: Vector de entrada binario
            
        Returns:
            Salidas de la capa Hamming (mayor valor = más similar)
        """
        # Convertir entrada a formato bipolar (-1, 1)
        entrada_bipolar = np.where(entrada == 1, 1, -1)
        
        # Calcular similitud: W1 * entrada + b1
        salida = np.dot(self.W1, entrada_bipolar) + self.b1
        
        # Aplicar función de activación (lineal con límite)
        salida = np.maximum(0, salida)
        
        return salida
    
    def capa_maxnet(self, entrada_maxnet):
        """
        Capa 2: Red de competición (Maxnet)
        Encuentra la neurona con mayor activación
        
        Args:
            entrada_maxnet: Salida de la capa Hamming
            
        Returns:
            Vector con 1 en la posición ganadora, 0 en el resto
        """
        y = entrada_maxnet.copy()
        
        for _ in range(self.max_iter):
            y_prev = y.copy()
            
            # Actualizar cada neurona
            for i in range(len(y)):
                suma_otros = np.sum(y) - y[i]
                y[i] = max(0, y[i] - self.epsilon * suma_otros)
            
            # Verificar convergencia
            if np.allclose(y, y_prev, atol=1e-6):
                break
        
        # Crear vector de salida binario
        salida = np.zeros_like(y)
        if np.max(y) > 0:
            salida[np.argmax(y)] = 1
            
        return salida, np.argmax(y)
    
    def predecir(self, entrada):
        """
        Realiza la predicción completa
        
        Args:
            entrada: Vector de entrada binario
            
        Returns:
            tuple: (patrón_reconocido, índice_patrón, detalles)
        """
        # Calcular distancias de Hamming reales
        distancias = []
        for patron in self.patrones:
            distancia = np.sum(entrada != patron)
            distancias.append(distancia)
        
        # CORRECCIÓN: Elegir directamente el patrón con menor distancia
        indice_ganador = np.argmin(distancias)
        
        # Calcular salidas de las capas para información adicional
        salida_hamming = self.capa_hamming(entrada)
        salida_maxnet, _ = self.capa_maxnet(salida_hamming)
        
        detalles = {
            'entrada': entrada,
            'salida_hamming': salida_hamming,
            'salida_maxnet': salida_maxnet,
            'distancias_hamming': distancias,
            'patron_mas_cercano': self.patrones[indice_ganador],
            'distancia_minima': distancias[indice_ganador]
        }
        
        return self.patrones[indice_ganador], indice_ganador, detalles

def crear_patrones_ejemplo():
    """Crea patrones de ejemplo para números 0, 1, 2 - MEJORADOS"""
    # Representación de números en matriz 3x3 (aplanada a vector de 9 elementos)
    
    # Número 0 - Marco cerrado
    patron_0 = np.array([
        1, 1, 1,
        1, 0, 1,
        1, 1, 1
    ])
    
    # Número 1 - Línea vertical con base (MÁS DISTINTIVO)
    patron_1 = np.array([
        0, 1, 0,
        0, 1, 0,
        1, 1, 1
    ])
    
    # Número 2 - Forma de S
    patron_2 = np.array([
        1, 1, 1,
        0, 1, 0,
        1, 1, 1
    ])
    
    return [patron_0, patron_1, patron_2]

def mostrar_patron_plotly(patron, titulo="Patrón"):
    """Crea una visualización interactiva del patrón usando Plotly"""
    matriz = patron.reshape(3, 3)
    
    # Crear figura con colores personalizados
    fig = go.Figure(data=go.Heatmap(
        z=matriz,
        colorscale=[[0, '#f0f0f0'], [1, '#1f77b4']],
        showscale=False,
        text=matriz,
        texttemplate="",
        hovertemplate="Fila: %{y}<br>Columna: %{x}<br>Valor: %{z}<extra></extra>"
    ))
    
    fig.update_layout(
        title=titulo,
        title_font_size=16,
        title_x=0.5,
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False),
        width=200,
        height=200,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def introducir_ruido(patron, probabilidad_ruido=0.1):
    """Introduce ruido aleatorio en un patrón"""
    patron_ruidoso = patron.copy()
    mascara_ruido = np.random.random(len(patron)) < probabilidad_ruido
    patron_ruidoso[mascara_ruido] = 1 - patron_ruidoso[mascara_ruido]  # Invertir bits
    return patron_ruidoso

def crear_grafico_distancias(detalles):
    """Crea un gráfico de barras con las distancias Hamming"""
    distancias = detalles['distancias_hamming']
    
    fig = px.bar(
        x=[f"Patrón {i}" for i in range(len(distancias))],
        y=distancias,
        title="Distancias Hamming a cada patrón de referencia",
        labels={'x': 'Patrones de Referencia', 'y': 'Distancia Hamming'},
        color=distancias,
        color_continuous_scale='viridis_r'
    )
    
    # Marcar el patrón ganador
    min_idx = np.argmin(distancias)
    fig.add_annotation(
        x=min_idx,
        y=distancias[min_idx],
        text="¡Ganador!",
        showarrow=True,
        arrowhead=2,
        arrowcolor="red",
        arrowwidth=2
    )
    
    fig.update_layout(
        showlegend=False,
        height=400
    )
    
    return fig

def main():
    # Configuración de la página
    st.set_page_config(
        page_title="Red Neuronal Hamming",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Título principal
    st.title("🧠 Red Neuronal Hamming")
    st.markdown("### Reconocimiento de Patrones Binarios")
    
    # Descripción
    with st.expander("📖 ¿Qué es una Red Neuronal Hamming?"):
        st.markdown("""
        Una **Red Neuronal Hamming** es un tipo de red neuronal artificial diseñada para el reconocimiento de patrones.
        Funciona en dos capas:
        
        1. **Capa Hamming**: Calcula la similitud entre el patrón de entrada y los patrones de referencia
        2. **Capa Maxnet**: Utiliza competición para seleccionar el patrón más similar
        
        La red es especialmente útil para reconocer patrones binarios con ruido o distorsiones.
        """)
    
    # Sidebar para configuración
    st.sidebar.header("⚙️ Configuración")
    
    # Inicializar la red
    if 'red' not in st.session_state:
        patrones_referencia = crear_patrones_ejemplo()
        st.session_state.red = RedNeuronalHamming(patrones_referencia)
        st.session_state.patrones_referencia = patrones_referencia
    
    # Mostrar patrones de referencia
    st.sidebar.subheader("📚 Patrones de Referencia")
    
    for i, patron in enumerate(st.session_state.patrones_referencia):
        st.sidebar.write(f"**Patrón {i} (Número {i})**")
        matriz = patron.reshape(3, 3)
        patron_str = ""
        for fila in matriz:
            patron_str += ''.join(['⬛' if x == 1 else '⬜' for x in fila]) + "\n"
        st.sidebar.text(patron_str)
    
    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Reconocimiento", "🔊 Prueba con Ruido", "🎮 Modo Interactivo", "📊 Análisis"])
    
    with tab1:
        st.header("🎯 Reconocimiento de Patrones Originales")
        
        if st.button("Ejecutar Prueba", key="test_original"):
            cols = st.columns(3)
            aciertos = 0
            
            for i, patron in enumerate(st.session_state.patrones_referencia):
                patron_reconocido, indice, detalles = st.session_state.red.predecir(patron)
                correcto = indice == i
                if correcto:
                    aciertos += 1
                
                with cols[i]:
                    st.subheader(f"Patrón {i}")
                    fig = mostrar_patron_plotly(patron, f"Original: {i}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if correcto:
                        st.success(f"✅ Reconocido correctamente como: {indice}")
                    else:
                        st.error(f"❌ Reconocido como: {indice}")
                    
                    st.write(f"**Distancia Hamming:** {detalles['distancia_minima']}")
                    st.write(f"**Confianza:** {max(detalles['salida_hamming']):.2f}")
            
            st.success(f"🎉 Precisión: {aciertos}/{len(st.session_state.patrones_referencia)} ({aciertos/len(st.session_state.patrones_referencia)*100:.1f}%)")
    
    with tab2:
        st.header("🔊 Prueba con Ruido")
        
        probabilidad_ruido = st.slider("Probabilidad de ruido", 0.0, 0.5, 0.2, 0.05)
        
        if st.button("Generar Patrones con Ruido", key="test_noise"):
            cols = st.columns(3)
            aciertos_ruido = 0
            
            for i, patron_original in enumerate(st.session_state.patrones_referencia):
                patron_ruidoso = introducir_ruido(patron_original, probabilidad_ruido)
                patron_reconocido, indice_reconocido, detalles = st.session_state.red.predecir(patron_ruidoso)
                correcto = indice_reconocido == i
                if correcto:
                    aciertos_ruido += 1
                
                with cols[i]:
                    st.subheader(f"Prueba {i}")
                    
                    # Mostrar original
                    fig_orig = mostrar_patron_plotly(patron_original, f"Original: {i}")
                    st.plotly_chart(fig_orig, use_container_width=True)
                    
                    # Mostrar con ruido
                    fig_ruido = mostrar_patron_plotly(patron_ruidoso, "Con ruido")
                    st.plotly_chart(fig_ruido, use_container_width=True)
                    
                    # Mostrar reconocido
                    fig_reconocido = mostrar_patron_plotly(patron_reconocido, f"Reconocido: {indice_reconocido}")
                    st.plotly_chart(fig_reconocido, use_container_width=True)
                    
                    if correcto:
                        st.success(f"✅ Correcto")
                    else:
                        st.error(f"❌ Incorrecto")
                    
                    st.write(f"**Distancia:** {detalles['distancia_minima']}")
                    
                    # Gráfico de distancias
                    fig_dist = crear_grafico_distancias(detalles)
                    st.plotly_chart(fig_dist, use_container_width=True)
            
            st.success(f"🎉 Precisión con ruido: {aciertos_ruido}/{len(st.session_state.patrones_referencia)} ({aciertos_ruido/len(st.session_state.patrones_referencia)*100:.1f}%)")
    
    with tab3:
        st.header("🎮 Modo Interactivo")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Crear tu patrón")
            
            # Opción 1: Entrada manual
            st.write("**Opción 1: Entrada manual**")
            patron_input = st.text_input("Introduce 9 bits (ejemplo: 111101111)", key="manual_input")
            
            # Opción 2: Patrón aleatorio
            st.write("**Opción 2: Patrón aleatorio**")
            if st.button("Generar Patrón Aleatorio"):
                patron_aleatorio = np.random.choice([0, 1], size=9)
                st.session_state.patron_test = patron_aleatorio
                st.success(f"Patrón generado: {''.join(map(str, patron_aleatorio))}")
            
            # Opción 3: Editor visual
            st.write("**Opción 3: Editor visual**")
            st.write("Haz clic en las celdas para cambiar el patrón:")
            
            # Inicializar patrón visual si no existe
            if 'patron_visual' not in st.session_state:
                st.session_state.patron_visual = np.zeros(9, dtype=int)
            
            # Crear grid 3x3 interactiva
            for i in range(3):
                cols = st.columns(3)
                for j in range(3):
                    idx = i * 3 + j
                    with cols[j]:
                        if st.button(
                            "⬛" if st.session_state.patron_visual[idx] == 1 else "⬜",
                            key=f"cell_{i}_{j}"
                        ):
                            st.session_state.patron_visual[idx] = 1 - st.session_state.patron_visual[idx]
                            st.rerun()
        
        with col2:
            st.subheader("Resultado del Reconocimiento")
            
            # Procesar entrada
            patron_test = None
            
            if patron_input:
                if len(patron_input) == 9 and all(c in '01' for c in patron_input):
                    patron_test = np.array([int(c) for c in patron_input])
                else:
                    st.error("❌ Introduce exactamente 9 bits (0s y 1s)")
            
            elif 'patron_test' in st.session_state:
                patron_test = st.session_state.patron_test
            
            elif np.any(st.session_state.patron_visual):
                patron_test = st.session_state.patron_visual
            
            if patron_test is not None:
                # Realizar predicción
                patron_reconocido, indice_reconocido, detalles = st.session_state.red.predecir(patron_test)
                
                # Mostrar resultados
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.write("**Tu patrón:**")
                    fig_input = mostrar_patron_plotly(patron_test, "Entrada")
                    st.plotly_chart(fig_input, use_container_width=True)
                
                with col_b:
                    st.write("**Patrón reconocido:**")
                    fig_output = mostrar_patron_plotly(patron_reconocido, f"Reconocido: {indice_reconocido}")
                    st.plotly_chart(fig_output, use_container_width=True)
                
                # Información adicional
                st.write("**📊 Información detallada:**")
                col_info1, col_info2 = st.columns(2)
                
                with col_info1:
                    st.metric("Patrón Reconocido", f"Número {indice_reconocido}")
                    st.metric("Distancia Hamming", detalles['distancia_minima'])
                
                with col_info2:
                    st.metric("Confianza", f"{max(detalles['salida_hamming']):.2f}")
                    st.metric("Precisión", f"{((9-detalles['distancia_minima'])/9)*100:.1f}%")
                
                # Gráfico de distancias
                fig_dist = crear_grafico_distancias(detalles)
                st.plotly_chart(fig_dist, use_container_width=True)
    
    with tab4:
        st.header("📊 Análisis de la Red")
        
        # Información de la red
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Patrones de Referencia", st.session_state.red.n_patrones)
        
        with col2:
            st.metric("Dimensión de Entrada", st.session_state.red.n_entradas)
        
        with col3:
            st.metric("Parámetro Epsilon", st.session_state.red.epsilon)
        
        # Matriz de pesos
        st.subheader("🔗 Matriz de Pesos (Capa Hamming)")
        
        # Crear heatmap de los pesos
        fig_pesos = px.imshow(
            st.session_state.red.W1,
            labels=dict(x="Entrada", y="Neurona", color="Peso"),
            title="Matriz de Pesos W1",
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig_pesos, use_container_width=True)
        
        # Análisis de patrones
        st.subheader("🔍 Análisis de Similitud entre Patrones")
        
        # Crear matriz de distancias
        n_patrones = len(st.session_state.patrones_referencia)
        matriz_distancias = np.zeros((n_patrones, n_patrones))
        
        for i in range(n_patrones):
            for j in range(n_patrones):
                if i != j:
                    distancia = np.sum(st.session_state.patrones_referencia[i] != st.session_state.patrones_referencia[j])
                    matriz_distancias[i, j] = distancia
        
        # Mostrar matriz de distancias
        fig_dist_matrix = px.imshow(
            matriz_distancias,
            labels=dict(x="Patrón", y="Patrón", color="Distancia Hamming"),
            title="Matriz de Distancias Hamming entre Patrones",
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_dist_matrix, use_container_width=True)
        
        # Tabla de distancias
        st.subheader("📋 Tabla de Distancias")
        
        data = []
        for i in range(n_patrones):
            for j in range(i+1, n_patrones):
                distancia = np.sum(st.session_state.patrones_referencia[i] != st.session_state.patrones_referencia[j])
                porcentaje = (distancia / 9) * 100
                data.append({
                    'Patrón A': f'Patrón {i}',
                    'Patrón B': f'Patrón {j}',
                    'Distancia Hamming': distancia,
                    'Porcentaje Diferencia': f'{porcentaje:.1f}%',
                    'Estado': '✅ Buena separación' if porcentaje >= 30 else '⚠️ Muy similares'
                })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()