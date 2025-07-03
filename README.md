# Perceptrón desde Cero: ¿Tarjeta Platinum?

Este proyecto muestra cómo funciona un perceptrón para clasificar si una persona obtiene una tarjeta de crédito, según su edad y ahorro.

🔍 **¿Qué hace?** Entrena un perceptrón manualmente (desde cero). Compara resultados con scikit‑learn. Visualiza: Zona de decisión (quién es aprobado o denegado), estructura del perceptrón y frontera de decisión del modelo entrenado.

📦 **Requisitos** Asegúrate de tener Python 3 y estas librerías:

pip install numpy matplotlib scikit‑learn
▶️ Cómo ejecutar Guarda el archivo como perceptron_tarjeta.py y corre:

bash
Copiar
Editar
python perceptron_tarjeta.py
📈 ¿Qué vas a ver? Pesos y sesgo aprendidos por el perceptrón, comparación con el perceptrón de scikit‑learn, y gráficos que muestran clasificación de personas, zona de decisión, estructura interna del perceptrón y línea de decisión del modelo.

📊 Datos de ejemplo Los datos representan personas con diferentes edades y niveles de ahorro. El modelo aprende a decidir si su solicitud de tarjeta es:

✅ Aprobada
❌ Denegada

------------------------------------------------

# 🧠 Red Neuronal de Hamming para Reconocimiento de Patrones

Este proyecto implementa una **Red Neuronal de Hamming** desde cero en Python, con el objetivo de reconocer patrones binarios (como números 0, 1 y 2 en una matriz de 3x3), incluso si contienen **ruido**.

---

## 📌 ¿Qué hace esta red?

- Compara un patrón de entrada con patrones aprendidos.
- Encuentra el más **similar** usando la distancia de Hamming.
- Usa una red de competición tipo **Maxnet** para decidir cuál patrón gana.
- ¡Funciona incluso con ruido aleatorio!

---

## 🔧 Estructura del código

- `RedNeuronalHamming`: clase principal que contiene dos capas:
  - **Capa Hamming**: mide similitud patrón por patrón.
  - **Capa Maxnet**: decide cuál patrón es el más parecido.
- Patrones de referencia: 0, 1 y 2 en una matriz 3x3.
- Se puede agregar **ruido** y hacer pruebas interactivas.

---

## 🚀 ¿Cómo usarlo?

1. Asegúrate de tener Python 3 y `matplotlib` instalado.
2. Ejecuta el script:


python hamming.py


# 🖥️ Red Neuronal MNIST: Clasificador de Dígitos Manuscritos

Este proyecto proporciona un **manual de usuario** para ejecutar y entender un script en Python que:

- Carga y visualiza el dataset MNIST.
- Construye, entrena y evalúa una red neuronal de una capa oculta con TensorFlow/Keras.
- Muestra la arquitectura de la red como diagrama.
- Grafica la curva de pérdida y precisión durante el entrenamiento.
- Presenta predicciones sobre ejemplos de prueba.
- Genera matriz de confusión y reporte de clasificación.

---

## 📋 Requisitos

- **Python 3.7+**
- **TensorFlow 2.x**  
- **Matplotlib**  
- **NumPy**  
- **Seaborn** (opcional, para la matriz de confusión)  
- **Scikit‑learn** (para métricas)

Instala todo con:
```bash
pip install tensorflow matplotlib numpy seaborn scikit-learn
🚀 Uso
Descarga o clona este repositorio.

Guarda el script como mnist_classifier.py (o el nombre que prefieras).

Desde la terminal, en el directorio del script, ejecuta:

bash
Copiar
Editar
python mnist_classifier.py
🔍 Flujo de ejecución
Carga del dataset

Se descargan automáticamente los arreglos x_train, y_train, x_test, y_test.

Se imprime en consola la forma de los datos.

Visualización inicial

Se muestran 12 ejemplos de dígitos manuscritos con sus etiquetas.

Normalización y aplanado

Convierte píxeles de [0,255] a [0.0,1.0].

Aplana cada imagen 28×28 en vectores de longitud 784.

Definición del modelo

Capa oculta: 128 neuronas con ReLU + Dropout 20%.

Capa de salida: 10 neuronas con Softmax.

Visualización de la arquitectura

Diagrama estático que muestra las capas y conexiones (simplificado).

Explicación en consola

Resumen del flujo: entrada → oculta → salida → entrenamiento.

Entrenamiento

Épocas: 10 (puedes ajustar).

Batch size: 128.

Validación con 10% de los datos de entrenamiento.

Optimizador: Adam.

Pérdida: sparse_categorical_crossentropy.

Evaluación

Precisión y pérdida sobre el conjunto de prueba.

Mensaje de valoración según el porcentaje de acierto.

Gráficas de entrenamiento

Curva de pérdida (entrenamiento vs. validación).

Curva de precisión (entrenamiento vs. validación).

Predicción de ejemplos

Muestra 10 dígitos de prueba con etiqueta real y predicción.

Matriz de confusión y reporte

Heatmap con conteos de verdaderos/falsos positivos.

Informe con precisión, recall y F1‑score por dígito.

🎛️ Personalización
Épocas: modifica epochs= en model.fit().

Batch size: ajusta batch_size=.

Dropout: cambia el porcentaje en layers.Dropout().

Tamaño de la capa oculta: varía layers.Dense(128, ...).

Visualización: comenta o descomenta secciones plt.show().

📊 Interpretación de resultados
Precisión de Test: porcentaje de dígitos correctamente clasificados.

Curvas de entrenamiento: sirven para detectar sobreajuste/infraajuste.

Matriz de Confusión: identifica dígitos que el modelo confunde con más frecuencia.

Reporte de clasificación: muestra métricas detalladas por clase (0–9).

🛠️ Solución de problemas
Si falla la descarga de MNIST, revisa tu conexión o actualiza TensorFlow.
