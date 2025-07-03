import numpy as np

class RedHammingSimple:
    """
    Red Neuronal Hamming SÚPER SIMPLE
    
    ¿Qué hace? 
    - Memoriza patrones (como fotos en blanco y negro)
    - Cuando le muestras algo parecido, te dice cuál es
    
    ¿Cómo funciona?
    - Cuenta cuántos "pixeles" son diferentes
    - El que tenga menos diferencias, ¡ese es!
    """
    
    def __init__(self):
        # Aquí guardamos los patrones que la red "recuerda"
        self.patrones_memoria = {}
        self.nombres = {}
    
    def aprender(self, patron, nombre):
        """
        Enseñamos un patrón a la red
        
        patron: lista de 0s y 1s (como pixeles blancos y negros)
        nombre: cómo se llama este patrón
        """
        indice = len(self.patrones_memoria)
        self.patrones_memoria[indice] = patron
        self.nombres[indice] = nombre
        print(f"✓ Aprendido: {nombre} -> {patron}")
    
    def contar_diferencias(self, patron1, patron2):
        """
        Cuenta cuántos 'pixeles' son diferentes entre dos patrones
        Esto se llama "Distancia de Hamming"
        """
        diferencias = 0
        for i in range(len(patron1)):
            if patron1[i] != patron2[i]:
                diferencias += 1
        return diferencias
    
    def reconocer(self, patron_nuevo):
        """
        La red trata de reconocer qué es el patrón nuevo
        """
        print(f"\n🔍 Tratando de reconocer: {patron_nuevo}")
        
        if len(self.patrones_memoria) == 0:
            print("❌ La red no ha aprendido nada aún")
            return None
        
        mejor_coincidencia = None
        menor_diferencias = float('inf')
        
        print("\n📊 Comparando con lo que sé:")
        
        # Comparar con cada patrón que recuerda
        for indice, patron_conocido in self.patrones_memoria.items():
            diferencias = self.contar_diferencias(patron_nuevo, patron_conocido)
            nombre = self.nombres[indice]
            
            print(f"   vs {nombre}: {diferencias} diferencias")
            
            # ¿Este es el más parecido hasta ahora?
            if diferencias < menor_diferencias:
                menor_diferencias = diferencias
                mejor_coincidencia = indice
        
        # Resultado
        if mejor_coincidencia is not None:
            nombre_ganador = self.nombres[mejor_coincidencia]
            print(f"\n🎯 ¡Es un {nombre_ganador}! ({menor_diferencias} diferencias)")
            return nombre_ganador
        
        return None


def demo_super_simple():
    """
    Demo con formas simples de 3x3 pixeles
    """
    print("="*50)
    print("🧠 RED NEURONAL HAMMING - EJEMPLO SÚPER SIMPLE")
    print("="*50)
    print("\nVamos a enseñarle a reconocer formas simples:")
    print("Cada forma es de 3x3 = 9 pixeles")
    print("1 = pixel negro, 0 = pixel blanco\n")
    
    # Crear la red
    red = RedHammingSimple()
    
    # Enseñarle algunas formas básicas
    print("📚 FASE DE APRENDIZAJE:")
    print("-" * 30)
    
    # Cuadrado
    cuadrado = [1, 1, 1,
                1, 0, 1, 
                1, 1, 1]
    red.aprender(cuadrado, "CUADRADO")
    
    # Cruz
    cruz = [0, 1, 0,
            1, 1, 1,
            0, 1, 0]
    red.aprender(cruz, "CRUZ")
    
    # Línea horizontal
    linea = [0, 0, 0,
             1, 1, 1,
             0, 0, 0]
    red.aprender(linea, "LÍNEA")
    
    print("\n" + "="*50)
    print("🧪 FASE DE RECONOCIMIENTO:")
    print("="*50)
    
    # Ahora probemos con patrones exactos
    print("\n1️⃣ Probando con patrones exactos:")
    red.reconocer(cuadrado)
    red.reconocer(cruz)
    
    # Probemos con patrones con un poco de "ruido"
    print("\n2️⃣ Probando con patrones con ruido:")
    
    # Cuadrado con un pixel cambiado
    cuadrado_roto = [1, 1, 0,  # Era 1, ahora es 0
                     1, 0, 1, 
                     1, 1, 1]
    red.reconocer(cuadrado_roto)
    
    # Cruz con dos pixeles cambiados
    cruz_rota = [0, 1, 1,  # Era 0, ahora es 1
                 1, 0, 1,  # Era 1, ahora es 0
                 0, 1, 0]
    red.reconocer(cruz_rota)
    
    print("\n3️⃣ Probando con algo completamente diferente:")
    
    # Algo random
    random_pattern = [1, 0, 1,
                      0, 1, 0,
                      1, 0, 1]
    red.reconocer(random_pattern)


def visualizar_patron(patron, nombre="Patrón"):
    """
    Muestra el patrón de forma visual
    """
    print(f"\n{nombre}:")
    for i in range(0, 9, 3):  # Cada 3 elementos = una fila
        fila = ""
        for j in range(3):
            if patron[i + j] == 1:
                fila += "██ "  # Pixel negro
            else:
                fila += "   "  # Pixel blanco
        print(fila)


def demo_con_visualizacion():
    """
    Demo con visualización de los patrones
    """
    print("\n" + "="*50)
    print("👁️ DEMO CON VISUALIZACIÓN")
    print("="*50)
    
    # Los mismos patrones pero visualizados
    cuadrado = [1, 1, 1, 1, 0, 1, 1, 1, 1]
    cruz = [0, 1, 0, 1, 1, 1, 0, 1, 0]
    linea = [0, 0, 0, 1, 1, 1, 0, 0, 0]
    
    visualizar_patron(cuadrado, "CUADRADO")
    visualizar_patron(cruz, "CRUZ")
    visualizar_patron(linea, "LÍNEA")
    
    print("\nAhora un cuadrado con ruido:")
    cuadrado_roto = [1, 1, 0, 1, 0, 1, 1, 1, 1]
    visualizar_patron(cuadrado_roto, "CUADRADO CON RUIDO")
    
    # Crear red y probar
    red = RedHammingSimple()
    red.aprender(cuadrado, "CUADRADO")
    red.aprender(cruz, "CRUZ")
    red.aprender(linea, "LÍNEA")
    
    red.reconocer(cuadrado_roto)


def explicacion_teoria():
    """
    Explicación simple de cómo funciona
    """
    print("\n" + "="*50)
    print("🎓 ¿CÓMO FUNCIONA LA RED HAMMING?")
    print("="*50)
    
    print("""
🧠 CONCEPTO BÁSICO:
   La red memoriza patrones y los compara bit por bit
   
📏 DISTANCIA DE HAMMING:
   - Cuenta cuántos bits son diferentes
   - Ejemplo: [1,0,1] vs [1,1,1] = 1 diferencia
   
🎯 RECONOCIMIENTO:
   - Compara el patrón nuevo con todos los memorizados
   - El que tenga MENOS diferencias, ¡ese es el ganador!
   
💪 VENTAJAS:
   ✓ Muy simple de entender
   ✓ Tolerante al ruido
   ✓ Rápida para patrones pequeños
   
⚠️ LIMITACIONES:
   ✗ Solo funciona bien con patrones similares en tamaño
   ✗ Puede confundirse si hay muchos patrones parecidos
   
🔥 APLICACIONES REALES:
   • Reconocimiento de caracteres
   • Códigos de barras
   • Detección de errores en comunicaciones
   • Clasificación de imágenes binarias
    """)


if __name__ == "__main__":
    # Ejecutar todas las demos
    demo_super_simple()
    demo_con_visualizacion()
    explicacion_teoria()
    
    print("\n" + "="*50)
    print("🏁 ¡DEMO COMPLETADA!")
    print("="*50)
    print("""
💡 RESUMEN:
La Red Neuronal Hamming es como tener una memoria fotográfica
que puede reconocer cosas incluso si están un poco dañadas.

¡Es perfecta para empezar a entender las redes neuronales!
    """)