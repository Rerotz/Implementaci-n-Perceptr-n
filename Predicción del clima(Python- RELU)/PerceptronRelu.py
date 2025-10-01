import numpy as np
import random

class PerceptronReLU:
    def _init_(self, num_features, learning_rate=0.01, epochs=100):
        """
        Inicializa el perceptrón con función de activación ReLU pura
        
        Args:
            num_features: Número de características de entrada
            learning_rate: Tasa de aprendizaje
            epochs: Número de iteraciones
        """
        self.weights = np.random.randn(num_features) * 0.1
        self.bias = np.random.randn() * 0.1
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.errors_history = []
    
    def relu(self, x):
        """Función de activación ReLU"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivada de la función ReLU"""
        return (x > 0).astype(float)
    
    def predict(self, X):
        """Realiza predicciones usando solo ReLU"""
        z = np.dot(X, self.weights) + self.bias
        return self.relu(z)
    
    def train(self, X, y):
        """
        Entrena el perceptrón usando solo ReLU
        
        Args:
            X: Matriz de características (samples x features)
            y: Vector de etiquetas (0 o 1)
        """
        for epoch in range(self.epochs):
            total_error = 0
            
            for i in range(len(X)):
                # Forward pass
                z = np.dot(X[i], self.weights) + self.bias
                prediction = self.relu(z)
                
                # Calcular error
                error = y[i] - prediction
                total_error += error ** 2
                
                # Backward pass con derivada de ReLU
                gradient = error * self.relu_derivative(z)
                
                # Actualizar pesos y bias
                self.weights += self.learning_rate * gradient * X[i]
                self.bias += self.learning_rate * gradient
            
            self.errors_history.append(total_error / len(X))
            
            if (epoch + 1) % 10 == 0:
                print(f"Época {epoch + 1}/{self.epochs}, Error promedio: {self.errors_history[-1]:.4f}")
    
    def classify(self, X, threshold=0.5):
        """Clasifica las entradas usando un umbral"""
        predictions = self.predict(X)
        return (predictions >= threshold).astype(int)


def crear_dataset_clima():
    """
    Crea un dataset sintético para predicción del clima
    Características: [temperatura, humedad, presion_atmosferica, velocidad_viento, nubosidad]
    0 = No lloverá, 1 = Lloverá
    """
    # Días sin lluvia (valores normalizados para que ReLU produzca valores < 0.5)
    sin_lluvia = np.array([
        [0.71, 0.40, 0.60, 0.20, 0.20],   # Día soleado
        [0.80, 0.35, 0.72, 0.12, 0.10],   # Día despejado
        [0.63, 0.45, 0.64, 0.24, 0.15],   # Día normal
        [0.74, 0.38, 0.68, 0.16, 0.25],   # Parcialmente nublado
        [0.86, 0.30, 0.80, 0.08, 0.05],   # Muy soleado
        [0.69, 0.42, 0.56, 0.20, 0.18],   # Día agradable
        [0.77, 0.36, 0.76, 0.12, 0.12],   # Día despejado
        [0.66, 0.44, 0.60, 0.24, 0.22],   # Día normal
        [0.83, 0.33, 0.84, 0.08, 0.08],   # Muy soleado
        [0.71, 0.40, 0.64, 0.16, 0.20],   # Día soleado
    ])
    
    # Días con lluvia (valores normalizados para que ReLU produzca valores > 0.5)
    con_lluvia = np.array([
        [0.51, 0.85, 0.32, 0.60, 0.90],  # Día lluvioso
        [0.46, 0.90, 0.20, 0.72, 0.95],  # Tormenta
        [0.54, 0.80, 0.40, 0.48, 0.85],  # Lluvia moderada
        [0.49, 0.88, 0.28, 0.64, 0.92],  # Lluvia fuerte
        [0.57, 0.75, 0.36, 0.56, 0.80],  # Lluvia ligera
        [0.43, 0.92, 0.16, 0.80, 0.98],  # Tormenta fuerte
        [0.51, 0.83, 0.32, 0.52, 0.88],  # Día lluvioso
        [0.46, 0.87, 0.24, 0.68, 0.93],  # Lluvia moderada
        [0.54, 0.78, 0.40, 0.44, 0.82],  # Lluvia ligera
        [0.49, 0.85, 0.28, 0.60, 0.90],  # Día lluvioso
    ])
    
    # Combinar datos
    X = np.vstack([sin_lluvia, con_lluvia])
    y = np.array([0] * len(sin_lluvia) + [1] * len(con_lluvia))
    
    # Datos originales para mostrar
    X_original = np.array([
        [25, 40, 1015, 5, 20], [28, 35, 1018, 3, 10], [22, 45, 1016, 6, 15],
        [26, 38, 1017, 4, 25], [30, 30, 1020, 2, 5], [24, 42, 1014, 5, 18],
        [27, 36, 1019, 3, 12], [23, 44, 1015, 6, 22], [29, 33, 1021, 2, 8],
        [25, 40, 1016, 4, 20], [18, 85, 1008, 15, 90], [16, 90, 1005, 18, 95],
        [19, 80, 1010, 12, 85], [17, 88, 1007, 16, 92], [20, 75, 1009, 14, 80],
        [15, 92, 1004, 20, 98], [18, 83, 1008, 13, 88], [16, 87, 1006, 17, 93],
        [19, 78, 1010, 11, 82], [17, 85, 1007, 15, 90]
    ])
    
    return X, y, X_original


# Programa principal
if _name_ == "_main_":
    print("=" * 70)
    print("PERCEPTRÓN - PREDICCIÓN DEL CLIMA (FUNCIÓN ReLU PURA)")
    print("=" * 70)
    
    # Crear dataset
    X, y, X_original = crear_dataset_clima()
    
    print(f"\nDatos de entrenamiento: {len(X)} muestras")
    print(f"Características: temperatura, humedad, presión, viento, nubosidad")
    print(f"Función de activación: ReLU (Rectified Linear Unit)")
    
    # Mezclar datos
    indices = list(range(len(X)))
    random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    X_original = X_original[indices]
    
    # Crear y entrenar el perceptrón
    perceptron = PerceptronReLU(num_features=X.shape[1], learning_rate=0.15, epochs=100)
    
    print("\n" + "=" * 70)
    print("ENTRENAMIENTO")
    print("=" * 70)
    perceptron.train(X, y)
    
    # Evaluar el modelo
    print("\n" + "=" * 70)
    print("EVALUACIÓN DEL MODELO")
    print("=" * 70)
    
    predictions = perceptron.classify(X)
    accuracy = np.mean(predictions == y) * 100
    
    print(f"\nPrecisión en el conjunto de entrenamiento: {accuracy:.2f}%")
    
    # Mostrar predicciones
    print("\n" + "=" * 70)
    print("PREDICCIONES DETALLADAS")
    print("=" * 70)
    print(f"{'#':<4} {'Temp':<6} {'Hum':<6} {'Pres':<6} {'Vien':<6} {'Nub':<6} {'Real':<12} {'Pred':<12} {'Salida':<8} {'✓/✗':<4}")
    print("-" * 70)
    
    for i in range(len(X)):
        output = perceptron.predict(X[i])
        pred = predictions[i]
        real = y[i]
        correcto = "✓" if pred == real else "✗"
        clase_real = "Lluvia" if real == 1 else "Sin Lluvia"
        clase_pred = "Lluvia" if pred == 1 else "Sin Lluvia"
        
        temp, hum, pres, vien, nub = X_original[i]
        print(f"{i+1:<4} {temp:<6.0f} {hum:<6.0f} {pres:<6.0f} {vien:<6.0f} {nub:<6.0f} {clase_real:<12} {clase_pred:<12} {output:.4f}   {correcto:<4}")
    
    # Probar con nuevos datos
    print("\n" + "=" * 70)
    print("PRUEBA CON NUEVOS DATOS")
    print("=" * 70)
    
    # Nuevos datos normalizados manualmente
    nuevos_datos = np.array([
        [0.74, 0.38, 0.68, 0.16, 0.18],  # Debería ser Sin Lluvia
        [0.49, 0.86, 0.24, 0.64, 0.91],  # Debería ser Lluvia
        [0.80, 0.32, 0.76, 0.12, 0.10],  # Debería ser Sin Lluvia
        [0.46, 0.91, 0.20, 0.76, 0.96],  # Debería ser Lluvia
    ])
    
    nuevos_datos_original = np.array([
        [26, 38, 1017, 4, 18],
        [17, 86, 1006, 16, 91],
        [28, 32, 1019, 3, 10],
        [16, 91, 1005, 19, 96],
    ])
    
    condiciones = ["Día soleado", "Día tormentoso", "Día despejado", "Tormenta fuerte"]
    
    for i, (datos, datos_orig, cond) in enumerate(zip(nuevos_datos, nuevos_datos_original, condiciones)):
        output = perceptron.predict(datos)
        pred = perceptron.classify(datos.reshape(1, -1))[0]
        clase = "Lluvia" if pred == 1 else "Sin Lluvia"
        
        temp, hum, pres, vien, nub = datos_orig
        print(f"\nMuestra {i+1} ({cond}):")
        print(f"  Temp: {temp}°C, Humedad: {hum}%, Presión: {pres}hPa")
        print(f"  Viento: {vien}km/h, Nubosidad: {nub}%")
        print(f"  → Predicción: {clase} (Salida ReLU: {output:.4f})")
    
    print("\n" + "=" * 70)
    print("INFORMACIÓN DE LA FUNCIÓN ReLU")
    print("=" * 70)
    print("ReLU(x) = max(0, x)")
    print("- Si x > 0: devuelve x")
    print("- Si x ≤ 0: devuelve 0")
    print("Umbral de clasificación: 0.5")
    print("- Salida ≥ 0.5 → Lluvia (clase 1)")
    print("- Salida < 0.5 → Sin Lluvia (clase 0)")
    
    print("\n" + "=" * 70)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 70)