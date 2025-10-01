import numpy as np
import random

class PerceptronSigmoidal:
    def _init_(self, num_features, learning_rate=0.1, epochs=100):
        """
        Inicializa el perceptrón con función de activación sigmoidal
        
        Args:
            num_features: Número de características de entrada
            learning_rate: Tasa de aprendizaje
            epochs: Número de iteraciones
        """
        self.weights = np.random.randn(num_features)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.errors_history = []
    
    def sigmoid(self, x):
        """Función de activación sigmoidal"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """Derivada de la función sigmoidal"""
        sig = self.sigmoid(x)
        return sig * (1 - sig)
    
    def predict(self, X):
        """Realiza predicciones"""
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def train(self, X, y):
        """
        Entrena el perceptrón
        
        Args:
            X: Matriz de características (samples x features)
            y: Vector de etiquetas (0 o 1)
        """
        for epoch in range(self.epochs):
            total_error = 0
            
            for i in range(len(X)):
                # Forward pass
                z = np.dot(X[i], self.weights) + self.bias
                prediction = self.sigmoid(z)
                
                # Calcular error
                error = y[i] - prediction
                total_error += error ** 2
                
                # Backward pass (gradiente descendente)
                gradient = error * self.sigmoid_derivative(z)
                
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


# Dataset de ejemplo para clasificación de Spam
# Características: [longitud_mensaje, num_enlaces, num_palabras_spam, num_mayusculas, num_signos_exclamacion]
def crear_dataset_spam():
    """
    Crea un dataset sintético para clasificación de spam
    0 = No Spam, 1 = Spam
    """
    # Mensajes normales (No Spam)
    normal = np.array([
        [50, 0, 0, 5, 0],    # Mensaje corto, sin enlaces, sin palabras spam
        [100, 1, 0, 10, 0],  # Mensaje normal con un enlace
        [75, 0, 1, 8, 0],    # Mensaje normal
        [120, 1, 1, 15, 1],  # Mensaje largo normal
        [60, 0, 0, 6, 0],    # Mensaje corto normal
        [90, 1, 0, 12, 0],   # Mensaje normal con enlace
        [80, 0, 1, 9, 1],    # Mensaje normal
        [110, 1, 1, 14, 0],  # Mensaje largo normal
    ])
    
    # Mensajes de spam
    spam = np.array([
        [200, 5, 10, 50, 5],  # Mensaje largo con muchos enlaces y palabras spam
        [150, 4, 8, 40, 4],   # Spam típico
        [180, 6, 12, 55, 6],  # Spam agresivo
        [160, 3, 9, 45, 3],   # Spam moderado
        [220, 7, 15, 60, 7],  # Spam muy agresivo
        [170, 5, 11, 48, 5],  # Spam típico
        [190, 4, 10, 52, 4],  # Spam
        [210, 6, 13, 58, 6],  # Spam agresivo
    ])
    
    # Combinar datos
    X = np.vstack([normal, spam])
    y = np.array([0] * len(normal) + [1] * len(spam))
    
    # Normalizar características
    X = X / np.max(X, axis=0)
    
    return X, y


# Programa principal
if _name_ == "_main_":
    print("=" * 60)
    print("PERCEPTRÓN - CLASIFICACIÓN DE SPAM (FUNCIÓN SIGMOIDAL)")
    print("=" * 60)
    
    # Crear dataset
    X, y = crear_dataset_spam()
    
    print(f"\nDatos de entrenamiento: {len(X)} muestras")
    print(f"Características: longitud, enlaces, palabras_spam, mayúsculas, exclamaciones")
    
    # Mezclar datos
    indices = list(range(len(X)))
    random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    # Crear y entrenar el perceptrón
    perceptron = PerceptronSigmoidal(num_features=X.shape[1], learning_rate=0.5, epochs=50)
    
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO")
    print("=" * 60)
    perceptron.train(X, y)
    
    # Evaluar el modelo
    print("\n" + "=" * 60)
    print("EVALUACIÓN DEL MODELO")
    print("=" * 60)
    
    predictions = perceptron.classify(X)
    accuracy = np.mean(predictions == y) * 100
    
    print(f"\nPrecisión en el conjunto de entrenamiento: {accuracy:.2f}%")
    
    # Mostrar predicciones
    print("\n" + "=" * 60)
    print("PREDICCIONES DETALLADAS")
    print("=" * 60)
    print(f"{'Muestra':<8} {'Real':<10} {'Predicción':<12} {'Probabilidad':<12} {'Correcto':<10}")
    print("-" * 60)
    
    for i in range(len(X)):
        prob = perceptron.predict(X[i])
        pred = predictions[i]
        real = y[i]
        correcto = "✓" if pred == real else "✗"
        clase_real = "Spam" if real == 1 else "No Spam"
        clase_pred = "Spam" if pred == 1 else "No Spam"
        
        print(f"{i+1:<8} {clase_real:<10} {clase_pred:<12} {prob:.4f}       {correcto:<10}")
    
    # Probar con nuevos datos
    print("\n" + "=" * 60)
    print("PRUEBA CON NUEVOS DATOS")
    print("=" * 60)
    
    nuevos_datos = np.array([
        [70, 0, 1, 8, 0],      # Debería ser No Spam
        [195, 6, 11, 53, 6],   # Debería ser Spam
        [85, 1, 0, 10, 1],     # Debería ser No Spam
        [175, 5, 9, 47, 5],    # Debería ser Spam
    ])
    
    # Normalizar
    nuevos_datos = nuevos_datos / np.array([220, 7, 15, 60, 7])
    
    for i, datos in enumerate(nuevos_datos):
        prob = perceptron.predict(datos)
        pred = perceptron.classify(datos.reshape(1, -1))[0]
        clase = "Spam" if pred == 1 else "No Spam"
        print(f"Mensaje {i+1}: {clase} (Probabilidad: {prob:.4f})")
    
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 60)