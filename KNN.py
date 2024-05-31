import numpy as np

# Função para calcular a distância euclidiana entre dois pontos
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

# Classe do modelo KNN
class KNN:
    def __init__(self, k=3):
        self.k = k

    # Método de treinamento: apenas armazena os dados de treinamento
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # Método de previsão
    def predict(self, X):
        predictions = []

        # Itera sobre cada instância de teste
        for x_test in X:
            # Calcula as distâncias entre o ponto de teste e todos os pontos de treinamento
            distances = [euclidean_distance(x_test, x_train) for x_train in self.X_train]
            
            # Ordena os índices das distâncias em ordem crescente
            k_indices = np.argsort(distances)[:self.k]
            
            # Obtém os rótulos das k instâncias mais próximas
            k_nearest_labels = [self.y_train[i] for i in k_indices]

            # Realiza a votação majoritária para determinar a classe do ponto de teste
            most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
            predictions.append(most_common)

        return predictions

# Exemplo de uso
if __name__ == "__main__":
    # Dados de treinamento e teste
    X_train = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    y_train = np.array([0, 0, 1, 1, 0, 1])

    X_test = np.array([[1, 3], [8, 9], [0, 3], [5, 4], [6, 4]])

    # Criando e treinando o modelo KNN
    clf = KNN(k=2)
    clf.fit(X_train, y_train)

    # Realizando previsões
    predictions = clf.predict(X_test)
    print("Predictions:", predictions) 
