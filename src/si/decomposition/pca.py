print("PCA module loaded successfully.")

import numpy as np

class PCA:
    def __init__(self, n_components):
        """
        Inicializa a classe PCA.

        Parâmetros:
        - n_components: int
            Número de componentes principais a serem mantidos.
        """
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def _fit(self, X):
        """
        Ajusta o modelo PCA aos dados.

        Parâmetros:
        - X: numpy.ndarray
            Dados de entrada com forma (n_amostras, n_features).
        """
        # 1. Centralizar os dados
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # 2. Calcular a matriz de covariância
        cov_matrix = np.cov(X_centered, rowvar=False)

        # 3. Decomposição em valores e vetores próprios
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 4. Ordenar os valores/vetores próprios em ordem decrescente
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # 5. Selecionar os n_components principais
        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]

    def _transform(self, X):
        """
        Transforma os dados para as dimensões reduzidas.

        Parâmetros:
        - X: numpy.ndarray
            Dados de entrada com forma (n_amostras, n_features).

        Retorna:
        - X_reduced: numpy.ndarray
            Dados transformados com dimensões reduzidas.
        """
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        """
        Ajusta o modelo PCA e transforma os dados.

        Parâmetros:
        - X: numpy.ndarray
            Dados de entrada com forma (n_amostras, n_features).

        Retorna:
        - X_reduced: numpy.ndarray
            Dados transformados com dimensões reduzidas.
        """
        self._fit(X)
        return self._transform(X)
