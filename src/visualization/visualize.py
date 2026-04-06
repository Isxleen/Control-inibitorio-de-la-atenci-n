import matplotlib.pyplot as plt
import seaborn as sns
import shap as shap
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, title="Matriz de Confusión"):
    """
    Plotea una matriz de confusión bonita usando Seaborn.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title(title)
    plt.show()

def plot_shap_summary(model, X, model_type='tree'):
    """
    Genera gráficos SHAP para interpretar el modelo.
    Soporta TreeExplainer (RF, XGB) y KernelExplainer (Otros).
    """
    try:
        if model_type == 'tree':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
        else:
            explainer = shap.KernelExplainer(model.predict_proba, X)
            shap_values = explainer.shap_values(X)


        shap.summary_plot(shap_values, X, show=False, plot_size=(12, 8), max_display=15)

        plt.show()

    except Exception as e:
        print(f"No se pudo generar SHAP automáticamente: {e}")

def plot_pca_2d(X_pca, clusters):
    """
    Visualiza predicción PCA 2D.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters, palette='viridis', s=100)
    plt.title('Análisis de Componentes Principales (PCA)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()
