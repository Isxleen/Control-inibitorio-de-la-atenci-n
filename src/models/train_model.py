from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import pandas as pd
import numpy as np

def train_stacking_ensemble(X, y):
    """
    Entrena un Stacking Classifier robusto.
    Base Learners: Random Forest, SVM (con probabilidad), XGBoost, KNN.
    Meta Learner: Logistic Regression.
    """
    # 1. Definir Base Learners
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('svm', SVC(probability=True, kernel='rbf', random_state=42)),
        ('xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ]
    
    # 2. Definir Stacking
    clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=5
    )
    
    # 3. Split y Entrenamiento
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print("Entrenando Stacking Ensemble...")
    clf.fit(X_train, y_train)
    
    # 4. Evaluación
    y_pred = clf.predict(X_test)
    print("\n--- Resultados del Stacking Classifier ---\n")
    print(classification_report(y_test, y_pred))
    
    return clf, X_test, y_test, y_pred

def evaluate_models_cv(X, y):
    """
    Compara modelos individuales vs Stacking usando Cross Validation.
    """
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        'SVM': SVC(random_state=42)
    }
    
    results = {}
    print("--- Validación Cruzada (cv=5) ---")
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
        results[name] = scores
        
    return results
