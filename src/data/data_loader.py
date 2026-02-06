import pandas as pd
import numpy as np
import os

def load_data(data_path):
    """
    Identifica la extensión y carga el archivo.
    """
    if data_path.endswith('.csv'):
        return pd.read_csv(data_path)
    elif data_path.endswith('.xlsx'):
        return pd.read_excel(data_path, engine='openpyxl')
    else:
        raise ValueError(f"Formato no soportado: {data_path}")

def clean_rt_data(df, task_name="Task"):
    """
    Limpieza estándar para tiempos de reacción:
    1. Filtra trials de práctica (values.practice == 0/False) si existe la columna.
    2. Filtra errores (correct == 1).
    3. Filtra outliers fisiológicos (e.g. < 200ms o > 3000ms).
    """
    print(f"--- Limpiando {task_name} (N={len(df)}) ---")
    
    # Normalizar columnas a minúsculas
    df.columns = [c.lower() for c in df.columns]
    
    # 1. Práctica
    if 'values.practice' in df.columns:
        original_n = len(df)
        df = df[df['values.practice'] == 0] # Asumiendo 0 es test block
        print(f"Filtrados trials de práctica: {original_n} -> {len(df)}")
    elif 'blockcode' in df.columns:
        # A veces blockcode contiene 'practice'
        if df['blockcode'].dtype == 'O':
             df = df[~df['blockcode'].astype(str).str.contains('practice', case=False, na=False)]
             print(f"Filtrados trials por blockcode: {len(df)}")

    # 2. Correctos
    if 'correct' in df.columns:
        df = df[df['correct'] == 1] # 1 = Correcto (Inquisit standard)
    
    # 3. Latencia (RT)
    if 'latency' in df.columns:
        # Filtrado fisiológico
        df = df[(df['latency'] > 200) & (df['latency'] < 3000)]
        
        # Opcional: Filtrado estadístico (e.g. +- 3 SD por sujeto)
        # Se puede implementar si es necesario
    
    print(f"Final N ({task_name}): {len(df)}")
    return df

def get_raw_datasets(raw_data_dir):
    """
    Carga y limpia Stroop y Flanker. Retorna DataFrames.
    """
    path_stroop = os.path.join(raw_data_dir, "Raw_Stroop.xlsx")
    path_flanker = os.path.join(raw_data_dir, "Raw_Flanker.xlsx")
    
    df_stroop = load_data(path_stroop)
    df_flanker = load_data(path_flanker)
    
    df_stroop_clean = clean_rt_data(df_stroop, "Stroop")
    df_flanker_clean = clean_rt_data(df_flanker, "Flanker")
    
    return df_stroop_clean, df_flanker_clean

if __name__ == "__main__":
    # Test rápido
    base_dir = r"c:\Users\alion\OneDrive\Escritorio\Control inibitorio de la atención\data\raw"
    try:
        s, f = get_raw_datasets(base_dir)
        print("Carga exitosa.")
        print("Stroop Cols:", s.columns.tolist())
    except Exception as e:
        print("Error:", e)
