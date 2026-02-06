import os
import pandas as pd
from data.data_loader import get_raw_datasets
from features.build_features import process_features

def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # src/..
    raw_dir = os.path.join(project_dir, 'data', 'raw')
    processed_dir = os.path.join(project_dir, 'data', 'processed')
    
    print(">>> 1. Cargando Datos Crudos...")
    try:
        df_stroop, df_flanker = get_raw_datasets(raw_dir)
        print("Datos de tareas cargados.")
    except Exception as e:
        print(f"ERROR cargando Raw Data: {e}")
        return

    print(">>> 2. Cargando Dataset Agregado (Subject Level)...")
    base_path = os.path.join(raw_dir, 'subject_level_dataset.csv')
    if os.path.exists(base_path):
        df_base = pd.read_csv(base_path)
        print(f"Base data cargada. N={len(df_base)}")
    else:
        print("ERROR: No se encontró subject_level_dataset.csv")
        return

    print(">>> 3. Generando Features Avanzadas...")
    try:
        final_df = process_features(df_stroop, df_flanker, df_base)
        
        # Guardar
        output_path = os.path.join(processed_dir, 'final_dataset_with_features.csv')
        final_df.to_csv(output_path)
        print(f">>> ÉXITO. Dataset guardado en: {output_path}")
        print(final_df.head())
        
    except Exception as e:
        print(f"ERROR procesando features: {e}")
        # Debug: Imprimir columnas si falla
        print("Stroop Columns:", df_stroop.columns.tolist())

if __name__ == "__main__":
    main()
