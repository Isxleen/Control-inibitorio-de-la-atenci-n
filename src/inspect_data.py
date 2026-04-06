import pandas as pd
import os

data_dir = r"c/Users/isabelsalinas/Documents/Documentos TFG/TFG-Control_inhibitorio_atencion/data/raw"
files = ["Raw_Sart.xlsx", "Raw_Flanker.xlsx", "Raw_Stroop.xlsx"]

for f in files:
    path = os.path.join(data_dir, f)
    print(f"\n{'='*20}\nInspecting {f}\n{'='*20}")
    try:
        if f.endswith('.csv'):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path, nrows=100) # Read only first 100 rows for inspection to be fast
        
        print("COLUMNS:", df.columns.tolist())
        print("\nDTYPES:\n", df.dtypes)
        print("\nHEAD:\n", df.head(3))
    except Exception as e:
        print(f"Error reading {f}: {e}")
