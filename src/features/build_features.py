import pandas as pd
import numpy as np

def calculate_variability(df, task_prefix):
    """
    Calcula variabilidad intra-sujeto (SD del RT).
    Marcador importante de atención sostenida.
    """
    stats = df.groupby('subjectid')['latency'].agg(['std', 'mean', 'median'])
    stats.columns = [
        f'{task_prefix}_rt_std', 
        f'{task_prefix}_rt_mean_recalc', 
        f'{task_prefix}_rt_median'
    ]
    return stats

def calculate_interference(df, task_prefix, congruence_col='values.congruence'):
    """
    Calcula efecto de conflicto: Incongruente - Congruente.
    Asume que la columna tiene 2 valores:
    1 = Congruente (más rápido)
    2/0 = Incongruente (más lento) -> Validar códigos
    """
    # Agrupar por sujeto y congruencia
    # Primero detectamos los códigos
    codes = sorted(df[congruence_col].unique())
    print(f"[{task_prefix}] Códigos de congruencia encontrados: {codes}")
    
    # Asumimos heurística: Código con menor RT promedio global es Congruente
    global_means = df.groupby(congruence_col)['latency'].mean()
    congruent_code = global_means.idxmin()
    incongruent_code = global_means.idxmax()
    
    print(f"[{task_prefix}] Asumiendo Congruente={congruent_code}, Incongruente={incongruent_code}")
    
    pivot = df.pivot_table(
        index='subjectid', 
        columns=congruence_col, 
        values='latency', 
        aggfunc='mean'
    )
    
    # Calcular diferencia
    interference = pd.DataFrame(index=pivot.index)
    interference[f'{task_prefix}_congruent_rt'] = pivot[congruent_code]
    interference[f'{task_prefix}_incongruent_rt'] = pivot[incongruent_code]
    interference[f'{task_prefix}_interference'] = pivot[incongruent_code] - pivot[congruent_code]
    
    return interference

def process_features(df_stroop, df_flanker, df_base_subjects):
    """
    Pipeline principal de Features.
    """
    print("--- Extrayendo Features ---")
    
    # 1. Stroop Features
    # Stroop usa 'values.congruency' (verificado en logs)
    stroop_cong_col = 'values.congruency'
    if stroop_cong_col not in df_stroop.columns:
        print(f"WARN: No se encontró {stroop_cong_col} en Stroop. Cols: {df_stroop.columns.tolist()}")
        # Intento fallback
        stroop_cong_col = 'values.congruence'
    
    stroop_var = calculate_variability(df_stroop, "stroop")
    stroop_eff = calculate_interference(df_stroop, "stroop", congruence_col=stroop_cong_col)
    
    # 2. Flanker Features
    # Flanker usa 'values.congruence'
    flanker_cong_col = 'values.congruence'
    flanker_var = calculate_variability(df_flanker, "flanker")
    flanker_eff = calculate_interference(df_flanker, "flanker", congruence_col=flanker_cong_col)
    
    # 3. Merge
    final_df = df_base_subjects.copy()
    if 'subjectid' not in final_df.columns: # A veces es SubjectID
        final_df.rename(columns={'SubjectID': 'subjectid'}, inplace=True)
    
    final_df = final_df.set_index('subjectid')
    
    features_list = [stroop_var, stroop_eff, flanker_var, flanker_eff]
    for feat in features_list:
        final_df = final_df.join(feat, how='left')
        
    print("Features procesadas. Shape final:", final_df.shape)
    return final_df

if __name__ == "__main__":
    # Test script (requiere data_loader funcionando)
    pass
