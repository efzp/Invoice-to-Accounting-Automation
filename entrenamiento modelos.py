# entrenamiento_autogluon_contabilidad.py

from pathlib import Path
from datetime import datetime
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor


# ============================================================
# 1. RUTAS
# ============================================================

carpeta_salida = Path(
    r"C:/Users/eduar/OneDrive - TCP BAAS S.A.S/Automatización/facturas_bigquery/resultados"
)

ruta_dataset = carpeta_salida / "08_dataset_autogluon.xlsx"

# Métricas en OneDrive
carpeta_metricas = carpeta_salida / "metricas_modelo"

# Modelo fuera de OneDrive para evitar bloqueos de permisos
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

carpeta_modelo = Path(
    r"C:/modelos_autogluon"
) / f"modelo_autogluon_contabilidad_facturas_{timestamp}"

carpeta_modelo.mkdir(parents=True, exist_ok=True)
carpeta_metricas.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CARGAR DATASET
# ============================================================

if not ruta_dataset.exists():
    raise FileNotFoundError(
        f"No existe el archivo:\n{ruta_dataset}\n\n"
        "Verifica que el pipeline haya exportado 08_dataset_autogluon.xlsx."
    )

df_modelo = pd.read_excel(ruta_dataset)

print("Dataset cargado")
print("Ruta:", ruta_dataset)
print("Filas:", len(df_modelo))
print("Columnas:", len(df_modelo.columns))


# ============================================================
# 3. LIMPIEZA BÁSICA
# ============================================================

target = "target_plantilla_cuentas"

if target not in df_modelo.columns:
    raise ValueError(f"No existe la columna objetivo: {target}")

df_modelo = df_modelo.dropna(subset=[target]).copy()
df_modelo[target] = df_modelo[target].astype(str)

columnas_excluir = [
    "empresa",
    "llave_factura",
    "llave_asiento",
    "estado_match",
    "motivo_score",
    "score_total",
    "concepto_concat",
    "plantilla_cuentas",
    "plantilla_cuentas_dc",
]

df_modelo = df_modelo.drop(
    columns=[c for c in columnas_excluir if c in df_modelo.columns],
    errors="ignore",
)

df_modelo = df_modelo.dropna(axis=1, how="all")

print("\nDataset después de limpieza")
print("Filas:", len(df_modelo))
print("Columnas:", len(df_modelo.columns))


# ============================================================
# 4. FILTRAR CLASES CON POCOS EJEMPLOS
# ============================================================

min_observaciones_por_clase = 3

conteo_clases = df_modelo[target].value_counts()
clases_validas = conteo_clases[conteo_clases >= min_observaciones_por_clase].index

df_entrenable = df_modelo[df_modelo[target].isin(clases_validas)].copy()

print("\nClases originales:", df_modelo[target].nunique())
print("Clases entrenables:", df_entrenable[target].nunique())
print("Filas entrenables:", len(df_entrenable))

if len(df_entrenable) == 0:
    raise ValueError("No quedaron registros entrenables.")

if df_entrenable[target].nunique() < 2:
    raise ValueError("Solo quedó una clase entrenable.")


# ============================================================
# 5. VARIABLES DEL MODELO
# ============================================================

variables_predictoras = [c for c in df_entrenable.columns if c != target]

print("\nVariable objetivo:")
print(target)

print("\nVariables predictoras:")
for col in variables_predictoras:
    print("-", col)

print("\nNúmero de variables predictoras:", len(variables_predictoras))


# ============================================================
# 6. TRAIN / TEST
# ============================================================

n_filas = len(df_entrenable)
n_clases = df_entrenable[target].nunique()

# Garantiza que el test tenga espacio suficiente para las clases
test_size = max(0.1, n_clases / n_filas)

train_df, test_df = train_test_split(
    df_entrenable,
    test_size=test_size,
    random_state=42,
    stratify=df_entrenable[target],
)

print("\nTrain:", train_df.shape)
print("Test:", test_df.shape)
print("Test size usado:", round(test_size, 4))


# ============================================================
# 7. ENTRENAR AUTOGUON
# ============================================================

predictor = TabularPredictor(
    label=target,
    problem_type="multiclass",
    eval_metric="accuracy",
    path=str(carpeta_modelo),
).fit(
    train_data=train_df,
    tuning_data=test_df,
    presets="best_quality",
    time_limit=1500,

    # Evita carpetas temporales de dynamic stacking que pueden bloquearse en Windows
    dynamic_stacking=False,
    auto_stack=False,
)


# ============================================================
# 8. EVALUAR
# ============================================================

leaderboard = predictor.leaderboard(test_df, silent=True)
metricas = predictor.evaluate(test_df)

print("\nLeaderboard:")
print(leaderboard)

print("\nMétricas:")
print(metricas)


# ============================================================
# 9. PREDICCIONES SOBRE TEST
# ============================================================

X_test = test_df.drop(columns=[target])
predicciones = predictor.predict(X_test)
probas = predictor.predict_proba(X_test)

revision_test = test_df.copy()
revision_test["prediccion"] = predicciones
revision_test["acierto"] = revision_test[target] == revision_test["prediccion"]
revision_test["confianza_prediccion"] = probas.max(axis=1).values


# ============================================================
# 10. IMPORTANCIA DE VARIABLES
# ============================================================

try:
    importancia = predictor.feature_importance(test_df)
except Exception as e:
    importancia = None
    print("No se pudo calcular importancia de variables:", e)


# ============================================================
# 11. EXPORTAR RESULTADOS
# ============================================================

ruta_leaderboard = carpeta_metricas / "leaderboard_autogluon.xlsx"
ruta_metricas = carpeta_metricas / "metricas_autogluon.json"
ruta_columnas = carpeta_metricas / "columnas_modelo.json"
ruta_predicciones_test = carpeta_metricas / "predicciones_test.xlsx"
ruta_clases = carpeta_metricas / "distribucion_clases.xlsx"
ruta_importancia = carpeta_metricas / "importancia_variables.xlsx"
ruta_modelo_actual = carpeta_metricas / "ruta_modelo_actual.txt"

leaderboard.to_excel(ruta_leaderboard, index=False)

revision_test.to_excel(ruta_predicciones_test, index=False)

conteo_clases.to_frame("n_observaciones").to_excel(
    ruta_clases
)

if importancia is not None:
    importancia.to_excel(ruta_importancia)

columnas_modelo = [c for c in train_df.columns if c != target]

with open(ruta_columnas, "w", encoding="utf-8") as f:
    json.dump(columnas_modelo, f, ensure_ascii=False, indent=4)

metricas_json = {
    str(k): float(v) if hasattr(v, "__float__") else str(v)
    for k, v in metricas.items()
}

with open(ruta_metricas, "w", encoding="utf-8") as f:
    json.dump(metricas_json, f, ensure_ascii=False, indent=4)

with open(ruta_modelo_actual, "w", encoding="utf-8") as f:
    f.write(str(carpeta_modelo))


# ============================================================
# 12. RESUMEN FINAL
# ============================================================

print("\nEntrenamiento finalizado.")
print("Modelo guardado en:", carpeta_modelo)
print("Métricas guardadas en:", carpeta_metricas)
print("Leaderboard:", ruta_leaderboard)
print("Métricas JSON:", ruta_metricas)
print("Columnas modelo:", ruta_columnas)
print("Predicciones test:", ruta_predicciones_test)
print("Distribución clases:", ruta_clases)
print("Ruta modelo actual:", ruta_modelo_actual)