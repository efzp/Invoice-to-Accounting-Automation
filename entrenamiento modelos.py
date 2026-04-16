# entrenamiento_autogluon_contabilidad.py

from pathlib import Path
from datetime import datetime
import json

import pandas as pd
from autogluon.tabular import TabularPredictor


# =========================================================
# ENTRENAMIENTO
# =========================================================

base = Path(r"C:/Users/eduar/OneDrive - TCP BAAS S.A.S/Automatización/facturas_bigquery")
carpeta_salida = base / "resultados"
carpeta_metricas = carpeta_salida / "metricas_modelo"
carpeta_metricas.mkdir(parents=True, exist_ok=True)

ruta_dataset = carpeta_salida / "dataset_autogluon.xlsx"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
carpeta_modelo = Path(r"C:/modelos_autogluon") / f"modelo_autogluon_contabilidad_facturas_{timestamp}"
carpeta_modelo.mkdir(parents=True, exist_ok=True)

target = "target_plantilla_cuentas"
min_observaciones_por_clase = 3

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


# =========================================================
# CARGA Y LIMPIEZA
# =========================================================

if not ruta_dataset.exists():
    raise FileNotFoundError(
        f"No existe el archivo:\n{ruta_dataset}\n\n"
        "Ejecuta primero el pipeline para generar dataset_autogluon.xlsx."
    )

df_modelo = pd.read_excel(ruta_dataset)

if target not in df_modelo.columns:
    raise ValueError(f"No existe la columna objetivo: {target}")

print("Dataset cargado")
print("Ruta:", ruta_dataset)
print("Filas:", len(df_modelo))
print("Columnas:", len(df_modelo.columns))

df_modelo = (
    df_modelo
    .dropna(subset=[target])
    .drop(columns=[c for c in columnas_excluir if c in df_modelo.columns], errors="ignore")
    .dropna(axis=1, how="all")
    .copy()
)

df_modelo[target] = df_modelo[target].astype(str)

conteo_clases = df_modelo[target].value_counts()
clases_validas = conteo_clases[conteo_clases >= min_observaciones_por_clase].index
df_entrenable = df_modelo[df_modelo[target].isin(clases_validas)].copy()

if df_entrenable.empty:
    raise ValueError("No quedaron registros entrenables.")

if df_entrenable[target].nunique() < 2:
    raise ValueError("Solo quedó una clase entrenable.")

columnas_modelo = [c for c in df_entrenable.columns if c != target]

print("\nDataset después de limpieza")
print("Filas modelo original:", len(df_modelo))
print("Filas entrenables:", len(df_entrenable))
print("Clases originales:", df_modelo[target].nunique())
print("Clases entrenables:", df_entrenable[target].nunique())
print("Variables predictoras:", len(columnas_modelo))

print("\nVariable objetivo:")
print(target)

print("\nVariables predictoras:")
for col in columnas_modelo:
    print("-", col)


# =========================================================
# AUTOGUON
# =========================================================

predictor = TabularPredictor(
    label=target,
    problem_type="multiclass",
    eval_metric="accuracy",
    path=str(carpeta_modelo),
).fit(
    train_data=df_entrenable,
    presets="best_quality",
    time_limit=1500,
    dynamic_stacking=False,
)

leaderboard = predictor.leaderboard(silent=True)

print("\nLeaderboard interno AutoGluon:")
print(leaderboard)


# =========================================================
# DIAGNÓSTICOS INTERNOS
# =========================================================

X_train_ref = df_entrenable.drop(columns=[target])

predicciones_train = predictor.predict(X_train_ref)
probas_train = predictor.predict_proba(X_train_ref)

revision_training = df_entrenable.copy()
revision_training["prediccion"] = predicciones_train
revision_training["acierto"] = revision_training[target] == revision_training["prediccion"]
revision_training["confianza_prediccion"] = probas_train.max(axis=1).values

try:
    importancia = predictor.feature_importance(df_entrenable)
except Exception as e:
    importancia = None
    print("\nNo se pudo calcular importancia de variables:", e)


# =========================================================
# EXPORTACIÓN
# =========================================================

rutas = {
    "leaderboard": carpeta_metricas / "leaderboard_autogluon.xlsx",
    "resumen": carpeta_metricas / "resumen_entrenamiento_autogluon.json",
    "columnas": carpeta_metricas / "columnas_modelo.json",
    "predicciones_training": carpeta_metricas / "predicciones_training_referencia.xlsx",
    "clases": carpeta_metricas / "distribucion_clases.xlsx",
    "importancia": carpeta_metricas / "importancia_variables.xlsx",
    "modelo_actual": carpeta_metricas / "ruta_modelo_actual.txt",
}

leaderboard.to_excel(rutas["leaderboard"], index=False)
revision_training.to_excel(rutas["predicciones_training"], index=False)
conteo_clases.to_frame("n_observaciones").to_excel(rutas["clases"])

if importancia is not None:
    importancia.to_excel(rutas["importancia"])

with open(rutas["columnas"], "w", encoding="utf-8") as f:
    json.dump(columnas_modelo, f, ensure_ascii=False, indent=4)

mejor_modelo = None
score_val = None

if not leaderboard.empty:
    mejor_modelo = str(leaderboard.iloc[0].get("model", ""))

    if "score_val" in leaderboard.columns:
        score_val = leaderboard.iloc[0].get("score_val")
    elif "score_test" in leaderboard.columns:
        score_val = leaderboard.iloc[0].get("score_test")

resumen_entrenamiento = {
    "ruta_dataset": str(ruta_dataset),
    "ruta_modelo": str(carpeta_modelo),
    "target": target,
    "filas_modelo_original": int(len(df_modelo)),
    "filas_entrenables": int(len(df_entrenable)),
    "clases_originales": int(df_modelo[target].nunique()),
    "clases_entrenables": int(df_entrenable[target].nunique()),
    "min_observaciones_por_clase": int(min_observaciones_por_clase),
    "n_variables_predictoras": int(len(columnas_modelo)),
    "mejor_modelo_leaderboard": mejor_modelo,
    "score_validacion_interna": (
        float(score_val) if score_val is not None and pd.notna(score_val) else None
    ),
    "nota": (
        "No se hizo train_test_split. El score corresponde al leaderboard interno "
        "de AutoGluon, no a una prueba externa. La validación real del MVP se hará "
        "con un dataset nuevo revisado por usuario."
    ),
}

with open(rutas["resumen"], "w", encoding="utf-8") as f:
    json.dump(resumen_entrenamiento, f, ensure_ascii=False, indent=4)

with open(rutas["modelo_actual"], "w", encoding="utf-8") as f:
    f.write(str(carpeta_modelo))


# =========================================================
# SALIDA FINAL
# =========================================================

print("\nEntrenamiento finalizado.")
print("Modelo guardado en:", carpeta_modelo)
print("Métricas guardadas en:", carpeta_metricas)
print("Leaderboard interno:", rutas["leaderboard"])
print("Resumen entrenamiento:", rutas["resumen"])
print("Columnas modelo:", rutas["columnas"])
print("Predicciones training referencia:", rutas["predicciones_training"])
print("Distribución clases:", rutas["clases"])
print("Ruta modelo actual:", rutas["modelo_actual"])

if score_val is not None:
    print("Score validación interna mejor modelo:", score_val)