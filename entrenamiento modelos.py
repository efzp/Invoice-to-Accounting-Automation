# entrenamiento_autogluon_contabilidad.py

from pathlib import Path
from datetime import datetime
import json

import pandas as pd
from autogluon.tabular import TabularPredictor


# =========================================================
# CONFIGURACIÓN
# =========================================================

base = Path(r"C:/Users/eduar/OneDrive - TCP BAAS S.A.S/Automatización/facturas_bigquery")
carpeta_salida = base / "resultados"
carpeta_metricas = carpeta_salida / "metricas_modelo"
carpeta_metricas.mkdir(parents=True, exist_ok=True)

ruta_dataset = carpeta_salida / "dataset_autogluon.xlsx"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
carpeta_modelo = (
    Path(r"C:/modelos_autogluon")
    / f"modelo_autogluon_contabilidad_facturas_{timestamp}"
)
carpeta_modelo.mkdir(parents=True, exist_ok=True)

rutas = {
    "leaderboard": carpeta_metricas / "leaderboard_autogluon.xlsx",
    "resumen": carpeta_metricas / "resumen_entrenamiento_autogluon.json",
    "columnas": carpeta_metricas / "columnas_modelo.json",
    "predicciones_training": (
        carpeta_metricas / "predicciones_training_referencia.xlsx"
    ),
    "clases": carpeta_metricas / "distribucion_clases.xlsx",
    "importancia": carpeta_metricas / "importancia_variables.xlsx",
    "modelo_actual": carpeta_metricas / "ruta_modelo_actual.txt",
}


# =========================================================
# OPCIONES
# =========================================================

target = "target_plantilla_cuentas"
min_observaciones_por_clase = 3
aplicar_oversampling_reciente = True
columna_fecha_oversampling = "fecha_emision"
meses_recientes_oversampling = 4
factor_oversampling_reciente = 4
autogluon_presets = "best_quality"
autogluon_time_limit = 1500
autogluon_dynamic_stacking = False

columnas_excluir = [
    "llave_factura", "llave_asiento",
    "estado_match", "motivo_score", "score_total",
    "concepto_concat", "plantilla_cuentas",
    "plantilla_cuentas_dc",
]

columnas_remover_del_modelo = [
    "empresa", "nombre_proveedor_modelo",
    "item1_proveedor_modelo", "descripcion_item_1_modelo",
    "descripcion_modelo_norm", "iva_total", "inc_total",
    "tax_exclusive_amount", "tax_inclusive_amount",
    "line_extension_amount", "cantidad_items_total",
    "valor_base_sugerido", "valor_iva_sugerido",
    "valor_inc_sugerido", "valor_cxp_sugerido",
    "prefijo_factura", "anio", "mes", "trimestre",
]

columnas_categoricas_forzadas = [
    "nit_proveedor_norm", "codigo_industria_proveedor_limpio",
]


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================

def extraer_primer_codigo_industria(valor):
    if pd.isna(valor):
        return pd.NA

    texto = str(valor).strip()
    if texto == "" or texto.lower() in {"nan", "none", "<na>"}:
        return pd.NA

    partes = [p.strip() for p in texto.split(";") if p.strip() != ""]
    if not partes:
        return pd.NA

    return partes[0]


def normalizar_codigo_industria_para_modelo(
    df: pd.DataFrame,
    columna: str = "codigo_industria_proveedor_limpio",
) -> pd.DataFrame:
    if columna not in df.columns:
        return df

    df[columna] = df[columna].apply(extraer_primer_codigo_industria)
    return df


def forzar_columna_a_categoria(df: pd.DataFrame, columna: str) -> pd.DataFrame:
    if columna not in df.columns:
        return df

    df[columna] = (
        df[columna]
        .astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
        .astype("category")
    )
    return df


def imprimir_tabla(titulo: str, df: pd.DataFrame, max_filas: int | None = None) -> None:
    print(f"\n{titulo}")
    if df is None or df.empty:
        print("(sin datos)")
        return

    salida = df.copy()
    if max_filas is not None:
        salida = salida.head(max_filas)

    print(salida.to_string(index=False))


def resumen_dataset(df_modelo: pd.DataFrame, df_entrenable: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        {"indicador": "filas_modelo_original", "valor": len(df_modelo)},
        {"indicador": "filas_entrenables", "valor": len(df_entrenable)},
        {
            "indicador": "columnas_modelo_original",
            "valor": len(df_modelo.columns),
        },
        {"indicador": "variables_predictoras", "valor": len([c for c in df_entrenable.columns if c != target])},
        {"indicador": "clases_originales", "valor": df_modelo[target].nunique()},
        {"indicador": "clases_entrenables", "valor": df_entrenable[target].nunique()},
    ])


def aplicar_oversampling_por_recencia(
    df: pd.DataFrame,
    columna_fecha: str,
    meses_recientes: int,
    factor: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if factor <= 1 or columna_fecha not in df.columns:
        resumen = pd.DataFrame([{
            "oversampling_activo": False,
            "motivo": "sin_columna_fecha_o_factor_1",
            "filas_antes": len(df),
            "filas_recientes": 0,
            "filas_despues": len(df),
        }])
        return df.copy(), resumen

    salida = df.copy()
    fechas = pd.to_datetime(salida[columna_fecha], errors="coerce")

    if fechas.notna().sum() == 0:
        resumen = pd.DataFrame([{
            "oversampling_activo": False,
            "motivo": "fechas_invalidas",
            "filas_antes": len(df),
            "filas_recientes": 0,
            "filas_despues": len(df),
        }])
        return salida, resumen

    fecha_corte = fechas.max() - pd.DateOffset(months=meses_recientes)
    mask_reciente = fechas >= fecha_corte
    recientes = salida.loc[mask_reciente].copy()

    if recientes.empty:
        resumen = pd.DataFrame([{
            "oversampling_activo": False,
            "motivo": "sin_filas_recientes",
            "fecha_maxima": str(fechas.max().date()),
            "fecha_corte": str(fecha_corte.date()),
            "filas_antes": len(df),
            "filas_recientes": 0,
            "filas_despues": len(df),
        }])
        return salida, resumen

    replicas = [salida] + [recientes.copy() for _ in range(factor - 1)]
    salida = pd.concat(replicas, ignore_index=True)

    resumen = pd.DataFrame([{
        "oversampling_activo": True,
        "motivo": "ok",
        "fecha_maxima": str(fechas.max().date()),
        "fecha_corte": str(fecha_corte.date()),
        "meses_recientes": meses_recientes,
        "factor": factor,
        "filas_antes": len(df),
        "filas_recientes": len(recientes),
        "filas_despues": len(salida),
    }])
    return salida, resumen


def resumen_categoricas(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    filas = []
    for col in columnas:
        if col in df.columns:
            filas.append({
                "columna": col,
                "dtype": str(df[col].dtype),
                "nulos": int(df[col].isna().sum()),
                "categorias": int(df[col].nunique(dropna=True)),
            })
        else:
            filas.append({
                "columna": col,
                "dtype": "no_existe",
                "nulos": None,
                "categorias": None,
            })
    return pd.DataFrame(filas)


def resumen_predictoras(df: pd.DataFrame, target: str) -> pd.DataFrame:
    filas = []
    for col in df.columns:
        if col == target:
            continue
        filas.append({
            "columna": col,
            "dtype": str(df[col].dtype),
            "nulos": int(df[col].isna().sum()),
            "unicos": int(df[col].nunique(dropna=True)),
        })
    return pd.DataFrame(filas).sort_values(["dtype", "columna"]).reset_index(drop=True)


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

df_modelo = (
    df_modelo
    .dropna(subset=[target])
    .drop(
        columns=[
            c for c in (columnas_excluir + columnas_remover_del_modelo)
            if c in df_modelo.columns
        ],
        errors="ignore",
    )
    .dropna(axis=1, how="all")
    .copy()
)

df_modelo = normalizar_codigo_industria_para_modelo(
    df_modelo,
    "codigo_industria_proveedor_limpio",
)

df_modelo[target] = df_modelo[target].astype(str)

for col in columnas_categoricas_forzadas:
    df_modelo = forzar_columna_a_categoria(df_modelo, col)

conteo_clases = df_modelo[target].value_counts()
clases_validas = conteo_clases[conteo_clases >= min_observaciones_por_clase].index
df_entrenable = df_modelo[df_modelo[target].isin(clases_validas)].copy()

if df_entrenable.empty:
    raise ValueError("No quedaron registros entrenables.")

if df_entrenable[target].nunique() < 2:
    raise ValueError("Solo quedó una clase entrenable.")

resumen_oversampling = pd.DataFrame([{
    "oversampling_activo": False,
    "motivo": "desactivado",
    "filas_antes": len(df_entrenable),
    "filas_recientes": 0,
    "filas_despues": len(df_entrenable),
}])

if aplicar_oversampling_reciente:
    df_entrenable, resumen_oversampling = aplicar_oversampling_por_recencia(
        df_entrenable,
        columna_fecha=columna_fecha_oversampling,
        meses_recientes=meses_recientes_oversampling,
        factor=factor_oversampling_reciente,
    )

df_entrenable = df_entrenable.drop(
    columns=[columna_fecha_oversampling],
    errors="ignore",
)

columnas_modelo = [c for c in df_entrenable.columns if c != target]

imprimir_tabla(
    "RESUMEN DATASET",
    resumen_dataset(df_modelo, df_entrenable)
)

imprimir_tabla(
    "COLUMNAS CATEGÓRICAS FORZADAS",
    resumen_categoricas(df_entrenable, columnas_categoricas_forzadas)
)

imprimir_tabla(
    "VARIABLE OBJETIVO",
    pd.DataFrame([{"target": target}])
)

imprimir_tabla("OVERSAMPLING RECIENTE", resumen_oversampling)

imprimir_tabla(
    "VARIABLES PREDICTORAS",
    resumen_predictoras(df_entrenable, target)
)


# =========================================================
# AUTOGLOUON
# =========================================================

predictor = TabularPredictor(
    label=target,
    problem_type="multiclass",
    eval_metric="accuracy",
    path=str(carpeta_modelo),
).fit(
    train_data=df_entrenable,
    presets=autogluon_presets,
    time_limit=autogluon_time_limit,
    dynamic_stacking=autogluon_dynamic_stacking,
)

leaderboard = predictor.leaderboard(silent=True)

imprimir_tabla("LEADERBOARD INTERNO AUTOGLOUON", leaderboard)


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
    print(f"\nIMPORTANCIA VARIABLES\nNo se pudo calcular importancia de variables: {e}")

resumen_training = pd.DataFrame([
    {
        "accuracy_training_referencia": float(revision_training["acierto"].mean()),
        "n_registros": len(revision_training),
        "n_aciertos": int(revision_training["acierto"].sum()),
        "confianza_promedio": float(revision_training["confianza_prediccion"].mean()),
    }
])

imprimir_tabla("RESUMEN TRAINING REFERENCIA", resumen_training)

if importancia is not None:
    importancia_reset = importancia.reset_index().rename(columns={"index": "variable"})
    imprimir_tabla("IMPORTANCIA DE VARIABLES", importancia_reset, max_filas=30)


# =========================================================
# EXPORTACIÓN
# =========================================================

leaderboard.to_excel(rutas["leaderboard"], index=False)
revision_training.to_excel(
    rutas["predicciones_training"],
    index=False,
)
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
    "oversampling_reciente": resumen_oversampling.iloc[0].to_dict(),
    "clases_originales": int(df_modelo[target].nunique()),
    "clases_entrenables": int(df_entrenable[target].nunique()),
    "min_observaciones_por_clase": int(min_observaciones_por_clase),
    "n_variables_predictoras": int(len(columnas_modelo)),
    "columnas_categoricas_forzadas": [
        c for c in columnas_categoricas_forzadas if c in df_entrenable.columns
    ],
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

imprimir_tabla(
    "ARCHIVOS GENERADOS",
    pd.DataFrame([
        {"archivo": k, "ruta": str(v)}
        for k, v in rutas.items()
        if k != "importancia" or importancia is not None
    ])
)

if score_val is not None:
    imprimir_tabla(
        "RESUMEN FINAL",
        pd.DataFrame([{
            "mejor_modelo": mejor_modelo,
            "score_validacion_interna": float(score_val),
            "ruta_modelo": str(carpeta_modelo),
        }])
    )
else:
    imprimir_tabla(
        "RESUMEN FINAL",
        pd.DataFrame([{
            "mejor_modelo": mejor_modelo,
            "score_validacion_interna": None,
            "ruta_modelo": str(carpeta_modelo),
        }])
    )
