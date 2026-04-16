# predictor_facturas_contabilidad.py

from pathlib import Path
import json
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

from procesamiento_facturas_contabilidad_match import procesar_facturas


# =========================================================
# PREDICTOR
# =========================================================

base = Path(r"C:/Users/eduar/OneDrive - TCP BAAS S.A.S/Automatización/facturas_bigquery")
carpeta_salida = base / "resultados"
carpeta_metricas = carpeta_salida / "metricas_modelo"
carpeta_salida.mkdir(parents=True, exist_ok=True)

ruta_facturas_nuevas = base / "facturas_a_predecir.xlsx"
ruta_modelo_actual = carpeta_metricas / "ruta_modelo_actual.txt"
ruta_columnas_modelo = carpeta_metricas / "columnas_modelo.json"

rutas_posibles_ratios = [
    carpeta_salida / "perfil_ratios_valores.xlsx",
    carpeta_salida / "10_perfil_ratios_valores.xlsx",
    carpeta_salida / "07_perfil_ratios_valores.xlsx",
]

ruta_salida_predicciones = carpeta_salida / "prueba_usuario_predicciones_facturas.xlsx"
ruta_salida_lineas = carpeta_salida / "prueba_usuario_lineas_contables_sugeridas.xlsx"
ruta_salida_control = carpeta_salida / "prueba_usuario_control_cuadre.xlsx"

col_nit = "nit_proveedor_norm"
col_target = "target_plantilla_cuentas"
col_prediccion = "plantilla_predicha"
col_total = "payable_amount"
min_observaciones = 2


# =========================================================
# FUNCIONES
# =========================================================

def validar_archivo(ruta: Path, mensaje: str) -> None:
    if not ruta.exists():
        raise FileNotFoundError(f"{mensaje}\n{ruta}")


def cargar_ruta_modelo() -> Path:
    validar_archivo(
        ruta_modelo_actual,
        "No existe ruta_modelo_actual.txt. Ejecuta primero entrenamiento_autogluon_contabilidad.py.",
    )

    ruta_modelo = Path(ruta_modelo_actual.read_text(encoding="utf-8").strip())

    if not ruta_modelo.exists():
        raise FileNotFoundError(
            f"La carpeta del modelo no existe:\n{ruta_modelo}\n\n"
            "Revisa ruta_modelo_actual.txt o vuelve a entrenar el modelo."
        )

    return ruta_modelo


def cargar_columnas_modelo() -> list[str]:
    validar_archivo(
        ruta_columnas_modelo,
        "No existe columnas_modelo.json. Ejecuta primero entrenamiento_autogluon_contabilidad.py.",
    )

    with open(ruta_columnas_modelo, "r", encoding="utf-8") as f:
        return json.load(f)


def obtener_ruta_ratios() -> Path | None:
    for ruta in rutas_posibles_ratios:
        if ruta.exists():
            return ruta
    return None


def texto_llave(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    df = df.copy()

    for col in columnas:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("").str.strip()

    return df


def numero_seguro(serie: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(serie):
        return pd.to_numeric(serie, errors="coerce").fillna(0)

    return (
        serie.astype("string")
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace("", np.nan)
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0)
    )


def separar_cuenta_naturaleza(valor):
    if pd.isna(valor):
        return None, None

    texto = str(valor).strip()

    if "_" not in texto:
        return texto, None

    cuenta, naturaleza = texto.rsplit("_", 1)
    return cuenta, naturaleza


def preparar_X(facturas: pd.DataFrame, columnas_modelo: list[str]) -> pd.DataFrame:
    X = facturas.copy()

    for col in columnas_modelo:
        if col not in X.columns:
            X[col] = None

    return X[columnas_modelo].copy()


def predecir_facturas(
    facturas: pd.DataFrame,
    predictor: TabularPredictor,
    columnas_modelo: list[str],
) -> pd.DataFrame:
    X = preparar_X(facturas, columnas_modelo)

    salida = facturas.copy()
    salida[col_prediccion] = predictor.predict(X).astype("string")

    try:
        probas = predictor.predict_proba(X)
        salida["confianza_prediccion"] = probas.max(axis=1).values
    except Exception as e:
        print("No se pudo calcular predict_proba:", e)
        salida["confianza_prediccion"] = np.nan

    salida["estado_sugerencia"] = np.select(
        [
            salida["confianza_prediccion"] >= 0.80,
            salida["confianza_prediccion"] >= 0.50,
        ],
        [
            "SUGERENCIA_ALTA",
            "REVISION_RAPIDA",
        ],
        default="REVISION_MANUAL",
    )

    return salida


def construir_lineas_desde_ratios(
    facturas_predichas: pd.DataFrame,
    perfil_ratios: pd.DataFrame,
) -> pd.DataFrame:
    facturas = facturas_predichas.copy()
    perfil = perfil_ratios.copy()

    columnas_requeridas = [
        col_nit,
        col_target,
        "cuenta_naturaleza",
        "ratio_mediano",
    ]

    faltantes = [c for c in columnas_requeridas if c not in perfil.columns]

    if faltantes:
        raise ValueError(
            "El perfil de ratios no tiene las columnas requeridas:\n"
            + "\n".join(faltantes)
        )

    if col_nit not in facturas.columns:
        raise ValueError(f"Las facturas procesadas no tienen la columna {col_nit}.")

    if col_total not in facturas.columns:
        raise ValueError(f"Las facturas procesadas no tienen la columna {col_total}.")

    if "n_observaciones" in perfil.columns:
        perfil["n_observaciones"] = numero_seguro(perfil["n_observaciones"])
        perfil = perfil[perfil["n_observaciones"] >= min_observaciones].copy()

    facturas[col_target] = facturas[col_prediccion]

    llaves = [col_nit, col_target]

    facturas = texto_llave(facturas, llaves)
    perfil = texto_llave(perfil, llaves)

    lineas = facturas.merge(
        perfil,
        on=llaves,
        how="left",
        suffixes=("", "_ratio"),
    )

    lineas["valor_base_calculo"] = numero_seguro(lineas[col_total])
    lineas["ratio_mediano"] = numero_seguro(lineas["ratio_mediano"])
    lineas["valor_sugerido"] = lineas["valor_base_calculo"] * lineas["ratio_mediano"]

    lineas[["cuenta_contable", "naturaleza"]] = lineas["cuenta_naturaleza"].apply(
        lambda x: pd.Series(separar_cuenta_naturaleza(x))
    )

    lineas["debito"] = np.where(lineas["naturaleza"] == "D", lineas["valor_sugerido"], 0)
    lineas["credito"] = np.where(lineas["naturaleza"] == "C", lineas["valor_sugerido"], 0)

    lineas["tiene_ratio_historico"] = lineas["cuenta_naturaleza"].notna()

    lineas["alerta_prediccion"] = np.where(
        lineas["tiene_ratio_historico"],
        "",
        "Sin perfil histórico suficiente para construir líneas por ratio",
    )

    columnas_salida = [
        "id_carga",
        "archivo_xml",
        "id_factura",
        "factura_completa",
        "factura_match_norm",
        col_nit,
        "nombre_proveedor",
        col_prediccion,
        "confianza_prediccion",
        "estado_sugerencia",
        "cuenta_contable",
        "naturaleza",
        "cuenta_naturaleza",
        "ratio_mediano",
        "ratio_promedio",
        "ratio_std",
        "n_observaciones",
        "valor_base_calculo",
        "valor_sugerido",
        "debito",
        "credito",
        "tiene_ratio_historico",
        "alerta_prediccion",
    ]

    columnas_salida = [c for c in columnas_salida if c in lineas.columns]

    return lineas[columnas_salida].copy()


def construir_control_cuadre(lineas: pd.DataFrame) -> pd.DataFrame:
    control = (
        lineas.groupby("id_factura", dropna=False)
        .agg(
            total_debito=("debito", "sum"),
            total_credito=("credito", "sum"),
            n_lineas=("cuenta_contable", "count"),
        )
        .reset_index()
    )

    control["diferencia"] = (control["total_debito"] - control["total_credito"]).round(2)
    control["cuadra"] = control["diferencia"].abs() <= 1

    return control


# =========================================================
# EJECUCIÓN
# =========================================================

validar_archivo(
    ruta_facturas_nuevas,
    "No existe el archivo de facturas nuevas. Debe tener la misma estructura que base datos factura.xlsx.",
)

ruta_modelo = cargar_ruta_modelo()
columnas_modelo = cargar_columnas_modelo()

print("Cargando modelo:")
print(ruta_modelo)

predictor = TabularPredictor.load(str(ruta_modelo))

facturas_nuevas = procesar_facturas(
    ruta_excel=str(ruta_facturas_nuevas),
    hoja=0,
    empresa="cpabaas",
)

print("\nFacturas nuevas procesadas:", facturas_nuevas.shape)
print("Columnas modelo cargadas:", len(columnas_modelo))

facturas_predichas = predecir_facturas(
    facturas=facturas_nuevas,
    predictor=predictor,
    columnas_modelo=columnas_modelo,
)

facturas_predichas.to_excel(ruta_salida_predicciones, index=False)

print("\nPredicciones exportadas en:")
print(ruta_salida_predicciones)

print("\nDistribución de sugerencias:")
print(facturas_predichas["estado_sugerencia"].value_counts(dropna=False))

print("\nDistribución de plantillas predichas:")
print(facturas_predichas[col_prediccion].value_counts(dropna=False).head(20))


# =========================================================
# RATIOS Y CONTROL
# =========================================================

ruta_ratios = obtener_ruta_ratios()

if ruta_ratios is None:
    lineas_sugeridas = pd.DataFrame()
    print("\nNo se encontró perfil de ratios. Solo se exportaron las predicciones.")

else:
    print("\nPerfil de ratios cargado desde:")
    print(ruta_ratios)

    perfil_ratios = pd.read_excel(ruta_ratios, dtype="object")

    lineas_sugeridas = construir_lineas_desde_ratios(
        facturas_predichas=facturas_predichas,
        perfil_ratios=perfil_ratios,
    )

    lineas_sugeridas.to_excel(ruta_salida_lineas, index=False)

    print("\nLíneas contables sugeridas exportadas en:")
    print(ruta_salida_lineas)
    print("Líneas sugeridas:", len(lineas_sugeridas))
    print("Líneas con ratio histórico:", int(lineas_sugeridas["tiene_ratio_historico"].sum()))
    print("Líneas sin ratio histórico:", int((~lineas_sugeridas["tiene_ratio_historico"]).sum()))

    if not lineas_sugeridas.empty:
        control_cuadre = construir_control_cuadre(lineas_sugeridas)
        control_cuadre.to_excel(ruta_salida_control, index=False)

        print("\nControl de cuadre exportado en:")
        print(ruta_salida_control)

        print("\nResumen control de cuadre:")
        print(control_cuadre["cuadra"].value_counts(dropna=False))


# =========================================================
# RESUMEN FINAL
# =========================================================

print("\nPredicción finalizada.")
print("Modelo usado:", ruta_modelo)
print("Archivo de facturas nuevas:", ruta_facturas_nuevas)
print("Facturas predichas:", len(facturas_predichas))
print("Archivo de predicciones:", ruta_salida_predicciones)

if not lineas_sugeridas.empty:
    print("Archivo de líneas:", ruta_salida_lineas)
    print("Archivo de control:", ruta_salida_control)
else:
    print("Líneas sugeridas: 0")