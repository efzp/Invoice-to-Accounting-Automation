# validar_cobertura_prueba_y_llaves_nit_cuenta.py

from pathlib import Path
import pandas as pd

from procesamiento_facturas_contabilidad_match import cargar_excel, procesar_facturas
from funciones_de_limpieza_base_datos_factura import (
    normalizar_nit,
    normalizar_alfanumerico,
    texto_contiene_factura,
)


# =========================================================
# RUTAS
# =========================================================

base = Path(r"C:/Users/eduar/OneDrive - TCP BAAS S.A.S/Automatización/facturas_bigquery")

ruta_facturas_esperadas = base / "facturas_a_predecir.xlsx"
ruta_prueba = base / "prueba de facturacion.xlsx"
ruta_plantilla = base / "resultados" / "plantilla.xlsx"
ruta_salida = base / "resultados" / "validacion_cobertura_prueba_y_llaves.xlsx"

hoja_prueba = 0
hoja_plantilla = "plantilla"
solo_fc = True


# =========================================================
# FUNCIONES
# =========================================================

def validar_archivo(ruta: Path, mensaje: str) -> None:
    if not ruta.exists():
        raise FileNotFoundError(f"{mensaje}\n{ruta}")


def texto_serie(df: pd.DataFrame, col: str) -> pd.Series:
    return (
        df[col].astype("string").fillna("").str.strip()
        if col in df.columns
        else pd.Series([""] * len(df), index=df.index, dtype="string")
    )


def normalizar_cuenta(serie: pd.Series) -> pd.Series:
    return (
        serie.astype("string")
        .fillna("")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def preparar_facturas_esperadas(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    if "nit_proveedor_norm" not in d.columns or "factura_match_norm" not in d.columns:
        raise ValueError("Las facturas esperadas no tienen nit_proveedor_norm o factura_match_norm.")

    d = d.loc[d["nit_proveedor_norm"].notna() & d["factura_match_norm"].notna()].copy()

    d["nit"] = d["nit_proveedor_norm"].astype("string").fillna("").str.strip()
    d["factura_norm"] = d["factura_match_norm"].astype("string").fillna("").str.strip()
    d["llave_factura"] = d["nit"] + "|" + d["factura_norm"]

    columnas = [
        "llave_factura",
        "nit",
        "factura_norm",
        "id_factura",
        "factura_completa",
        "nombre_proveedor",
    ]
    columnas = [c for c in columnas if c in d.columns]

    return d[columnas].drop_duplicates().reset_index(drop=True)


def preparar_prueba(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    if solo_fc and "tipodoc" in d.columns:
        d = d.loc[texto_serie(d, "tipodoc").str.upper() == "FC"].copy()

    if "identidadtercero" not in d.columns or "cuenta" not in d.columns or "concepto" not in d.columns:
        raise ValueError("La prueba debe tener columnas IDENTIDADTERCERO, CUENTA y CONCEPTO.")

    d["nit"] = d["identidadtercero"].apply(normalizar_nit)
    d["cuenta"] = normalizar_cuenta(d["cuenta"])
    d["concepto_norm"] = d["concepto"].apply(normalizar_alfanumerico)

    d = d.loc[d["nit"].notna() & d["cuenta"].ne("")].copy()
    d["llave_nit_cuenta"] = d["nit"].fillna("") + "|" + d["cuenta"]

    columnas = ["nit", "cuenta", "concepto", "concepto_norm", "llave_nit_cuenta"]
    columnas = [c for c in columnas if c in d.columns]

    return d[columnas].reset_index(drop=True)


def preparar_plantilla(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    if solo_fc and "tipo_documento" in d.columns:
        d = d.loc[texto_serie(d, "tipo_documento").str.upper() == "FC"].copy()

    if "identidad" not in d.columns or "cuenta" not in d.columns:
        raise ValueError("La plantilla debe tener columnas IDENTIDAD y CUENTA.")

    d["nit"] = d["identidad"].apply(normalizar_nit)
    d["cuenta"] = normalizar_cuenta(d["cuenta"])
    d["llave_nit_cuenta"] = d["nit"].fillna("") + "|" + d["cuenta"]

    d = d.loc[d["nit"].notna() & d["cuenta"].ne("")].copy()

    return d[["nit", "cuenta", "llave_nit_cuenta"]].drop_duplicates().reset_index(drop=True)


def identificar_facturas_procesadas(
    facturas_esperadas: pd.DataFrame,
    prueba_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cand = facturas_esperadas.merge(
        prueba_df[["nit", "cuenta", "concepto", "concepto_norm", "llave_nit_cuenta"]],
        on="nit",
        how="left",
    )

    cand["match_factura"] = cand.apply(
        lambda row: texto_contiene_factura(row.get("concepto_norm"), row.get("factura_norm")),
        axis=1,
    )

    detalle_match = cand.loc[cand["match_factura"]].copy()

    facturas_procesadas = (
        detalle_match[["llave_factura", "nit", "factura_norm", "id_factura", "factura_completa", "nombre_proveedor"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    llaves_procesadas = set(facturas_procesadas["llave_factura"].astype("string"))
    facturas_no_procesadas = facturas_esperadas.loc[
        ~facturas_esperadas["llave_factura"].astype("string").isin(llaves_procesadas)
    ].copy().reset_index(drop=True)

    return facturas_procesadas, facturas_no_procesadas, detalle_match.reset_index(drop=True)


def construir_llaves_prueba_procesada(
    detalle_match: pd.DataFrame,
) -> pd.DataFrame:
    if detalle_match.empty:
        return pd.DataFrame(columns=["nit", "cuenta", "llave_nit_cuenta"])

    return (
        detalle_match[["nit", "cuenta", "llave_nit_cuenta"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )


def construir_resumen(
    facturas_esperadas: pd.DataFrame,
    facturas_procesadas: pd.DataFrame,
    facturas_no_procesadas: pd.DataFrame,
    llaves_prueba: pd.DataFrame,
    llaves_plantilla: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparacion_llaves = llaves_prueba.merge(
        llaves_plantilla[["llave_nit_cuenta"]].assign(generada_en_plantilla=True),
        on="llave_nit_cuenta",
        how="left",
    )

    comparacion_llaves["generada_en_plantilla"] = comparacion_llaves["generada_en_plantilla"].fillna(False)

    total_facturas_esperadas = len(facturas_esperadas)
    total_facturas_procesadas = len(facturas_procesadas)
    total_facturas_no_procesadas = len(facturas_no_procesadas)
    cobertura_facturas = (
        total_facturas_procesadas / total_facturas_esperadas
        if total_facturas_esperadas > 0 else 0.0
    )

    total_llaves_prueba = len(llaves_prueba)
    llaves_generadas = int(comparacion_llaves["generada_en_plantilla"].sum())
    cobertura_llaves = llaves_generadas / total_llaves_prueba if total_llaves_prueba > 0 else 0.0

    resumen = pd.DataFrame([
        {"metrica": "facturas_esperadas", "valor": total_facturas_esperadas},
        {"metrica": "facturas_procesadas_en_prueba", "valor": total_facturas_procesadas},
        {"metrica": "facturas_no_procesadas_en_prueba", "valor": total_facturas_no_procesadas},
        {"metrica": "cobertura_facturas_procesadas", "valor": cobertura_facturas},
        {"metrica": "cobertura_facturas_procesadas_pct", "valor": f"{round(cobertura_facturas * 100, 2)}%"},
        {"metrica": "llaves_unicas_prueba_procesada", "valor": total_llaves_prueba},
        {"metrica": "llaves_unicas_plantilla", "valor": len(llaves_plantilla)},
        {"metrica": "llaves_prueba_en_plantilla", "valor": llaves_generadas},
        {"metrica": "cobertura_llaves_nit_cuenta", "valor": cobertura_llaves},
        {"metrica": "cobertura_llaves_nit_cuenta_pct", "valor": f"{round(cobertura_llaves * 100, 2)}%"},
    ])

    return resumen, comparacion_llaves.sort_values(["nit", "cuenta"]).reset_index(drop=True)


# =========================================================
# EJECUCIÓN
# =========================================================

validar_archivo(ruta_facturas_esperadas, "No existe el archivo de facturas esperadas.")
validar_archivo(ruta_prueba, "No existe el archivo de prueba de facturación.")
validar_archivo(ruta_plantilla, "No existe el archivo de plantilla.")

facturas_esperadas_raw = procesar_facturas(
    ruta_excel=str(ruta_facturas_esperadas),
    hoja=0,
    empresa="cpabaas",
)

prueba_raw = cargar_excel(str(ruta_prueba), hoja=hoja_prueba)
plantilla_raw = cargar_excel(str(ruta_plantilla), hoja=hoja_plantilla)

facturas_esperadas = preparar_facturas_esperadas(facturas_esperadas_raw)
prueba_df = preparar_prueba(prueba_raw)
plantilla_df = preparar_plantilla(plantilla_raw)

facturas_procesadas, facturas_no_procesadas, detalle_match = identificar_facturas_procesadas(
    facturas_esperadas=facturas_esperadas,
    prueba_df=prueba_df,
)

llaves_prueba = construir_llaves_prueba_procesada(detalle_match)

resumen, comparacion_llaves = construir_resumen(
    facturas_esperadas=facturas_esperadas,
    facturas_procesadas=facturas_procesadas,
    facturas_no_procesadas=facturas_no_procesadas,
    llaves_prueba=llaves_prueba,
    llaves_plantilla=plantilla_df,
)

with pd.ExcelWriter(ruta_salida, engine="openpyxl") as writer:
    resumen.to_excel(writer, sheet_name="resumen", index=False)
    facturas_procesadas.to_excel(writer, sheet_name="facturas_procesadas", index=False)
    facturas_no_procesadas.to_excel(writer, sheet_name="facturas_no_procesadas", index=False)
    detalle_match.to_excel(writer, sheet_name="detalle_match_prueba", index=False)
    llaves_prueba.to_excel(writer, sheet_name="llaves_prueba_procesada", index=False)
    plantilla_df.to_excel(writer, sheet_name="llaves_plantilla", index=False)
    comparacion_llaves.to_excel(writer, sheet_name="detalle_llaves", index=False)

print("Validación finalizada.")
print("Facturas esperadas:", ruta_facturas_esperadas)
print("Prueba facturación:", ruta_prueba)
print("Plantilla:", ruta_plantilla)
print("Salida:", ruta_salida)
print("\nResumen:")
print(resumen)