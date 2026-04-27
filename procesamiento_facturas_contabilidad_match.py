# procesamiento_facturas_contabilidad_match.py

import numpy as np
import pandas as pd
from pathlib import Path

from funciones_de_limpieza_base_datos_factura import (
    asegurar_columnas,
    coalesce_alfa_num,
    coalesce_fecha,
    construir_base_descripciones_xml,
    construir_descripcion_modelo,
    construir_llave_asiento,
    construir_llave_factura,
    detectar_prefijo_factura,
    es_nulo,
    lista_unicos_limpios,
    moda_o_primero,
    normalizar_alfanumerico,
    normalizar_flag_binaria,
    normalizar_lista_cuentas,
    normalizar_lista_cuentas_dc,
    normalizar_nit,
    normalizar_nombre_columna,
    normalizar_texto_basico,
    normalizar_texto_modelo,
    normalizar_valor_monetario,
    primer_valor_no_nulo,
    safe_to_datetime,
    texto_contiene_factura,
)

UVT_2026 = 52374
UMBRAL_RF_SERVICIOS = 2 * UVT_2026
UMBRAL_RETEICA_BOGOTA_SERVICIOS = 4 * UVT_2026
UMBRAL_RETEICA_BOGOTA_COMPRAS = 27 * UVT_2026

COLUMNAS_FACTURAS = [
    "id_carga", "id_factura", "cufe", "factura_completa",
    "fecha_emision", "nit_proveedor", "nombre_proveedor",
    "ciudad_proveedor", "tax_level_proveedor", "tax_scheme_id",
    "tax_scheme_nombre", "codigo_industria_proveedor",
    "cantidad_lineas_xml", "line_extension_amount",
    "tax_exclusive_amount", "tax_inclusive_amount", "payable_amount",
    "iva_total", "inc_total", "descuento_total", "recargo_total",
    "tiene_iva", "tiene_inc", "flag_descuento", "flag_recargo",
    "cantidad_items_total", "descripcion_item_1", "item1_proveedor",
    "n_registros_sugeridos", "valor_base_sugerido",
    "valor_iva_sugerido", "valor_inc_sugerido",
    "valor_cxp_sugerido", "observaciones",
]

COLUMNAS_MODELO_AUTOG = [
    "nit_proveedor_norm", "ciudad_proveedor_modelo",
    "tax_level_proveedor_limpio", "tax_scheme_id_limpio",
    "tax_scheme_nombre_limpio", "codigo_industria_proveedor_limpio",
    "descripcion_item_1_modelo", "cantidad_lineas_xml",
    "line_extension_amount", "tax_exclusive_amount",
    "tax_inclusive_amount", "payable_amount", "iva_total", "inc_total",
    "descuento_total", "recargo_total", "cantidad_items_total",
    "n_registros_sugeridos", "valor_base_sugerido",
    "valor_iva_sugerido", "valor_inc_sugerido", "valor_cxp_sugerido",
    "tiene_iva", "tiene_inc", "flag_descuento", "flag_recargo",
    "flag_umbral_rf_servicios", "flag_umbral_reteica_bogota_servicios",
    "flag_umbral_reteica_bogota_compras",
    "flag_diferencia_payable_tax_inclusive",
    "standard_item_identification_limpio", "descripciones_lineas_limpia_modelo",
    "target_plantilla_cuentas",
]


def cargar_excel(ruta_excel: str, hoja=0) -> pd.DataFrame:
    df = pd.read_excel(ruta_excel, sheet_name=hoja).copy()
    df.columns = [normalizar_nombre_columna(c) for c in df.columns]
    return df


def ultimos_n_caracteres(x, n: int = 6) -> str | None:
    if es_nulo(x):
        return None
    x = normalizar_alfanumerico(x)
    if es_nulo(x):
        return None
    return x[-n:] if len(x) > n else x


def extraer_primer_codigo_industria(valor):
    if es_nulo(valor):
        return None
    texto = str(valor).strip()
    if texto == "":
        return None
    partes = [p.strip() for p in texto.split(";") if str(p).strip() != ""]
    if not partes:
        return None
    return partes[0]


def resolver_carpeta_xml_facturas(
    ruta_excel: str,
    ruta_xml_descripciones: str | Path | None = None,
) -> Path | None:
    if ruta_xml_descripciones is not None:
        ruta = Path(ruta_xml_descripciones)
        return ruta if ruta.exists() else None

    ruta = Path(ruta_excel).parent / "XML definitivos"
    return ruta if ruta.exists() else None


def agregar_descripciones_xml_por_cufe(
    facturas_df: pd.DataFrame,
    carpeta_xml: str | Path | None,
) -> pd.DataFrame:
    if carpeta_xml is None:
        return facturas_df

    xml_df = construir_base_descripciones_xml(carpeta_xml, incluir_original=False)
    if xml_df.empty:
        return facturas_df

    xml_df = xml_df.copy()
    xml_df["cufe_norm"] = xml_df["cufe"].apply(normalizar_alfanumerico)
    xml_df = xml_df.drop(columns=["cufe"]).drop_duplicates("cufe_norm")

    return facturas_df.merge(
        xml_df,
        on="cufe_norm",
        how="left",
        suffixes=("", "_xml"),
    )


def score_match(fila: pd.Series) -> tuple[float, str]:
    score = 0.0
    motivos = []

    if fila.get("flag_texto_match_exacto", False):
        score += 100
        motivos.append("texto_factura")
    elif fila.get("flag_texto_match_ult6", False):
        score += 75
        motivos.append("texto_factura_ult6")
    elif fila.get("flag_texto_match_ult5", False):
        score += 60
        motivos.append("texto_factura_ult5")
    elif fila.get("flag_texto_match_ult4", False):
        score += 45
        motivos.append("texto_factura_ult4")

    nit_fact = fila.get("nit_proveedor_norm")
    nit_asi = fila.get("nit_dominante_norm")
    if not es_nulo(nit_fact) and not es_nulo(nit_asi) and nit_fact == nit_asi:
        score += 35
        motivos.append("nit")

    fecha_fact = fila.get("fecha_emision")
    fecha_asi = fila.get("fecha_asiento")
    if not pd.isna(fecha_fact) and not pd.isna(fecha_asi):
        dif_dias = abs((fecha_asi - fecha_fact).days)
        if dif_dias <= 10:
            score += 20
            motivos.append("fecha_10d")
        elif dif_dias <= 30:
            score += 10
            motivos.append("fecha_30d")
        elif dif_dias <= 45:
            score += 4
            motivos.append("fecha_45d")
        else:
            score -= min(dif_dias * 0.2, 20)

    total_fact = fila.get("total_factura")
    monto_asi = fila.get("monto_asiento")
    if not pd.isna(total_fact) and not pd.isna(monto_asi) and total_fact != 0:
        dif_valor = abs(monto_asi - total_fact)
        pct = dif_valor / abs(total_fact)
        if pct <= 0.05:
            score += 25
            motivos.append("valor_5pct")
        elif pct <= 0.08:
            score += 15
            motivos.append("valor_8pct")
        elif pct <= 0.10:
            score += 7
            motivos.append("valor_10pct")
        else:
            score -= min(pct * 20, 25)

    return score, "|".join(motivos) if motivos else "sin_evidencia"


def procesar_facturas(
    ruta_excel: str,
    hoja=0,
    empresa: str = "demo",
    ruta_xml_descripciones: str | Path | None = None,
) -> pd.DataFrame:
    df = cargar_excel(ruta_excel, hoja=hoja)
    df = asegurar_columnas(df, COLUMNAS_FACTURAS)
    df["empresa"] = empresa

    df["codigo_industria_proveedor"] = df["codigo_industria_proveedor"].apply(
        extraer_primer_codigo_industria
    )

    cols_texto_basico = [
        "id_carga", "id_factura", "cufe", "factura_completa",
        "nombre_proveedor", "ciudad_proveedor", "tax_level_proveedor",
        "tax_scheme_id", "tax_scheme_nombre",
        "codigo_industria_proveedor", "descripcion_item_1",
        "item1_proveedor", "observaciones",
    ]
    for col in cols_texto_basico:
        df[f"{col}_limpio"] = df[col].apply(normalizar_texto_basico)

    cols_texto_modelo = [
        "nombre_proveedor", "ciudad_proveedor",
        "descripcion_item_1",
    ]
    for col in cols_texto_modelo:
        df[f"{col}_modelo"] = df[col].apply(normalizar_texto_modelo)

    for col in ["id_factura", "factura_completa", "cufe"]:
        df[f"{col}_norm"] = df[col].apply(normalizar_alfanumerico)

    carpeta_xml = resolver_carpeta_xml_facturas(ruta_excel, ruta_xml_descripciones)
    df = agregar_descripciones_xml_por_cufe(df, carpeta_xml)

    for col in ["standard_item_identification", "descripciones_lineas_limpia"]:
        if col not in df.columns:
            df[col] = None

    df["standard_item_identification_limpio"] = df[
        "standard_item_identification"
    ].apply(normalizar_texto_basico)
    df["descripciones_lineas_limpia_modelo"] = df[
        "descripciones_lineas_limpia"
    ].apply(normalizar_texto_modelo)

    df["nit_proveedor_limpio"] = df["nit_proveedor"].apply(
        normalizar_texto_basico
    )
    df["nit_proveedor_norm"] = df["nit_proveedor"].apply(normalizar_nit)

    df["factura_match_norm"] = df.apply(
        lambda row: coalesce_alfa_num(
            row.get("id_factura"),
            row.get("factura_completa"),
        ),
        axis=1,
    )
    df["factura_match_ult6"] = df["factura_match_norm"].apply(
        lambda x: ultimos_n_caracteres(x, 6)
    )
    df["factura_match_ult5"] = df["factura_match_norm"].apply(
        lambda x: ultimos_n_caracteres(x, 5)
    )
    df["factura_match_ult4"] = df["factura_match_norm"].apply(
        lambda x: ultimos_n_caracteres(x, 4)
    )
    df["prefijo_factura"] = df["factura_match_norm"].apply(
        detectar_prefijo_factura
    )

    df["proveedor_nombre_limpio"] = df["nombre_proveedor_limpio"]
    df["ciudad_proveedor_limpia"] = df["ciudad_proveedor_limpio"]
    df["item_descripcion_1_limpia"] = df["descripcion_item_1_limpio"]

    df["descripcion_modelo"] = df.apply(construir_descripcion_modelo, axis=1)
    df["descripcion_modelo_norm"] = df["descripcion_modelo"].apply(
        normalizar_texto_modelo
    )

    df["fecha_emision"] = safe_to_datetime(df["fecha_emision"])
    df["anio"] = df["fecha_emision"].dt.year
    df["mes"] = df["fecha_emision"].dt.month
    df["trimestre"] = df["fecha_emision"].dt.quarter
    df["anio_mes"] = df["fecha_emision"].dt.to_period("M").astype("string")

    cols_numericas = [
        "cantidad_lineas_xml", "line_extension_amount",
        "tax_exclusive_amount", "tax_inclusive_amount",
        "payable_amount", "iva_total", "inc_total",
        "descuento_total", "recargo_total",
        "cantidad_items_total", "n_registros_sugeridos",
        "valor_base_sugerido", "valor_iva_sugerido",
        "valor_inc_sugerido", "valor_cxp_sugerido",
    ]
    for col in cols_numericas:
        df[col] = df[col].apply(normalizar_valor_monetario)

    for col in ["tiene_iva", "tiene_inc", "flag_descuento", "flag_recargo"]:
        df[col] = df[col].apply(normalizar_flag_binaria)

    df["base_amount"] = df["line_extension_amount"].fillna(0)
    df["total_factura"] = df["payable_amount"]
    df["cantidad_items"] = df["cantidad_items_total"]

    for col in [
        "iva_total", "inc_total", "descuento_total", "recargo_total",
        "payable_amount", "tax_inclusive_amount",
    ]:
        df[col] = df[col].fillna(0)

    df["tiene_iva"] = (df["iva_total"] > 0).astype(int)
    df["tiene_inc"] = (df["inc_total"] > 0).astype(int)
    df["flag_descuento"] = (
        (df["descuento_total"] > 0) | (df["flag_descuento"] == 1)
    ).astype(int)
    df["flag_recargo"] = (
        (df["recargo_total"] > 0) | (df["flag_recargo"] == 1)
    ).astype(int)

    base_ret = df["tax_exclusive_amount"].fillna(0)
    df["flag_umbral_rf_servicios"] = (
        base_ret >= UMBRAL_RF_SERVICIOS
    ).astype(int)
    df["flag_umbral_reteica_bogota_servicios"] = (
        base_ret >= UMBRAL_RETEICA_BOGOTA_SERVICIOS
    ).astype(int)
    df["flag_umbral_reteica_bogota_compras"] = (
        base_ret >= UMBRAL_RETEICA_BOGOTA_COMPRAS
    ).astype(int)

    df["total_impuestos"] = df["iva_total"] + df["inc_total"]
    df["base_mas_impuestos"] = df["base_amount"] + df["total_impuestos"]
    df["diferencia_total_vs_componentes"] = (
        df["total_factura"] - df["base_mas_impuestos"]
    )

    df["flag_diferencia_payable_tax_inclusive"] = (
        (
            df["payable_amount"].fillna(0)
            - df["tax_inclusive_amount"].fillna(0)
        ).abs() > 0.01
    ).astype(int)

    df["flag_sin_nit"] = df["nit_proveedor_norm"].isna()
    df["flag_sin_factura"] = df["factura_match_norm"].isna()
    df["flag_sin_fecha"] = df["fecha_emision"].isna()
    df["flag_sin_total"] = df["total_factura"].isna()

    df["llave_factura"] = df.apply(
        lambda row: construir_llave_factura(
            row.get("nit_proveedor_norm"),
            row.get("factura_match_norm"),
        ),
        axis=1,
    )

    return df.copy()


def procesar_movimientos(
    ruta_excel: str,
    hoja=0,
    empresa: str = "demo",
    columnas_map: dict | None = None,
) -> pd.DataFrame:
    df = cargar_excel(ruta_excel, hoja=hoja)

    columnas_default = {
        "fecha_mov": "fecha",
        "tipo_doc": "tipo_doc",
        "numero_doc": "numero_doc",
        "cuenta": "cuenta",
        "nombre_cuenta": "nombre_cuenta",
        "identidad": "identidad",
        "nombre_tercero": "nombre_del_tercero",
        "concepto": "concepto",
        "codigo_centro_costo": "c_de_costo",
        "centro_costo": "centro_de_costo",
        "usuario": "usuario",
        "numero_movil": "numero_movil",
        "nombre_centro_costo": "nombre_c_de_costo",
        "debito": "debito",
        "credito": "credito",
    }
    if columnas_map is not None:
        columnas_default.update(columnas_map)

    rename_map = {}
    for col_std, col_real in columnas_default.items():
        col_real_norm = normalizar_nombre_columna(col_real)
        if col_real_norm in df.columns:
            rename_map[col_real_norm] = col_std

    df = df.rename(columns=rename_map)
    df = asegurar_columnas(df, list(columnas_default.keys()))
    df["empresa"] = empresa

    cols_texto_basico = [
        "tipo_doc", "numero_doc", "cuenta", "nombre_cuenta",
        "identidad", "nombre_tercero", "concepto",
        "codigo_centro_costo", "centro_costo", "usuario",
        "numero_movil", "nombre_centro_costo",
    ]
    for col in cols_texto_basico:
        df[f"{col}_limpio"] = df[col].apply(normalizar_texto_basico)

    cols_texto_modelo = [
        "nombre_cuenta", "nombre_tercero",
        "concepto", "nombre_centro_costo",
    ]
    for col in cols_texto_modelo:
        df[f"{col}_modelo"] = df[col].apply(normalizar_texto_modelo)

    for col in ["tipo_doc", "numero_doc", "cuenta", "concepto"]:
        df[f"{col}_norm"] = df[col].apply(normalizar_alfanumerico)

    df["nit_tercero_norm"] = df["identidad"].apply(normalizar_nit)
    df["cuenta_limpia"] = df["cuenta_limpio"]
    df["concepto_limpio"] = df["concepto_limpio"]

    df["fecha_mov"] = safe_to_datetime(df["fecha_mov"])
    df["anio"] = df["fecha_mov"].dt.year
    df["mes"] = df["fecha_mov"].dt.month
    df["anio_mes"] = df["fecha_mov"].dt.to_period("M").astype("string")

    for col in ["debito", "credito"]:
        df[col] = df[col].apply(normalizar_valor_monetario).fillna(0)

    df["llave_asiento_base"] = df.apply(
        lambda row: construir_llave_asiento(
            row.get("anio"),
            row.get("tipo_doc_norm"),
            row.get("numero_doc_norm"),
        ),
        axis=1,
    )

    return df.copy()


def filtrar_movimientos_fc(movimientos_df: pd.DataFrame) -> pd.DataFrame:
    return movimientos_df.loc[movimientos_df["tipo_doc_norm"] == "fc"].copy()


def agrupar_asientos_fc(movimientos_fc_df: pd.DataFrame) -> pd.DataFrame:
    resumen = []

    for keys, df_asi in movimientos_fc_df.groupby(
        ["anio", "tipo_doc_norm", "numero_doc_norm"],
        dropna=False,
    ):
        anio, tipo_doc_norm, numero_doc_norm = keys
        fecha_asi = coalesce_fecha(
            primer_valor_no_nulo(df_asi["fecha_mov"]),
            df_asi["fecha_mov"].min(),
        )
        concepto = " | ".join(lista_unicos_limpios(df_asi["concepto_limpio"]))
        total_debito = df_asi["debito"].sum()
        total_credito = df_asi["credito"].sum()

        resumen.append(
            {
                "empresa": (
                    moda_o_primero(df_asi["empresa"])
                    if "empresa" in df_asi.columns
                    else None
                ),
                "anio": anio,
                "tipo_doc_norm": tipo_doc_norm,
                "numero_doc_norm": numero_doc_norm,
                "llave_asiento": construir_llave_asiento(
                    anio,
                    tipo_doc_norm,
                    numero_doc_norm,
                ),
                "fecha_asiento": fecha_asi,
                "nit_dominante_norm": moda_o_primero(
                    df_asi["nit_tercero_norm"]
                ),
                "concepto_concat": concepto,
                "concepto_concat_norm": normalizar_alfanumerico(concepto),
                "n_lineas": len(df_asi),
                "total_debito": total_debito,
                "total_credito": total_credito,
                "monto_asiento": max(total_debito, total_credito),
                "plantilla_cuentas": normalizar_lista_cuentas(df_asi),
                "plantilla_cuentas_dc": normalizar_lista_cuentas_dc(df_asi),
            }
        )

    return pd.DataFrame(resumen)


def generar_candidatos_match(
    facturas_df: pd.DataFrame,
    asientos_fc_df: pd.DataFrame,
) -> pd.DataFrame:
    fact_cols = [
        "empresa", "llave_factura", "id_factura", "factura_completa",
        "factura_match_norm", "factura_match_ult6", "factura_match_ult5",
        "factura_match_ult4", "nit_proveedor_norm", "fecha_emision",
        "total_factura", "base_amount", "iva_total", "inc_total",
    ]
    as_cols = [
        "llave_asiento", "anio", "tipo_doc_norm", "numero_doc_norm",
        "fecha_asiento", "nit_dominante_norm", "concepto_concat",
        "concepto_concat_norm", "n_lineas", "total_debito",
        "total_credito", "monto_asiento", "plantilla_cuentas",
        "plantilla_cuentas_dc",
    ]

    fact = facturas_df[[c for c in fact_cols if c in facturas_df.columns]].copy()
    asi = asientos_fc_df[[c for c in as_cols if c in asientos_fc_df.columns]].copy()

    cand = fact.merge(
        asi,
        left_on=["nit_proveedor_norm"],
        right_on=["nit_dominante_norm"],
        how="left",
    )

    cand["flag_texto_match_exacto"] = cand.apply(
        lambda row: texto_contiene_factura(
            row.get("concepto_concat_norm"),
            row.get("factura_match_norm"),
        ),
        axis=1,
    )
    cand["flag_texto_match_ult6"] = cand.apply(
        lambda row: texto_contiene_factura(
            row.get("concepto_concat_norm"),
            row.get("factura_match_ult6"),
        ),
        axis=1,
    )
    cand["flag_texto_match_ult5"] = cand.apply(
        lambda row: texto_contiene_factura(
            row.get("concepto_concat_norm"),
            row.get("factura_match_ult5"),
        ),
        axis=1,
    )
    cand["flag_texto_match_ult4"] = cand.apply(
        lambda row: texto_contiene_factura(
            row.get("concepto_concat_norm"),
            row.get("factura_match_ult4"),
        ),
        axis=1,
    )
    cand["flag_texto_match"] = (
        cand["flag_texto_match_exacto"]
        | cand["flag_texto_match_ult6"]
        | cand["flag_texto_match_ult5"]
        | cand["flag_texto_match_ult4"]
    )

    cand["dif_dias"] = np.nan
    mask_fecha = (~cand["fecha_emision"].isna()) & (~cand["fecha_asiento"].isna())
    cand.loc[mask_fecha, "dif_dias"] = (
        cand.loc[mask_fecha, "fecha_asiento"]
        - cand.loc[mask_fecha, "fecha_emision"]
    ).dt.days.abs()

    cand["dif_valor"] = np.nan
    cand["dif_valor_pct"] = np.nan
    mask_valor = (
        (~cand["total_factura"].isna())
        & (~cand["monto_asiento"].isna())
        & (cand["total_factura"] != 0)
    )
    cand.loc[mask_valor, "dif_valor"] = (
        cand.loc[mask_valor, "monto_asiento"]
        - cand.loc[mask_valor, "total_factura"]
    ).abs()
    cand.loc[mask_valor, "dif_valor_pct"] = (
        cand.loc[mask_valor, "dif_valor"]
        / cand.loc[mask_valor, "total_factura"].abs()
    )

    scores = cand.apply(score_match, axis=1)
    cand["score_total"] = [s[0] for s in scores]
    cand["motivo_score"] = [s[1] for s in scores]

    return cand


def resolver_match(candidatos_df: pd.DataFrame) -> pd.DataFrame:
    if candidatos_df.empty:
        return pd.DataFrame()

    resultados = []
    tol_multi = 0.02

    for llave_factura, df_cand in candidatos_df.groupby(
        "llave_factura",
        dropna=False,
    ):
        df_cand = df_cand.copy()

        if df_cand["llave_asiento"].isna().all():
            base = df_cand.iloc[0].copy()
            resultados.append(
                {
                    "llave_factura": llave_factura,
                    "llave_asiento": None,
                    "score_total": np.nan,
                    "estado_match": "SIN_MATCH",
                    "motivo_score": "sin_candidatos",
                    "id_factura": base.get("id_factura"),
                    "factura_match_norm": base.get("factura_match_norm"),
                    "nit_proveedor_norm": base.get("nit_proveedor_norm"),
                    "fecha_emision": base.get("fecha_emision"),
                    "total_factura": base.get("total_factura"),
                }
            )
            continue

        df_validos = df_cand.loc[~df_cand["llave_asiento"].isna()].copy()
        df_multi = df_validos.loc[
            df_validos["flag_texto_match"].fillna(False)
        ].copy()

        if len(df_multi) > 1:
            total_fact = df_multi["total_factura"].iloc[0]
            suma_asis = df_multi["monto_asiento"].sum()
            if not pd.isna(total_fact) and total_fact != 0:
                dif_pct = abs(suma_asis - total_fact) / abs(total_fact)
                if dif_pct <= tol_multi:
                    resultados.append(
                        {
                            "llave_factura": llave_factura,
                            "llave_asiento": ";".join(
                                df_multi["llave_asiento"].astype(str).tolist()
                            ),
                            "score_total": df_multi["score_total"].max(),
                            "estado_match": "REVISION_MULTIASIENTO",
                            "motivo_score": "texto_multiple|valor_sumado",
                            "id_factura": df_multi["id_factura"].iloc[0],
                            "factura_match_norm": df_multi["factura_match_norm"].iloc[0],
                            "nit_proveedor_norm": df_multi["nit_proveedor_norm"].iloc[0],
                            "fecha_emision": df_multi["fecha_emision"].iloc[0],
                            "total_factura": total_fact,
                            "fecha_asiento": df_multi["fecha_asiento"].min(),
                            "monto_asiento": suma_asis,
                            "tipo_doc_norm": df_multi["tipo_doc_norm"].iloc[0],
                            "numero_doc_norm": ";".join(
                                df_multi["numero_doc_norm"].astype(str).tolist()
                            ),
                            "plantilla_cuentas": tuple(),
                            "plantilla_cuentas_dc": tuple(),
                            "n_lineas": df_multi["n_lineas"].sum(),
                            "concepto_concat": " | ".join(
                                lista_unicos_limpios(df_multi["concepto_concat"])
                            ),
                        }
                    )
                    continue

        df_validos = df_validos.sort_values(
            by=[
                "score_total",
                "flag_texto_match_exacto",
                "flag_texto_match_ult6",
                "flag_texto_match_ult5",
                "flag_texto_match_ult4",
                "dif_valor_pct",
            ],
            ascending=[False, False, False, False, False, True],
            na_position="last",
        )

        mejor = df_validos.iloc[0]
        top2 = df_validos.head(2)
        estado = "REVISION_MANUAL"

        if len(df_validos) == 1 and mejor.get("flag_texto_match", False):
            estado = "OK_UNICO"
        elif mejor.get("flag_texto_match_exacto", False) and (
            len(top2) == 1
            or pd.isna(top2.iloc[1]["score_total"])
            or (mejor["score_total"] - top2.iloc[1]["score_total"] >= 20)
        ):
            estado = "OK_TEXTO"
        elif mejor.get("flag_texto_match_ult6", False) and (
            not pd.isna(mejor.get("dif_valor_pct"))
            and mejor["dif_valor_pct"] <= 0.08
        ):
            estado = "OK_TEXTO"
        elif mejor.get("flag_texto_match_ult5", False) and (
            not pd.isna(mejor.get("dif_valor_pct"))
            and mejor["dif_valor_pct"] <= 0.06
        ):
            estado = "OK_TEXTO"
        elif mejor.get("flag_texto_match_ult4", False) and (
            not pd.isna(mejor.get("dif_valor_pct"))
            and mejor["dif_valor_pct"] <= 0.05
        ):
            estado = "OK_TEXTO"
        elif (
            mejor.get("flag_texto_match", False)
            and not pd.isna(mejor.get("dif_valor_pct"))
            and mejor["dif_valor_pct"] <= 0.10
        ):
            estado = "OK_VALOR"

        resultados.append(
            {
                "llave_factura": llave_factura,
                "llave_asiento": mejor.get("llave_asiento"),
                "score_total": mejor.get("score_total"),
                "estado_match": estado,
                "motivo_score": mejor.get("motivo_score"),
                "id_factura": mejor.get("id_factura"),
                "factura_match_norm": mejor.get("factura_match_norm"),
                "nit_proveedor_norm": mejor.get("nit_proveedor_norm"),
                "fecha_emision": mejor.get("fecha_emision"),
                "total_factura": mejor.get("total_factura"),
                "fecha_asiento": mejor.get("fecha_asiento"),
                "monto_asiento": mejor.get("monto_asiento"),
                "tipo_doc_norm": mejor.get("tipo_doc_norm"),
                "numero_doc_norm": mejor.get("numero_doc_norm"),
                "plantilla_cuentas": mejor.get("plantilla_cuentas"),
                "plantilla_cuentas_dc": mejor.get("plantilla_cuentas_dc"),
                "n_lineas": mejor.get("n_lineas"),
                "concepto_concat": mejor.get("concepto_concat"),
            }
        )

    return pd.DataFrame(resultados)


def construir_dataset_modelo(
    facturas_df: pd.DataFrame,
    match_df: pd.DataFrame,
    asientos_fc_df: pd.DataFrame,
) -> pd.DataFrame:
    dataset = facturas_df.merge(
        match_df,
        on=["llave_factura"],
        how="left",
        suffixes=("", "_match"),
    )
    dataset["target_plantilla_cuentas"] = dataset["plantilla_cuentas_dc"].where(
        ~dataset["plantilla_cuentas_dc"].isna(),
        dataset["plantilla_cuentas"],
    )
    dataset["flag_match_ok"] = dataset["estado_match"].isin(
        ["OK_UNICO", "OK_TEXTO", "OK_VALOR"]
    )
    return dataset


def construir_dataset_autogluon(
    dataset_modelo_df: pd.DataFrame,
) -> pd.DataFrame:
    estados_ok = ["OK_UNICO", "OK_TEXTO", "OK_VALOR"]
    df = dataset_modelo_df.loc[
        dataset_modelo_df["estado_match"].isin(estados_ok)
        & dataset_modelo_df["target_plantilla_cuentas"].notna()
    ].copy()
    df["target_plantilla_cuentas"] = df["target_plantilla_cuentas"].astype(str)
    return df[
        [c for c in COLUMNAS_MODELO_AUTOG if c in df.columns]
    ].copy()


def construir_lineas_historicas_valores(
    dataset_modelo_df: pd.DataFrame,
    movimientos_fc_df: pd.DataFrame,
) -> pd.DataFrame:
    estados_ok = ["OK_UNICO", "OK_TEXTO", "OK_VALOR"]
    base = dataset_modelo_df.loc[
        dataset_modelo_df["estado_match"].isin(estados_ok)
        & dataset_modelo_df["llave_asiento"].notna(),
        [
            "empresa", "llave_factura", "llave_asiento",
            "nit_proveedor_norm", "total_factura",
            "target_plantilla_cuentas",
        ],
    ].copy()

    lineas = movimientos_fc_df.copy().merge(
        base,
        left_on="llave_asiento_base",
        right_on="llave_asiento",
        how="inner",
        suffixes=("_mov", "_fact"),
    )

    lineas["valor_linea"] = np.where(
        lineas["debito"] > 0,
        lineas["debito"],
        lineas["credito"],
    )
    lineas["naturaleza"] = np.select(
        [
            (lineas["debito"] > 0) & (lineas["credito"] > 0),
            lineas["debito"] > 0,
            lineas["credito"] > 0,
        ],
        ["DC", "D", "C"],
        default="N",
    )
    lineas["cuenta_naturaleza"] = (
        lineas["cuenta_limpia"].astype(str) + "_" + lineas["naturaleza"]
    )
    lineas["ratio_total"] = np.where(
        lineas["total_factura"] != 0,
        lineas["valor_linea"] / lineas["total_factura"].abs(),
        np.nan,
    )

    return lineas.copy()


def construir_perfil_ratios_valores(
    lineas_historicas_df: pd.DataFrame,
) -> pd.DataFrame:
    if lineas_historicas_df.empty:
        return pd.DataFrame()

    empresa_col = (
        "empresa_mov" if "empresa_mov" in lineas_historicas_df.columns
        else "empresa"
    )

    perfil = (
        lineas_historicas_df.groupby(
            [
                empresa_col, "nit_proveedor_norm",
                "target_plantilla_cuentas", "cuenta_naturaleza",
            ],
            dropna=False,
        )
        .agg(
            ratio_mediano=("ratio_total", "median"),
            ratio_promedio=("ratio_total", "mean"),
            ratio_std=("ratio_total", "std"),
            n_observaciones=("ratio_total", "count"),
        )
        .reset_index()
        .rename(columns={empresa_col: "empresa"})
    )

    return perfil


def construir_resumen_calidad(
    facturas_df: pd.DataFrame,
    movimientos_df: pd.DataFrame,
    movimientos_fc_df: pd.DataFrame,
    asientos_fc_df: pd.DataFrame,
    candidatos_df: pd.DataFrame,
    match_df: pd.DataFrame,
    dataset_autogluon_df: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"indicador": "facturas_total", "valor": len(facturas_df)},
            {
                "indicador": "facturas_sin_nit",
                "valor": int(facturas_df["flag_sin_nit"].sum()),
            },
            {
                "indicador": "facturas_sin_factura",
                "valor": int(facturas_df["flag_sin_factura"].sum()),
            },
            {
                "indicador": "facturas_con_descuento",
                "valor": int(facturas_df["flag_descuento"].sum()),
            },
            {
                "indicador": "facturas_con_recargo",
                "valor": int(facturas_df["flag_recargo"].sum()),
            },
            {
                "indicador": "facturas_umbral_rf_servicios",
                "valor": int(facturas_df["flag_umbral_rf_servicios"].sum()),
            },
            {
                "indicador": "facturas_umbral_reteica_servicios",
                "valor": int(
                    facturas_df["flag_umbral_reteica_bogota_servicios"].sum()
                ),
            },
            {
                "indicador": "facturas_umbral_reteica_compras",
                "valor": int(
                    facturas_df["flag_umbral_reteica_bogota_compras"].sum()
                ),
            },
            {
                "indicador": "facturas_dif_payable_tax_inclusive",
                "valor": int(
                    facturas_df["flag_diferencia_payable_tax_inclusive"].sum()
                ),
            },
            {"indicador": "movimientos_total", "valor": len(movimientos_df)},
            {
                "indicador": "movimientos_fc_total",
                "valor": len(movimientos_fc_df),
            },
            {
                "indicador": "asientos_fc_agrupados",
                "valor": len(asientos_fc_df),
            },
            {"indicador": "candidatos_total", "valor": len(candidatos_df)},
            {"indicador": "matches_total", "valor": len(match_df)},
            {
                "indicador": "matches_ok",
                "valor": int(
                    match_df["estado_match"].isin(
                        ["OK_UNICO", "OK_TEXTO", "OK_VALOR"]
                    ).sum()
                ),
            },
            {
                "indicador": "matches_revision",
                "valor": int(
                    match_df["estado_match"]
                    .astype(str)
                    .str.contains("REVISION", na=False)
                    .sum()
                ),
            },
            {
                "indicador": "matches_sin_match",
                "valor": int((match_df["estado_match"] == "SIN_MATCH").sum()),
            },
            {
                "indicador": "filas_autogluon",
                "valor": len(dataset_autogluon_df),
            },
        ]
    )


def ejecutar_pipeline(
    ruta_facturas: str,
    ruta_movimientos: str,
    hoja_facturas=0,
    hoja_movimientos=0,
    empresa: str = "demo",
    columnas_movimientos: dict | None = None,
    ruta_xml_descripciones: str | Path | None = None,
) -> dict:
    facturas_df = procesar_facturas(
        ruta_facturas,
        hoja_facturas,
        empresa,
        ruta_xml_descripciones,
    )
    movimientos_df = procesar_movimientos(
        ruta_movimientos,
        hoja_movimientos,
        empresa,
        columnas_movimientos,
    )
    movimientos_fc_df = filtrar_movimientos_fc(movimientos_df)
    asientos_fc_df = agrupar_asientos_fc(movimientos_fc_df)
    candidatos_df = generar_candidatos_match(facturas_df, asientos_fc_df)
    match_df = resolver_match(candidatos_df)
    dataset_modelo_df = construir_dataset_modelo(
        facturas_df,
        match_df,
        asientos_fc_df,
    )
    dataset_autogluon_df = construir_dataset_autogluon(dataset_modelo_df)
    lineas_historicas_valores_df = construir_lineas_historicas_valores(
        dataset_modelo_df,
        movimientos_fc_df,
    )
    perfil_ratios_valores_df = construir_perfil_ratios_valores(
        lineas_historicas_valores_df
    )
    resumen_calidad_df = construir_resumen_calidad(
        facturas_df,
        movimientos_df,
        movimientos_fc_df,
        asientos_fc_df,
        candidatos_df,
        match_df,
        dataset_autogluon_df,
    )

    return {
        "facturas_df": facturas_df,
        "movimientos_df": movimientos_df,
        "movimientos_fc_df": movimientos_fc_df,
        "asientos_fc_df": asientos_fc_df,
        "candidatos_df": candidatos_df,
        "match_df": match_df,
        "dataset_modelo_df": dataset_modelo_df,
        "dataset_autogluon_df": dataset_autogluon_df,
        "lineas_historicas_valores_df": lineas_historicas_valores_df,
        "perfil_ratios_valores_df": perfil_ratios_valores_df,
        "resumen_calidad_df": resumen_calidad_df,
    }
