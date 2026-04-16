import pandas as pd
import numpy as np

from funciones_de_limpieza_base_datos_factura import (
    es_nulo,
    limpiar_espacios,
    normalizar_texto_basico,
    normalizar_texto_modelo,
    normalizar_alfanumerico,
    normalizar_nit,
    safe_to_datetime,
    detectar_prefijo_factura,
    coalesce_alfa_num,
    construir_descripcion_modelo,
    normalizar_nombre_columna,
    normalizar_valor_monetario,
    normalizar_lista_cuentas,
    normalizar_lista_cuentas_dc,
    texto_contiene_factura,
)


def asegurar_columnas(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    for col in columnas:
        if col not in df.columns:
            df[col] = None
    return df


def primer_valor_no_nulo(serie: pd.Series):
    for x in serie:
        if not es_nulo(x):
            return x
    return None


def moda_o_primero(serie: pd.Series):
    serie = serie.dropna()
    if len(serie) == 0:
        return None
    moda = serie.mode(dropna=True)
    return moda.iloc[0] if len(moda) > 0 else serie.iloc[0]


def lista_unicos_limpios(serie: pd.Series) -> list[str]:
    vals = []
    vistos = set()
    for x in serie:
        if es_nulo(x):
            continue
        x = limpiar_espacios(x)
        if x and x not in vistos:
            vals.append(x)
            vistos.add(x)
    return vals


def coalesce_fecha(*args):
    for x in args:
        y = pd.to_datetime(x, errors="coerce")
        if not pd.isna(y):
            return y
    return pd.NaT


def anio_a_texto(anio) -> str:
    if pd.isna(anio):
        return "-1"
    return str(int(anio))


def valor_a_texto_llave(x) -> str:
    if es_nulo(x):
        return ""
    return str(x)


def construir_llave_factura(nit_proveedor_norm, factura_match_norm) -> str:
    """
    Llave principal de factura para el MVP.

    La llave queda basada en nit_proveedor_norm, que es el identificador estable
    del proveedor. Se conserva factura_match_norm como segundo componente para
    diferenciar varias facturas del mismo proveedor.

    No incluye empresa porque el modelo se entrena por una sola empresa receptora.
    No usa nombre del proveedor porque es inestable.

    Estructura:
        nit_proveedor_norm|factura_match_norm
    """
    return (
        valor_a_texto_llave(nit_proveedor_norm)
        + "|"
        + valor_a_texto_llave(factura_match_norm)
    )


def construir_llave_asiento(anio, tipo_doc_norm, numero_doc_norm) -> str:
    """
    Llave técnica del asiento contable.

    No incluye empresa porque, en el MVP, el universo de movimientos pertenece
    a una sola empresa receptora. Se mantiene el año para reducir riesgo de
    colisión si el consecutivo local se reinicia por periodo.

    Nota:
    Esta llave identifica el comprobante/asiento contable. El match contra
    facturas se hace principalmente por nit_tercero_norm y evidencia del número
    de factura en el concepto.
    """
    return (
        anio_a_texto(anio)
        + "|"
        + valor_a_texto_llave(tipo_doc_norm)
        + "|"
        + valor_a_texto_llave(numero_doc_norm)
    )


def score_match(fila: pd.Series) -> tuple[float, str]:
    score = 0.0
    motivos = []

    if fila.get("flag_texto_match", False):
        score += 100
        motivos.append("texto_factura")

    # El merge de candidatos ya se hace por NIT/documento.
    if not es_nulo(fila.get("nit_proveedor_norm")) and not es_nulo(fila.get("nit_dominante_norm")):
        if fila.get("nit_proveedor_norm") == fila.get("nit_dominante_norm"):
            score += 35
            motivos.append("nit")

    fecha_factura = fila.get("fecha_emision")
    fecha_asiento = fila.get("fecha_asiento")

    if not pd.isna(fecha_factura) and not pd.isna(fecha_asiento):
        dif_dias = abs((fecha_asiento - fecha_factura).days)
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

    total_factura = fila.get("total_factura")
    monto_asiento = fila.get("monto_asiento")

    if not pd.isna(total_factura) and not pd.isna(monto_asiento) and total_factura != 0:
        dif_valor = abs(monto_asiento - total_factura)
        pct = dif_valor / abs(total_factura)
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


def cargar_excel(ruta_excel: str, hoja=0) -> pd.DataFrame:
    df = pd.read_excel(ruta_excel, sheet_name=hoja).copy()
    df.columns = [normalizar_nombre_columna(c) for c in df.columns]
    return df


def procesar_facturas(ruta_excel: str, hoja=0, empresa: str = "demo") -> pd.DataFrame:
    df = cargar_excel(ruta_excel, hoja=hoja)

    columnas_esperadas = [
        "id_carga", "archivo_xml", "id_factura", "cufe", "factura_completa",
        "fecha_emision", "nit_proveedor", "nombre_proveedor", "ciudad_proveedor",
        "tax_level_proveedor", "tax_scheme_id", "tax_scheme_nombre",
        "codigo_industria_proveedor", "cantidad_lineas_xml", "line_extension_amount",
        "tax_exclusive_amount", "tax_inclusive_amount", "payable_amount",
        "iva_total", "inc_total", "tiene_iva", "tiene_inc", "cantidad_items_total",
        "descripcion_item_1", "item1_proveedor", "n_registros_sugeridos",
        "valor_base_sugerido", "valor_iva_sugerido", "valor_inc_sugerido",
        "valor_cxp_sugerido", "observaciones",
    ]
    df = asegurar_columnas(df, columnas_esperadas)

    # Se conserva empresa solo como metadato del experimento/MVP.
    # No se usa en la llave porque el modelo se entrena por empresa receptora.
    df["empresa"] = empresa

    cols_texto_basico = [
        "id_carga", "archivo_xml", "id_factura", "cufe", "factura_completa",
        "nombre_proveedor", "ciudad_proveedor", "tax_level_proveedor",
        "tax_scheme_id", "tax_scheme_nombre", "codigo_industria_proveedor",
        "descripcion_item_1", "item1_proveedor", "observaciones",
    ]
    for col in cols_texto_basico:
        df[f"{col}_limpio"] = df[col].apply(normalizar_texto_basico)

    cols_texto_modelo = ["nombre_proveedor", "ciudad_proveedor", "descripcion_item_1", "item1_proveedor"]
    for col in cols_texto_modelo:
        df[f"{col}_modelo"] = df[col].apply(normalizar_texto_modelo)

    cols_alfa_num = ["id_factura", "factura_completa", "cufe"]
    for col in cols_alfa_num:
        df[f"{col}_norm"] = df[col].apply(normalizar_alfanumerico)

    df["nit_proveedor_limpio"] = df["nit_proveedor"].apply(normalizar_texto_basico)
    df["nit_proveedor_norm"] = df["nit_proveedor"].apply(normalizar_nit)

    df["factura_match_norm"] = df.apply(
        lambda row: coalesce_alfa_num(row.get("id_factura"), row.get("factura_completa")),
        axis=1,
    )
    df["prefijo_factura"] = df["factura_match_norm"].apply(detectar_prefijo_factura)

    df["proveedor_nombre_limpio"] = df["nombre_proveedor_limpio"]
    df["ciudad_proveedor_limpia"] = df["ciudad_proveedor_limpio"]
    df["item_descripcion_1_limpia"] = df["descripcion_item_1_limpio"]

    df["descripcion_modelo"] = df.apply(construir_descripcion_modelo, axis=1)
    df["descripcion_modelo_norm"] = df["descripcion_modelo"].apply(normalizar_texto_modelo)

    df["fecha_emision"] = safe_to_datetime(df["fecha_emision"])
    df["anio"] = df["fecha_emision"].dt.year
    df["mes"] = df["fecha_emision"].dt.month
    df["trimestre"] = df["fecha_emision"].dt.quarter
    df["anio_mes"] = df["fecha_emision"].dt.to_period("M").astype("string")

    cols_numericas = [
        "cantidad_lineas_xml", "line_extension_amount", "tax_exclusive_amount",
        "tax_inclusive_amount", "payable_amount", "iva_total", "inc_total",
        "cantidad_items_total", "n_registros_sugeridos", "valor_base_sugerido",
        "valor_iva_sugerido", "valor_inc_sugerido", "valor_cxp_sugerido",
    ]
    for col in cols_numericas:
        df[col] = df[col].apply(normalizar_valor_monetario)

    df["base_amount"] = df["line_extension_amount"]
    df["total_factura"] = df["payable_amount"]
    df["cantidad_items"] = df["cantidad_items_total"]

    df["base_amount"] = df["base_amount"].fillna(0)
    df["iva_total"] = df["iva_total"].fillna(0)
    df["inc_total"] = df["inc_total"].fillna(0)

    df["tiene_iva"] = df["iva_total"] > 0
    df["tiene_inc"] = df["inc_total"] > 0

    df["total_impuestos"] = df["iva_total"] + df["inc_total"]
    df["base_mas_impuestos"] = df["base_amount"] + df["total_impuestos"]
    df["diferencia_total_vs_componentes"] = df["total_factura"] - df["base_mas_impuestos"]

    df["flag_sin_nit"] = df["nit_proveedor_norm"].isna()
    df["flag_sin_factura"] = df["factura_match_norm"].isna()
    df["flag_sin_fecha"] = df["fecha_emision"].isna()
    df["flag_sin_total"] = df["total_factura"].isna()

    # Corrección principal:
    # La llave usa nit_proveedor_norm como base estable del proveedor.
    # factura_match_norm se mantiene para distinguir facturas del mismo proveedor.
    # No usa empresa ni nombre del proveedor.
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

    # Se conserva empresa como metadato, pero no participa en la llave del asiento.
    df["empresa"] = empresa

    cols_texto_basico = [
        "tipo_doc", "numero_doc", "cuenta", "nombre_cuenta", "identidad",
        "nombre_tercero", "concepto", "codigo_centro_costo", "centro_costo",
        "usuario", "numero_movil", "nombre_centro_costo",
    ]
    for col in cols_texto_basico:
        df[f"{col}_limpio"] = df[col].apply(normalizar_texto_basico)

    cols_texto_modelo = ["nombre_cuenta", "nombre_tercero", "concepto", "nombre_centro_costo"]
    for col in cols_texto_modelo:
        df[f"{col}_modelo"] = df[col].apply(normalizar_texto_modelo)

    cols_alfa_num = ["tipo_doc", "numero_doc", "cuenta", "concepto"]
    for col in cols_alfa_num:
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

    # Corrección principal:
    # La llave del asiento deja de usar empresa.
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

    # Corrección principal:
    # No se agrupa por empresa. El documento contable se identifica por año,
    # tipo y consecutivo local del software.
    group_cols = ["anio", "tipo_doc_norm", "numero_doc_norm"]

    for keys, df_asiento in movimientos_fc_df.groupby(group_cols, dropna=False):
        anio, tipo_doc_norm, numero_doc_norm = keys

        fecha_asiento = coalesce_fecha(
            primer_valor_no_nulo(df_asiento["fecha_mov"]),
            df_asiento["fecha_mov"].min(),
        )
        nit_dominante = moda_o_primero(df_asiento["nit_tercero_norm"])
        concepto_concat = " | ".join(lista_unicos_limpios(df_asiento["concepto_limpio"]))

        total_debito = df_asiento["debito"].sum()
        total_credito = df_asiento["credito"].sum()

        resumen.append({
            # Se conserva como metadato para trazabilidad, no como llave.
            "empresa": moda_o_primero(df_asiento["empresa"]) if "empresa" in df_asiento.columns else None,
            "anio": anio,
            "tipo_doc_norm": tipo_doc_norm,
            "numero_doc_norm": numero_doc_norm,
            "llave_asiento": construir_llave_asiento(anio, tipo_doc_norm, numero_doc_norm),
            "fecha_asiento": fecha_asiento,
            "nit_dominante_norm": nit_dominante,
            "concepto_concat": concepto_concat,
            "concepto_concat_norm": normalizar_alfanumerico(concepto_concat),
            "n_lineas": len(df_asiento),
            "total_debito": total_debito,
            "total_credito": total_credito,
            "monto_asiento": max(total_debito, total_credito),
            "plantilla_cuentas": normalizar_lista_cuentas(df_asiento),
            "plantilla_cuentas_dc": normalizar_lista_cuentas_dc(df_asiento),
        })

    return pd.DataFrame(resumen)


def generar_candidatos_match(
    facturas_df: pd.DataFrame,
    asientos_fc_df: pd.DataFrame,
) -> pd.DataFrame:
    fact_cols = [
        "empresa", "llave_factura", "id_factura", "factura_completa",
        "factura_match_norm", "nit_proveedor_norm", "fecha_emision",
        "total_factura", "base_amount", "iva_total", "inc_total",
    ]

    as_cols = [
        # Se excluye empresa del lado de asientos para evitar merge por empresa
        # o columnas empresa_x/empresa_y que contaminen el MVP.
        "llave_asiento", "anio", "tipo_doc_norm", "numero_doc_norm",
        "fecha_asiento", "nit_dominante_norm", "concepto_concat",
        "concepto_concat_norm", "n_lineas", "total_debito", "total_credito",
        "monto_asiento", "plantilla_cuentas", "plantilla_cuentas_dc",
    ]

    fact = facturas_df[[c for c in fact_cols if c in facturas_df.columns]].copy()
    asi = asientos_fc_df[[c for c in as_cols if c in asientos_fc_df.columns]].copy()

    # Corrección principal:
    # El candidato nace por identidad tributaria estable:
    # nit_proveedor_norm en facturas contra nit_dominante_norm en contabilidad.
    # Luego el score valida si el número de factura aparece en el concepto.
    cand = fact.merge(
        asi,
        left_on=["nit_proveedor_norm"],
        right_on=["nit_dominante_norm"],
        how="left",
    )

    cand["flag_texto_match"] = cand.apply(
        lambda row: texto_contiene_factura(
            row.get("concepto_concat_norm"),
            row.get("factura_match_norm"),
        ),
        axis=1,
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
    tolerancia_multiasiento = 0.02

    for llave_factura, df_cand in candidatos_df.groupby("llave_factura", dropna=False):
        df_cand = df_cand.copy()

        if df_cand["llave_asiento"].isna().all():
            fila_base = df_cand.iloc[0].copy()
            resultados.append({
                "llave_factura": llave_factura,
                "llave_asiento": None,
                "score_total": np.nan,
                "estado_match": "SIN_MATCH",
                "motivo_score": "sin_candidatos",
                "id_factura": fila_base.get("id_factura"),
                "factura_match_norm": fila_base.get("factura_match_norm"),
                "nit_proveedor_norm": fila_base.get("nit_proveedor_norm"),
                "fecha_emision": fila_base.get("fecha_emision"),
                "total_factura": fila_base.get("total_factura"),
            })
            continue

        df_validos = df_cand.loc[~df_cand["llave_asiento"].isna()].copy()

        df_multiasiento = df_validos.loc[
            df_validos["flag_texto_match"].fillna(False)
        ].copy()

        if len(df_multiasiento) > 1:
            total_factura = df_multiasiento["total_factura"].iloc[0]
            suma_asientos = df_multiasiento["monto_asiento"].sum()

            if not pd.isna(total_factura) and total_factura != 0:
                dif_pct_multi = abs(suma_asientos - total_factura) / abs(total_factura)

                if dif_pct_multi <= tolerancia_multiasiento:
                    resultados.append({
                        "llave_factura": llave_factura,
                        "llave_asiento": ";".join(df_multiasiento["llave_asiento"].astype(str).tolist()),
                        "score_total": df_multiasiento["score_total"].max(),
                        "estado_match": "REVISION_MULTIASIENTO",
                        "motivo_score": "texto_multiple|valor_sumado",
                        "id_factura": df_multiasiento["id_factura"].iloc[0],
                        "factura_match_norm": df_multiasiento["factura_match_norm"].iloc[0],
                        "nit_proveedor_norm": df_multiasiento["nit_proveedor_norm"].iloc[0],
                        "fecha_emision": df_multiasiento["fecha_emision"].iloc[0],
                        "total_factura": total_factura,
                        "fecha_asiento": df_multiasiento["fecha_asiento"].min(),
                        "monto_asiento": suma_asientos,
                        "tipo_doc_norm": df_multiasiento["tipo_doc_norm"].iloc[0],
                        "numero_doc_norm": ";".join(df_multiasiento["numero_doc_norm"].astype(str).tolist()),
                        "plantilla_cuentas": tuple(),
                        "plantilla_cuentas_dc": tuple(),
                        "n_lineas": df_multiasiento["n_lineas"].sum(),
                        "concepto_concat": " | ".join(lista_unicos_limpios(df_multiasiento["concepto_concat"])),
                    })
                    continue

        df_validos = df_validos.sort_values(
            by=["score_total", "flag_texto_match", "dif_valor_pct"],
            ascending=[False, False, True],
            na_position="last",
        )

        mejor = df_validos.iloc[0]
        top2 = df_validos.head(2)

        estado = "REVISION_MANUAL"

        if len(df_validos) == 1 and mejor.get("flag_texto_match", False):
            estado = "OK_UNICO"
        elif mejor.get("flag_texto_match", False) and (
            len(top2) == 1
            or pd.isna(top2.iloc[1]["score_total"])
            or (mejor["score_total"] - top2.iloc[1]["score_total"] >= 20)
        ):
            estado = "OK_TEXTO"
        elif mejor.get("flag_texto_match", False) and not pd.isna(mejor.get("dif_valor_pct")) and mejor["dif_valor_pct"] <= 0.10:
            estado = "OK_VALOR"

        resultados.append({
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
        })

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


def construir_dataset_autogluon(dataset_modelo_df: pd.DataFrame) -> pd.DataFrame:
    estados_ok = ["OK_UNICO", "OK_TEXTO", "OK_VALOR"]

    df = dataset_modelo_df.loc[
        dataset_modelo_df["estado_match"].isin(estados_ok)
        & dataset_modelo_df["target_plantilla_cuentas"].notna()
    ].copy()

    df["target_plantilla_cuentas"] = df["target_plantilla_cuentas"].astype(str)

    columnas_modelo = [
        # Empresa se conserva como metadato opcional.
        # Para un MVP por una sola empresa será una columna constante.
        "empresa",
        "nit_proveedor_norm",
        "nombre_proveedor_modelo",
        "ciudad_proveedor_modelo",
        "tax_level_proveedor_limpio",
        "tax_scheme_id_limpio",
        "tax_scheme_nombre_limpio",
        "codigo_industria_proveedor_limpio",
        "descripcion_item_1_modelo",
        "item1_proveedor_modelo",
        "descripcion_modelo_norm",
        "prefijo_factura",
        "cantidad_lineas_xml",
        "line_extension_amount",
        "tax_exclusive_amount",
        "tax_inclusive_amount",
        "payable_amount",
        "iva_total",
        "inc_total",
        "cantidad_items_total",
        "n_registros_sugeridos",
        "valor_base_sugerido",
        "valor_iva_sugerido",
        "valor_inc_sugerido",
        "valor_cxp_sugerido",
        "tiene_iva",
        "tiene_inc",
        "anio",
        "mes",
        "trimestre",
        "target_plantilla_cuentas",
    ]

    return df[[c for c in columnas_modelo if c in df.columns]].copy()


def construir_lineas_historicas_valores(
    dataset_modelo_df: pd.DataFrame,
    movimientos_fc_df: pd.DataFrame,
) -> pd.DataFrame:
    estados_ok = ["OK_UNICO", "OK_TEXTO", "OK_VALOR"]

    base = dataset_modelo_df.loc[
        dataset_modelo_df["estado_match"].isin(estados_ok)
        & dataset_modelo_df["llave_asiento"].notna()
    ][[
        "empresa",
        "llave_factura",
        "llave_asiento",
        "nit_proveedor_norm",
        "total_factura",
        "target_plantilla_cuentas",
    ]].copy()

    mov = movimientos_fc_df.copy()

    lineas = mov.merge(
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
        lineas["cuenta_limpia"].astype(str)
        + "_"
        + lineas["naturaleza"]
    )

    lineas["ratio_total"] = np.where(
        lineas["total_factura"] != 0,
        lineas["valor_linea"] / lineas["total_factura"].abs(),
        np.nan,
    )

    return lineas.copy()


def construir_perfil_ratios_valores(lineas_historicas_df: pd.DataFrame) -> pd.DataFrame:
    if lineas_historicas_df.empty:
        return pd.DataFrame()

    empresa_col = "empresa_mov" if "empresa_mov" in lineas_historicas_df.columns else "empresa"

    perfil = (
        lineas_historicas_df
        .groupby(
            [
                empresa_col,
                "nit_proveedor_norm",
                "target_plantilla_cuentas",
                "cuenta_naturaleza",
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
    """
    Resumen de control para revisar si el procesamiento está funcionando.
    """
    resumen = [
        {"indicador": "facturas_total", "valor": len(facturas_df)},
        {"indicador": "facturas_sin_nit", "valor": int(facturas_df["flag_sin_nit"].sum()) if "flag_sin_nit" in facturas_df.columns else np.nan},
        {"indicador": "facturas_sin_factura", "valor": int(facturas_df["flag_sin_factura"].sum()) if "flag_sin_factura" in facturas_df.columns else np.nan},
        {"indicador": "movimientos_total", "valor": len(movimientos_df)},
        {"indicador": "movimientos_fc_total", "valor": len(movimientos_fc_df)},
        {"indicador": "asientos_fc_agrupados", "valor": len(asientos_fc_df)},
        {"indicador": "candidatos_total", "valor": len(candidatos_df)},
        {"indicador": "matches_total", "valor": len(match_df)},
        {"indicador": "matches_ok", "valor": int(match_df["estado_match"].isin(["OK_UNICO", "OK_TEXTO", "OK_VALOR"]).sum()) if "estado_match" in match_df.columns else 0},
        {"indicador": "matches_revision", "valor": int(match_df["estado_match"].astype(str).str.contains("REVISION", na=False).sum()) if "estado_match" in match_df.columns else 0},
        {"indicador": "matches_sin_match", "valor": int((match_df["estado_match"] == "SIN_MATCH").sum()) if "estado_match" in match_df.columns else 0},
        {"indicador": "filas_autogluon", "valor": len(dataset_autogluon_df)},
    ]
    return pd.DataFrame(resumen)


def ejecutar_pipeline(
    ruta_facturas: str,
    ruta_movimientos: str,
    hoja_facturas=0,
    hoja_movimientos=0,
    empresa: str = "demo",
    columnas_movimientos: dict | None = None,
) -> dict:
    facturas_df = procesar_facturas(
        ruta_excel=ruta_facturas,
        hoja=hoja_facturas,
        empresa=empresa,
    )

    movimientos_df = procesar_movimientos(
        ruta_excel=ruta_movimientos,
        hoja=hoja_movimientos,
        empresa=empresa,
        columnas_map=columnas_movimientos,
    )

    movimientos_fc_df = filtrar_movimientos_fc(movimientos_df)
    asientos_fc_df = agrupar_asientos_fc(movimientos_fc_df)
    candidatos_df = generar_candidatos_match(facturas_df, asientos_fc_df)
    match_df = resolver_match(candidatos_df)
    dataset_modelo_df = construir_dataset_modelo(facturas_df, match_df, asientos_fc_df)
    dataset_autogluon_df = construir_dataset_autogluon(dataset_modelo_df)

    lineas_historicas_valores_df = construir_lineas_historicas_valores(
        dataset_modelo_df,
        movimientos_fc_df,
    )

    perfil_ratios_valores_df = construir_perfil_ratios_valores(
        lineas_historicas_valores_df
    )

    resumen_calidad_df = construir_resumen_calidad(
        facturas_df=facturas_df,
        movimientos_df=movimientos_df,
        movimientos_fc_df=movimientos_fc_df,
        asientos_fc_df=asientos_fc_df,
        candidatos_df=candidatos_df,
        match_df=match_df,
        dataset_autogluon_df=dataset_autogluon_df,
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
