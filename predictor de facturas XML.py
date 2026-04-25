# predictor_facturas_contabilidad.py

from pathlib import Path
import json
import ast

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

import procesamiento_facturas_contabilidad_match as pfcm


cargar_excel = getattr(pfcm, "cargar_excel", None)


def cargar_excel_flexible(ruta_excel: str, hoja=0) -> pd.DataFrame:
    if callable(cargar_excel):
        return cargar_excel(ruta_excel, hoja=hoja)

    df = pd.read_excel(ruta_excel, sheet_name=hoja).copy()

    normalizar_nombre_columna = getattr(pfcm, "normalizar_nombre_columna", None)
    if callable(normalizar_nombre_columna):
        df.columns = [normalizar_nombre_columna(c) for c in df.columns]
    else:
        from funciones_de_limpieza_base_datos_factura import normalizar_nombre_columna as _nnc
        df.columns = [_nnc(c) for c in df.columns]

    return df
procesar_facturas = getattr(pfcm, "procesar_facturas", None)
if procesar_facturas is None:
    procesar_facturas = getattr(pfcm, "preparar_facturas", None)

if procesar_facturas is None:
    raise ImportError(
        "No se encontró ni 'procesar_facturas' ni 'preparar_facturas' en "
        "procesamiento_facturas_contabilidad_match.py"
    )


ejecutar_pipeline = getattr(pfcm, "ejecutar_pipeline", None)
if ejecutar_pipeline is None:
    procesar_movimientos = getattr(pfcm, "procesar_movimientos", None)
    if procesar_movimientos is None:
        procesar_movimientos = getattr(pfcm, "preparar_movimientos", None)

    filtrar_movimientos_fc = getattr(pfcm, "filtrar_movimientos_fc", None)
    agrupar_asientos_fc = getattr(pfcm, "agrupar_asientos_fc", None)
    generar_candidatos_match = getattr(pfcm, "generar_candidatos_match", None)
    resolver_match = getattr(pfcm, "resolver_match", None)
    construir_dataset_modelo = getattr(pfcm, "construir_dataset_modelo", None)
    construir_dataset_autogluon = getattr(pfcm, "construir_dataset_autogluon", None)
    construir_lineas_historicas_valores = getattr(pfcm, "construir_lineas_historicas_valores", None)
    construir_perfil_ratios_valores = getattr(pfcm, "construir_perfil_ratios_valores", None)
    construir_resumen_calidad = getattr(pfcm, "construir_resumen_calidad", None)

    componentes_requeridos = [
        procesar_movimientos,
        filtrar_movimientos_fc,
        agrupar_asientos_fc,
        generar_candidatos_match,
        resolver_match,
        construir_dataset_modelo,
        construir_dataset_autogluon,
        construir_lineas_historicas_valores,
        construir_perfil_ratios_valores,
        construir_resumen_calidad,
    ]

    if any(x is None for x in componentes_requeridos):
        raise ImportError(
            "No se encontró 'ejecutar_pipeline' y tampoco están todos los componentes "
            "necesarios para reconstruirlo en procesamiento_facturas_contabilidad_match.py"
        )

    def ejecutar_pipeline(
        ruta_facturas: str,
        ruta_movimientos: str,
        hoja_facturas=0,
        hoja_movimientos=0,
        empresa: str = "demo",
        columnas_movimientos: dict | None = None,
    ) -> dict:
        facturas_df = procesar_facturas(ruta_facturas, hoja_facturas, empresa)
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
from funciones_de_limpieza_base_datos_factura import limpiar_espacios


# ========================================================Pu
# PREDICTOR
# =========================================================

base = Path(r"C:/Users/eduar/OneDrive - TCP BAAS S.A.S/Automatización/facturas_bigquery")
carpeta_salida = base / "resultados"
carpeta_metricas = carpeta_salida / "metricas_modelo"
carpeta_salida.mkdir(parents=True, exist_ok=True)

ruta_facturas_nuevas = base / "facturas_a_predecir.xlsx"
ruta_modelo_actual = carpeta_metricas / "ruta_modelo_actual.txt"
ruta_columnas_modelo = carpeta_metricas / "columnas_modelo.json"
ruta_contabilidad = Path(r"C:/Users/eduar/OneDrive - TCP BAAS S.A.S/Automatización/facturas_bigquery/contabilidad.xlsx")

rutas_posibles_ratios = [
    carpeta_salida / "perfil_ratios_valores.xlsx",
    carpeta_salida / "10_perfil_ratios_valores.xlsx",
    carpeta_salida / "07_perfil_ratios_valores.xlsx",
]

ruta_salida_lineas = carpeta_salida / "prueba_usuario_lineas_contables_sugeridas.xlsx"
ruta_salida_control = carpeta_salida / "prueba_usuario_control_cuadre.xlsx"
ruta_salida_plantilla = carpeta_salida / "plantilla.xlsx"
ruta_prueba_facturacion = base / "prueba de facturacion.xlsx"
ruta_salida_validacion = carpeta_salida / "validacion_prediccion_nit_plantilla.xlsx"

columnas_movimientos_validacion = {
    "fecha_mov": "FECHA",
    "tipo_doc": "TIPODOC",
    "numero_doc": "NUMDOC",
    "cuenta": "CUENTA",
    "nombre_cuenta": "NOM_CUENTA",
    "identidad": "IDENTIDADTERCERO",
    "nombre_tercero": "NOMBRETERCERO",
    "concepto": "CONCEPTO",
    "codigo_centro_costo": "CENTRO",
    "centro_costo": "C_C",
    "usuario": "CODIGO_USUARIO",
    "numero_movil": "DOC_FUENTE",
    "nombre_centro_costo": "NOM_CENTRO",
    "debito": "DEBITO",
    "credito": "CREDITO",
}

col_nit = "nit_proveedor_norm"
col_target = "target_plantilla_cuentas"
col_prediccion = "plantilla_predicha"
col_total = "payable_amount"
min_observaciones = 2

threshold_alta = 0.70
threshold_media = 0.50
threshold_baja = 0.30


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

    s = serie.astype("string").str.strip()

    mask_miles_punto = s.str.contains(r"^-?\d{1,3}(\.\d{3})+,\d+$", na=False)
    s.loc[mask_miles_punto] = (
        s.loc[mask_miles_punto]
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )

    mask_miles_coma = s.str.contains(r"^-?\d{1,3}(,\d{3})+\.\d+$", na=False)
    s.loc[mask_miles_coma] = s.loc[mask_miles_coma].str.replace(",", "", regex=False)

    mask_decimal_coma = (
        s.str.contains(",", na=False)
        & ~s.str.contains(r"\.", na=False)
        & s.str.contains(r"^-?\d+,\d+$", na=False)
    )
    s.loc[mask_decimal_coma] = s.loc[mask_decimal_coma].str.replace(",", ".", regex=False)

    mask_entero_miles_coma = s.str.contains(r"^-?\d{1,3}(,\d{3})+$", na=False)
    s.loc[mask_entero_miles_coma] = s.loc[mask_entero_miles_coma].str.replace(",", "", regex=False)

    mask_entero_miles_punto = s.str.contains(r"^-?\d{1,3}(\.\d{3})+$", na=False)
    s.loc[mask_entero_miles_punto] = s.loc[mask_entero_miles_punto].str.replace(".", "", regex=False)

    return (
        s.str.replace(r"[^0-9.\-]", "", regex=True)
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
    return texto.rsplit("_", 1)


def preparar_X(facturas: pd.DataFrame, columnas_modelo: list[str]) -> pd.DataFrame:
    X = facturas.copy()
    for col in columnas_modelo:
        if col not in X.columns:
            X[col] = None
    return X[columnas_modelo].copy()


def truncar_texto(valor, max_len: int) -> str:
    valor = limpiar_espacios(valor)
    return "" if valor is None else valor[:max_len]


def construir_concepto(numero_factura, proveedor, descripcion="", prefijo="", max_len: int = 70) -> str:
    partes = [
        truncar_texto(prefijo, 20),
        truncar_texto(numero_factura, 25),
        truncar_texto(proveedor, 15),
        truncar_texto(descripcion, 70),
    ]
    return " ".join([x for x in partes if x]).strip()[:max_len]


def normalizar_codigo_cuenta_serie(serie: pd.Series) -> pd.Series:
    return (
        serie.astype("string")
        .fillna("")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def entero_seguro(x) -> int:
    y = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    return int(round(0 if pd.isna(y) else float(y)))


def tomar_serie(df: pd.DataFrame, *cols: str) -> pd.Series:
    for col in cols:
        if col in df.columns:
            return df[col]
    return pd.Series([None] * len(df), index=df.index)


def cargar_catalogo_cuentas() -> pd.DataFrame:
    if not ruta_contabilidad.exists():
        print(f"No se encontró contabilidad para catálogo de cuentas:\n{ruta_contabilidad}")
        return pd.DataFrame(columns=["CUENTA", "NOMBRE"])

    df = cargar_excel_flexible(str(ruta_contabilidad), hoja=0)

    col_cuenta = "cuenta" if "cuenta" in df.columns else None
    col_nombre = next((c for c in ["nom_cuenta", "nombre_cuenta", "nombre"] if c in df.columns), None)

    if col_cuenta is None or col_nombre is None:
        print("No se pudieron identificar las columnas de cuenta/nombre en contabilidad.")
        return pd.DataFrame(columns=["CUENTA", "NOMBRE"])

    catalogo = df[[col_cuenta, col_nombre]].copy()
    catalogo.columns = ["CUENTA", "NOMBRE"]
    catalogo["CUENTA"] = normalizar_codigo_cuenta_serie(catalogo["CUENTA"])
    catalogo["NOMBRE"] = catalogo["NOMBRE"].astype("string").fillna("").str.strip()
    catalogo = catalogo.loc[catalogo["CUENTA"].ne("") & catalogo["NOMBRE"].ne("")].copy()

    if catalogo.empty:
        return pd.DataFrame(columns=["CUENTA", "NOMBRE"])

    return (
        catalogo.groupby("CUENTA", as_index=False)["NOMBRE"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
    )


def construir_perfil_fallback_por_plantilla(perfil: pd.DataFrame) -> pd.DataFrame:
    if perfil.empty:
        return pd.DataFrame(columns=[col_target, "cuenta_naturaleza", "ratio_mediano", "ratio_promedio", "ratio_std", "n_observaciones", "origen_ratio"])

    p = perfil.copy()
    for col in ["ratio_mediano", "ratio_promedio", "ratio_std", "n_observaciones"]:
        if col in p.columns:
            p[col] = numero_seguro(p[col])

    fallback = (
        p.groupby([col_target, "cuenta_naturaleza"], dropna=False)
        .agg(
            ratio_mediano=("ratio_mediano", "median"),
            ratio_promedio=("ratio_promedio", "mean"),
            ratio_std=("ratio_std", "median"),
            n_observaciones=("n_observaciones", "sum"),
        )
        .reset_index()
    )

    fallback = fallback.loc[fallback["n_observaciones"] >= min_observaciones].copy()
    fallback["origen_ratio"] = "FALLBACK_PLANTILLA"
    return fallback


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

    salida["confiabilidad_prediccion"] = np.select(
        [
            salida["confianza_prediccion"] >= threshold_alta,
            salida["confianza_prediccion"] >= threshold_media,
            salida["confianza_prediccion"] >= threshold_baja,
        ],
        ["ALTA", "MEDIA", "BAJA"],
        default="SIN_PREDICCION",
    )

    salida["generar_prediccion"] = salida["confianza_prediccion"] >= threshold_baja
    salida.loc[~salida["generar_prediccion"], col_prediccion] = None

    salida["estado_sugerencia"] = np.select(
        [
            salida["confiabilidad_prediccion"] == "ALTA",
            salida["confiabilidad_prediccion"] == "MEDIA",
            salida["confiabilidad_prediccion"] == "BAJA",
        ],
        ["SUGERENCIA_ALTA", "REVISION_RAPIDA", "REVISION_MANUAL"],
        default="SIN_PREDICCION",
    )

    return salida


def redondear_preservando_total(serie: pd.Series) -> pd.Series:
    valores = numero_seguro(serie).clip(lower=0)
    base = np.floor(valores).astype(int)
    faltan = int(round(float(valores.sum()))) - int(base.sum())

    if faltan > 0:
        fracciones = (valores - np.floor(valores)).sort_values(ascending=False)
        base.loc[fracciones.index[:faltan]] += 1

    return base.astype(int)


def ajustar_enteros_y_cuadre_por_factura(lineas: pd.DataFrame) -> pd.DataFrame:
    if lineas.empty:
        return lineas.copy()

    salida = lineas.copy()
    salida["debito_modelado"] = numero_seguro(salida["debito"])
    salida["credito_modelado"] = numero_seguro(salida["credito"])
    grupos = []

    for _, g in salida.groupby("id_factura", dropna=False, sort=False):
        g = g.copy()
        g["debito"] = redondear_preservando_total(g["debito_modelado"])
        g["credito"] = redondear_preservando_total(g["credito_modelado"])
        g["ajuste_redondeo_cuadre"] = 0

        diferencia = int(g["debito"].sum() - g["credito"].sum())

        if diferencia > 0:
            candidatos = g.index[g["credito"] > 0].tolist() or g.index.tolist()
            idx = g.loc[candidatos, "credito_modelado"].idxmax()
            g.loc[idx, "credito"] += diferencia
            g.loc[idx, "ajuste_redondeo_cuadre"] += diferencia
        elif diferencia < 0:
            ajuste = abs(diferencia)
            candidatos = g.index[g["debito"] > 0].tolist() or g.index.tolist()
            idx = g.loc[candidatos, "debito_modelado"].idxmax()
            g.loc[idx, "debito"] += ajuste
            g.loc[idx, "ajuste_redondeo_cuadre"] += ajuste

        g["debito"] = g["debito"].astype(int)
        g["credito"] = g["credito"].astype(int)
        grupos.append(g)

    return pd.concat(grupos, ignore_index=True)


def construir_lineas_desde_ratios(
    facturas_predichas: pd.DataFrame,
    perfil_ratios: pd.DataFrame,
) -> pd.DataFrame:
    columnas_salida = [
        "id_factura",
        "factura_completa",
        "fecha_emision",
        "descripcion_item_1",
        "nit_proveedor",
        col_nit,
        "nombre_proveedor",
        col_prediccion,
        "confianza_prediccion",
        "confiabilidad_prediccion",
        "estado_sugerencia",
        "origen_ratio",
        "cuenta_contable",
        "naturaleza",
        "cuenta_naturaleza",
        "ratio_mediano",
        "ratio_promedio",
        "ratio_std",
        "n_observaciones",
        "valor_base_calculo",
        "valor_sugerido",
        "debito_modelado",
        "credito_modelado",
        "debito",
        "credito",
        "ajuste_redondeo_cuadre",
        "tiene_ratio_historico",
        "alerta_prediccion",
    ]

    perfil = perfil_ratios.copy()
    facturas = facturas_predichas.copy()

    faltantes = [c for c in [col_nit, col_target, "cuenta_naturaleza", "ratio_mediano"] if c not in perfil.columns]
    if faltantes:
        raise ValueError("El perfil de ratios no tiene las columnas requeridas:\n" + "\n".join(faltantes))

    if "n_observaciones" in perfil.columns:
        perfil["n_observaciones"] = numero_seguro(perfil["n_observaciones"])
        perfil = perfil.loc[perfil["n_observaciones"] >= min_observaciones].copy()

    facturas[col_target] = facturas[col_prediccion]
    facturas = facturas.loc[
        facturas["generar_prediccion"].fillna(False)
        & facturas[col_target].notna()
        & facturas[col_target].astype("string").str.strip().ne("")
    ].copy()

    if facturas.empty:
        return pd.DataFrame(columns=columnas_salida)

    llaves_proveedor = [col_nit, col_target]
    perfil_proveedor = texto_llave(perfil.copy(), llaves_proveedor)
    perfil_proveedor["origen_ratio"] = "PROVEEDOR"

    facturas_base = texto_llave(facturas.copy(), llaves_proveedor)
    lineas_proveedor = facturas_base.merge(perfil_proveedor, on=llaves_proveedor, how="left", suffixes=("", "_ratio"))

    ids_con_proveedor = set(
        lineas_proveedor.loc[lineas_proveedor["cuenta_naturaleza"].notna(), "id_factura"].astype("string")
    )

    lineas_proveedor_ok = lineas_proveedor.loc[lineas_proveedor["cuenta_naturaleza"].notna()].copy()

    perfil_fallback = construir_perfil_fallback_por_plantilla(perfil)
    lineas_fallback_ok = pd.DataFrame(columns=lineas_proveedor_ok.columns)

    facturas_fallback = facturas.loc[~facturas["id_factura"].astype("string").isin(ids_con_proveedor)].copy()

    if not facturas_fallback.empty and not perfil_fallback.empty:
        facturas_fallback = texto_llave(facturas_fallback, [col_target])
        perfil_fallback = texto_llave(perfil_fallback, [col_target])

        lineas_fallback = facturas_fallback.merge(
            perfil_fallback,
            on=[col_target],
            how="left",
            suffixes=("", "_ratio"),
        )

        lineas_fallback_ok = lineas_fallback.loc[lineas_fallback["cuenta_naturaleza"].notna()].copy()

    ids_con_ratio = ids_con_proveedor | set(lineas_fallback_ok["id_factura"].astype("string"))
    lineas_ok = pd.concat([lineas_proveedor_ok, lineas_fallback_ok], ignore_index=True, sort=False)

    if not lineas_ok.empty:
        lineas_ok["valor_base_calculo"] = numero_seguro(lineas_ok[col_total])
        lineas_ok["ratio_mediano"] = numero_seguro(lineas_ok["ratio_mediano"])
        lineas_ok["valor_sugerido"] = lineas_ok["valor_base_calculo"] * lineas_ok["ratio_mediano"]

        lineas_ok[["cuenta_contable", "naturaleza"]] = lineas_ok["cuenta_naturaleza"].apply(
            lambda x: pd.Series(separar_cuenta_naturaleza(x))
        )

        lineas_ok["debito"] = np.where(lineas_ok["naturaleza"] == "D", lineas_ok["valor_sugerido"], 0)
        lineas_ok["credito"] = np.where(lineas_ok["naturaleza"] == "C", lineas_ok["valor_sugerido"], 0)
        lineas_ok["tiene_ratio_historico"] = True
        lineas_ok["alerta_prediccion"] = np.where(
            lineas_ok["origen_ratio"] == "FALLBACK_PLANTILLA",
            "Sin perfil del proveedor; se usó fallback por plantilla",
            "",
        )

        lineas_ok = ajustar_enteros_y_cuadre_por_factura(lineas_ok)
    else:
        lineas_ok = pd.DataFrame(columns=columnas_salida)

    facturas_sin_ratio = facturas.loc[~facturas["id_factura"].astype("string").isin(ids_con_ratio)].copy()

    if not facturas_sin_ratio.empty:
        facturas_sin_ratio["origen_ratio"] = "SIN_PERFIL"
        facturas_sin_ratio["cuenta_contable"] = None
        facturas_sin_ratio["naturaleza"] = None
        facturas_sin_ratio["cuenta_naturaleza"] = None
        facturas_sin_ratio["ratio_mediano"] = np.nan
        facturas_sin_ratio["ratio_promedio"] = np.nan
        facturas_sin_ratio["ratio_std"] = np.nan
        facturas_sin_ratio["n_observaciones"] = np.nan
        facturas_sin_ratio["valor_base_calculo"] = numero_seguro(facturas_sin_ratio[col_total])
        facturas_sin_ratio["valor_sugerido"] = 0
        facturas_sin_ratio["debito_modelado"] = 0
        facturas_sin_ratio["credito_modelado"] = 0
        facturas_sin_ratio["debito"] = 0
        facturas_sin_ratio["credito"] = 0
        facturas_sin_ratio["ajuste_redondeo_cuadre"] = 0
        facturas_sin_ratio["tiene_ratio_historico"] = False
        facturas_sin_ratio["alerta_prediccion"] = "Sin perfil del proveedor ni fallback por plantilla"

        lineas_final = pd.concat([lineas_ok, facturas_sin_ratio], ignore_index=True, sort=False)
    else:
        lineas_final = lineas_ok.copy()

    return lineas_final[[c for c in columnas_salida if c in lineas_final.columns]].copy()


def construir_control_cuadre(lineas: pd.DataFrame) -> pd.DataFrame:
    lineas_ok = lineas.loc[lineas["tiene_ratio_historico"].fillna(False)].copy()

    if lineas_ok.empty:
        return pd.DataFrame(columns=["id_factura", "total_debito", "total_credito", "n_lineas", "diferencia", "cuadra"])

    control = (
        lineas_ok.groupby("id_factura", dropna=False)
        .agg(
            total_debito=("debito", "sum"),
            total_credito=("credito", "sum"),
            n_lineas=("cuenta_contable", "count"),
        )
        .reset_index()
    )

    control["total_debito"] = control["total_debito"].astype(int)
    control["total_credito"] = control["total_credito"].astype(int)
    control["diferencia"] = control["total_debito"] - control["total_credito"]
    control["cuadra"] = control["diferencia"] == 0

    return control


def construir_archivo_plantilla(
    facturas_predichas: pd.DataFrame,
    lineas_sugeridas: pd.DataFrame,
    catalogo_cuentas: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_fact = facturas_predichas.copy().reset_index(drop=True)
    base_fact["NUMERO DOCUMENTO"] = np.arange(1, len(base_fact) + 1)
    base_fact["FECHA"] = pd.to_datetime(base_fact.get("fecha_emision"), errors="coerce").dt.date
    base_fact["TIPO DOCUMENTO"] = "FC"
    base_fact["IDENTIDAD"] = (
        base_fact.get("nit_proveedor", pd.Series(index=base_fact.index))
        .fillna(base_fact.get(col_nit, pd.Series(index=base_fact.index)))
        .astype("string")
        .fillna("")
        .str.strip()
    )
    base_fact["CONFIABILIDAD DE PREDICCION"] = (
        base_fact.get("confiabilidad_prediccion", pd.Series(index=base_fact.index))
        .astype("string")
        .fillna("")
        .str.strip()
    )

    base_fact["payable_amount"] = numero_seguro(base_fact.get("payable_amount", pd.Series(index=base_fact.index)))
    base_fact["iva_total"] = numero_seguro(base_fact.get("iva_total", pd.Series(index=base_fact.index)))
    base_fact["base_amount"] = numero_seguro(base_fact.get("base_amount", pd.Series(index=base_fact.index)))

    if "ica_total" in base_fact.columns:
        base_fact["ica_total"] = numero_seguro(base_fact["ica_total"])
    elif "inc_total" in base_fact.columns:
        base_fact["ica_total"] = numero_seguro(base_fact["inc_total"])
    else:
        base_fact["ica_total"] = 0

    columnas_base = [
        "id_factura",
        "FECHA",
        "TIPO DOCUMENTO",
        "NUMERO DOCUMENTO",
        "IDENTIDAD",
        "factura_completa",
        "nombre_proveedor",
        "descripcion_item_1",
        "payable_amount",
        "iva_total",
        "ica_total",
        "base_amount",
        "CONFIABILIDAD DE PREDICCION",
    ]
    base_fact = base_fact[[c for c in columnas_base if c in base_fact.columns]].copy()

    lineas_ok = lineas_sugeridas.loc[
        lineas_sugeridas.get("tiene_ratio_historico", pd.Series(index=lineas_sugeridas.index)).fillna(False)
    ].copy()

    if lineas_ok.empty:
        plantilla = pd.DataFrame(columns=[
            "FECHA", "TIPO DOCUMENTO", "NUMERO DOCUMENTO", "IDENTIDAD", "CUENTA",
            "NOMBRE", "CONCEPTO", "VALOR", "NATURALEZA", "CONFIABILIDAD DE PREDICCION",
        ])
        facturas_con_sugerencia = set()
    else:
        plantilla = lineas_ok.merge(
            base_fact[["id_factura", "NUMERO DOCUMENTO", "FECHA", "TIPO DOCUMENTO", "IDENTIDAD", "CONFIABILIDAD DE PREDICCION"]],
            on="id_factura",
            how="left",
        )

        plantilla["CUENTA"] = normalizar_codigo_cuenta_serie(
            plantilla.get("cuenta_contable", pd.Series(index=plantilla.index))
        )
        plantilla = plantilla.merge(catalogo_cuentas, on="CUENTA", how="left")
        plantilla["NOMBRE"] = plantilla.get("NOMBRE", pd.Series(index=plantilla.index)).astype("string").fillna("").str.strip()

        factura_serie = tomar_serie(plantilla, "factura_completa")
        proveedor_serie = tomar_serie(plantilla, "nombre_proveedor")
        descripcion_serie = tomar_serie(plantilla, "descripcion_item_1")

        plantilla["VALOR"] = np.where(
            numero_seguro(plantilla["debito"]) > 0,
            numero_seguro(plantilla["debito"]),
            numero_seguro(plantilla["credito"]),
        ).astype(int)

        plantilla["CONCEPTO"] = [
            construir_concepto(factura_serie.iloc[i], proveedor_serie.iloc[i], descripcion_serie.iloc[i])
            for i in range(len(plantilla))
        ]

        plantilla["NATURALEZA"] = plantilla.get("naturaleza", pd.Series(index=plantilla.index)).astype("string").fillna("").str.strip()

        plantilla = plantilla.loc[plantilla["CUENTA"].ne("") & (plantilla["VALOR"] > 0)].copy()
        plantilla = plantilla[
            [
                "FECHA",
                "TIPO DOCUMENTO",
                "NUMERO DOCUMENTO",
                "IDENTIDAD",
                "CUENTA",
                "NOMBRE",
                "CONCEPTO",
                "VALOR",
                "NATURALEZA",
                "CONFIABILIDAD DE PREDICCION",
            ]
        ].copy()

        facturas_con_sugerencia = set(lineas_ok["id_factura"].astype("string"))

    base_sin = base_fact.copy()
    base_sin["id_factura_str"] = base_sin["id_factura"].astype("string")
    base_sin = base_sin.loc[~base_sin["id_factura_str"].isin(facturas_con_sugerencia)].copy()

    componentes = [
        ("TOTAL", "payable_amount"),
        ("IVA", "iva_total"),
        ("ICA", "ica_total"),
        ("BASE", "base_amount"),
    ]

    filas_sin = []
    for _, fila in base_sin.iterrows():
        for etiqueta, col_valor in componentes:
            filas_sin.append({
                "FECHA": fila.get("FECHA"),
                "TIPO DOCUMENTO": fila.get("TIPO DOCUMENTO"),
                "NUMERO DOCUMENTO": fila.get("NUMERO DOCUMENTO"),
                "IDENTIDAD": fila.get("IDENTIDAD"),
                "CUENTA": "",
                "NOMBRE": "",
                "CONCEPTO": construir_concepto(
                    fila.get("factura_completa"),
                    fila.get("nombre_proveedor"),
                    "",
                    prefijo=etiqueta,
                ),
                "VALOR": entero_seguro(fila.get(col_valor, 0)),
                "NATURALEZA": "",
                "CONFIABILIDAD DE PREDICCION": fila.get("CONFIABILIDAD DE PREDICCION"),
            })

    sin_sugerencia = pd.DataFrame(filas_sin, columns=[
        "FECHA",
        "TIPO DOCUMENTO",
        "NUMERO DOCUMENTO",
        "IDENTIDAD",
        "CUENTA",
        "NOMBRE",
        "CONCEPTO",
        "VALOR",
        "NATURALEZA",
        "CONFIABILIDAD DE PREDICCION",
    ])

    return plantilla, sin_sugerencia


def exportar_archivo_plantilla(
    facturas_predichas: pd.DataFrame,
    lineas_sugeridas: pd.DataFrame,
    catalogo_cuentas: pd.DataFrame,
    ruta_salida: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    plantilla, sin_sugerencia = construir_archivo_plantilla(
        facturas_predichas=facturas_predichas,
        lineas_sugeridas=lineas_sugeridas,
        catalogo_cuentas=catalogo_cuentas,
    )

    with pd.ExcelWriter(ruta_salida, engine="openpyxl") as writer:
        plantilla.to_excel(writer, sheet_name="plantilla", index=False)
        sin_sugerencia.to_excel(writer, sheet_name="sin_sugerencia", index=False)

    return plantilla, sin_sugerencia


def serializar_plantilla_cuentas(valor) -> str:
    if pd.isna(valor):
        return ""

    if isinstance(valor, str):
        texto = valor.strip()
        if texto == "":
            return ""
        try:
            valor = ast.literal_eval(texto)
        except Exception:
            return texto

    if isinstance(valor, (tuple, list, set)):
        partes = []
        for x in valor:
            x = "" if pd.isna(x) else str(x).strip()
            if x != "":
                partes.append(x)
        return " | ".join(sorted(partes))

    texto = str(valor).strip()
    return texto


def construir_resumen_plantilla_predicha(
    lineas_sugeridas: pd.DataFrame,
) -> pd.DataFrame:
    columnas = [
        "id_factura",
        "nombre_proveedor_predicho",
        "nit_predicho_norm",
        "plantilla_cuentas_dc_predicha",
        "n_lineas_predichas",
    ]

    if lineas_sugeridas.empty:
        return pd.DataFrame(columns=columnas)

    base = lineas_sugeridas.loc[
        lineas_sugeridas.get(
            "tiene_ratio_historico",
            pd.Series(False, index=lineas_sugeridas.index),
        ).fillna(False)
    ].copy()

    if base.empty:
        return pd.DataFrame(columns=columnas)

    resumen = (
        base.groupby("id_factura", dropna=False)
        .agg(
            nombre_proveedor_predicho=("nombre_proveedor", lambda s: next(
                (
                    str(x).strip()
                    for x in s
                    if pd.notna(x) and str(x).strip() != ""
                ),
                "",
            )),
            nit_predicho_norm=(col_nit, lambda s: next(
                (
                    str(x).strip()
                    for x in s
                    if pd.notna(x) and str(x).strip() != ""
                ),
                "",
            )),
            plantilla_cuentas_dc_predicha=(
                "cuenta_naturaleza",
                lambda s: serializar_plantilla_cuentas(tuple(sorted(
                    str(x).strip()
                    for x in s
                    if pd.notna(x) and str(x).strip() != ""
                ))),
            ),
            n_lineas_predichas=("cuenta_naturaleza", "count"),
        )
        .reset_index()
    )

    return resumen[columnas].copy()


def construir_validacion_prediccion(
    ruta_facturas: Path,
    ruta_movimientos_reales: Path,
    lineas_sugeridas: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columnas_detalle = [
        "id_factura",
        "llave_factura",
        "llave_asiento",
        "estado_match",
        "nombre_tercero_real",
        "nit_real_norm",
        "plantilla_cuentas_dc_real",
        "nombre_proveedor_predicho",
        "nit_predicho_norm",
        "plantilla_cuentas_dc_predicha",
        "acierto_nit",
        "acierto_plantilla",
        "acierto_llave_nit_plantilla",
        "tiene_plantilla_predicha",
        "n_lineas_predichas",
    ]

    if not ruta_movimientos_reales.exists():
        detalle_vacio = pd.DataFrame(columns=columnas_detalle)
        resumen_vacio = pd.DataFrame(
            [{"indicador": "validacion_ejecutada", "valor": "NO_ARCHIVO_PRUEBA"}]
        )
        return detalle_vacio, resumen_vacio

    resultados_validacion = ejecutar_pipeline(
        ruta_facturas=str(ruta_facturas),
        ruta_movimientos=str(ruta_movimientos_reales),
        hoja_facturas=0,
        hoja_movimientos=0,
        empresa="cpabaas",
        columnas_movimientos=columnas_movimientos_validacion,
    )

    match_df = resultados_validacion["match_df"].copy()
    asientos_fc_df = resultados_validacion["asientos_fc_df"].copy()
    movimientos_fc_df = resultados_validacion["movimientos_fc_df"].copy()

    estados_ok = ["OK_UNICO", "OK_TEXTO", "OK_VALOR"]

    detalle_real = match_df.loc[
        match_df["estado_match"].isin(estados_ok)
        & match_df["llave_asiento"].notna(),
        ["id_factura", "llave_factura", "llave_asiento", "estado_match"],
    ].copy()

    if detalle_real.empty:
        detalle_vacio = pd.DataFrame(columns=columnas_detalle)
        resumen_vacio = pd.DataFrame(
            [{"indicador": "validacion_ejecutada", "valor": "SIN_MATCHES_OK"}]
        )
        return detalle_vacio, resumen_vacio

    asientos_reales = asientos_fc_df[
        ["llave_asiento", "nit_dominante_norm", "plantilla_cuentas_dc"]
    ].copy()

    terceros_reales = (
        movimientos_fc_df.groupby("llave_asiento_base", dropna=False)["nombre_tercero"]
        .agg(
            lambda s: next(
                (
                    str(x).strip()
                    for x in s
                    if pd.notna(x) and str(x).strip() != ""
                ),
                "",
            )
        )
        .reset_index()
        .rename(
            columns={
                "llave_asiento_base": "llave_asiento",
                "nombre_tercero": "nombre_tercero_real",
            }
        )
    )

    detalle_real = detalle_real.merge(
        asientos_reales,
        on="llave_asiento",
        how="left",
    ).merge(
        terceros_reales,
        on="llave_asiento",
        how="left",
    )

    detalle_real = detalle_real.rename(
        columns={
            "nit_dominante_norm": "nit_real_norm",
            "plantilla_cuentas_dc": "plantilla_cuentas_dc_real",
        }
    )

    detalle_real["nombre_tercero_real"] = (
        detalle_real["nombre_tercero_real"].astype("string").fillna("").str.strip()
    )
    detalle_real["nit_real_norm"] = (
        detalle_real["nit_real_norm"].astype("string").fillna("").str.strip()
    )
    detalle_real["plantilla_cuentas_dc_real"] = detalle_real[
        "plantilla_cuentas_dc_real"
    ].apply(serializar_plantilla_cuentas)

    plantilla_predicha = construir_resumen_plantilla_predicha(lineas_sugeridas)

    detalle = detalle_real.merge(
        plantilla_predicha,
        on="id_factura",
        how="left",
    )

    detalle["nombre_proveedor_predicho"] = (
        detalle["nombre_proveedor_predicho"].astype("string").fillna("").str.strip()
    )
    detalle["nit_predicho_norm"] = (
        detalle["nit_predicho_norm"].astype("string").fillna("").str.strip()
    )
    detalle["plantilla_cuentas_dc_predicha"] = (
        detalle["plantilla_cuentas_dc_predicha"].astype("string").fillna("").str.strip()
    )
    detalle["n_lineas_predichas"] = (
        pd.to_numeric(detalle["n_lineas_predichas"], errors="coerce").fillna(0).astype(int)
    )

    detalle["tiene_plantilla_predicha"] = (
        detalle["plantilla_cuentas_dc_predicha"].astype("string").str.strip().ne("")
    )
    detalle["acierto_nit"] = detalle["nit_real_norm"] == detalle["nit_predicho_norm"]
    detalle["acierto_plantilla"] = (
        detalle["plantilla_cuentas_dc_real"]
        == detalle["plantilla_cuentas_dc_predicha"]
    )
    detalle["acierto_llave_nit_plantilla"] = (
        detalle["acierto_nit"] & detalle["acierto_plantilla"]
    )

    total = len(detalle)
    aciertos_nit = int(detalle["acierto_nit"].sum())
    aciertos_plantilla = int(detalle["acierto_plantilla"].sum())
    aciertos_llave = int(detalle["acierto_llave_nit_plantilla"].sum())
    con_plantilla = int(detalle["tiene_plantilla_predicha"].sum())

    resumen = pd.DataFrame(
        [
            {"indicador": "validacion_ejecutada", "valor": "SI"},
            {"indicador": "asientos_validados", "valor": total},
            {"indicador": "asientos_con_plantilla_predicha", "valor": con_plantilla},
            {"indicador": "aciertos_nit", "valor": aciertos_nit},
            {"indicador": "aciertos_plantilla", "valor": aciertos_plantilla},
            {"indicador": "aciertos_llave_nit_plantilla", "valor": aciertos_llave},
            {
                "indicador": "pct_acierto_nit",
                "valor": round(aciertos_nit / total, 4) if total else np.nan,
            },
            {
                "indicador": "pct_acierto_plantilla",
                "valor": round(aciertos_plantilla / total, 4) if total else np.nan,
            },
            {
                "indicador": "pct_acierto_llave_nit_plantilla",
                "valor": round(aciertos_llave / total, 4) if total else np.nan,
            },
        ]
    )

    detalle = detalle[columnas_detalle].copy()

    return detalle, resumen


def exportar_validacion_prediccion(
    ruta_facturas: Path,
    ruta_movimientos_reales: Path,
    lineas_sugeridas: pd.DataFrame,
    ruta_salida: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detalle, resumen = construir_validacion_prediccion(
        ruta_facturas=ruta_facturas,
        ruta_movimientos_reales=ruta_movimientos_reales,
        lineas_sugeridas=lineas_sugeridas,
    )

    with pd.ExcelWriter(ruta_salida, engine="openpyxl") as writer:
        detalle.to_excel(writer, sheet_name="detalle_validacion", index=False)
        resumen.to_excel(writer, sheet_name="resumen_validacion", index=False)

    return detalle, resumen


# =========================================================
# EJECUCIÓN
# =========================================================

validar_archivo(
    ruta_facturas_nuevas,
    "No existe el archivo de facturas nuevas. Debe tener la misma estructura que base datos factura.xlsx.",
)

ruta_modelo = cargar_ruta_modelo()
columnas_modelo = cargar_columnas_modelo()
catalogo_cuentas = cargar_catalogo_cuentas()

print("Cargando modelo:")
print(ruta_modelo)

predictor = TabularPredictor.load(str(ruta_modelo))

def ejecutar_procesamiento_facturas(funcion, ruta_excel: str, hoja=0, empresa="cpabaas") -> pd.DataFrame:
    intentos = [
        {"ruta_excel": ruta_excel, "hoja": hoja, "empresa": empresa},
        {"archivo_excel": ruta_excel, "hoja": hoja, "empresa": empresa},
        {"ruta": ruta_excel, "hoja": hoja, "empresa": empresa},
    ]

    for kwargs in intentos:
        try:
            return funcion(**kwargs)
        except TypeError:
            pass

    try:
        return funcion(ruta_excel, hoja, empresa)
    except TypeError:
        return funcion(ruta_excel, hoja)


facturas_nuevas = ejecutar_procesamiento_facturas(
    funcion=procesar_facturas,
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

print("\nDistribución de sugerencias:")
print(facturas_predichas["estado_sugerencia"].value_counts(dropna=False))

print("\nDistribución de confiabilidad:")
print(facturas_predichas["confiabilidad_prediccion"].value_counts(dropna=False))

print("\nDistribución de plantillas predichas:")
print(facturas_predichas[col_prediccion].value_counts(dropna=False).head(20))


# =========================================================
# RATIOS Y CONTROL
# =========================================================

ruta_ratios = obtener_ruta_ratios()

if ruta_ratios is None:
    lineas_sugeridas = pd.DataFrame()
    print("\nNo se encontró perfil de ratios. Solo se exportará plantilla.")
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
    print("Líneas con ratio:", int(lineas_sugeridas["tiene_ratio_historico"].sum()) if "tiene_ratio_historico" in lineas_sugeridas.columns else 0)
    print("Líneas sin ratio:", int((~lineas_sugeridas["tiene_ratio_historico"]).sum()) if "tiene_ratio_historico" in lineas_sugeridas.columns and not lineas_sugeridas.empty else 0)

    if not lineas_sugeridas.empty:
        control_cuadre = construir_control_cuadre(lineas_sugeridas)
        control_cuadre.to_excel(ruta_salida_control, index=False)

        print("\nControl de cuadre exportado en:")
        print(ruta_salida_control)

        print("\nResumen control de cuadre:")
        print(control_cuadre["cuadra"].value_counts(dropna=False))

plantilla_df, sin_sugerencia_df = exportar_archivo_plantilla(
    facturas_predichas=facturas_predichas,
    lineas_sugeridas=lineas_sugeridas,
    catalogo_cuentas=catalogo_cuentas,
    ruta_salida=ruta_salida_plantilla,
)

print("\nArchivo plantilla exportado en:")
print(ruta_salida_plantilla)
print("Líneas hoja plantilla:", len(plantilla_df))
print("Líneas hoja sin_sugerencia:", len(sin_sugerencia_df))


# =========================================================
# VALIDACIÓN CONTRA PRUEBA DE FACTURACIÓN
# =========================================================

detalle_validacion_df, resumen_validacion_df = exportar_validacion_prediccion(
    ruta_facturas=ruta_facturas_nuevas,
    ruta_movimientos_reales=ruta_prueba_facturacion,
    lineas_sugeridas=lineas_sugeridas,
    ruta_salida=ruta_salida_validacion,
)

if not resumen_validacion_df.empty:
    print("\nValidación de predicción:")
    print(resumen_validacion_df)

if not detalle_validacion_df.empty:
    print("\nDetalle validación exportado en:")
    print(ruta_salida_validacion)
    pct_llave = resumen_validacion_df.loc[
        resumen_validacion_df["indicador"] == "pct_acierto_llave_nit_plantilla",
        "valor",
    ]
    if not pct_llave.empty:
        print(
            "% acierto llave nit+plantilla:",
            f"{float(pct_llave.iloc[0]) * 100:.2f}%",
        )


# =========================================================
# RESUMEN FINAL
# =========================================================

print("\nPredicción finalizada.")
print("Modelo usado:", ruta_modelo)
print("Archivo de facturas nuevas:", ruta_facturas_nuevas)
print("Facturas procesadas:", len(facturas_predichas))
print("Facturas con predicción:", int(facturas_predichas["generar_prediccion"].sum()))
print("Facturas sin predicción:", int((~facturas_predichas["generar_prediccion"]).sum()))
print("Archivo plantilla:", ruta_salida_plantilla)
print("Archivo validación:", ruta_salida_validacion)

if not lineas_sugeridas.empty:
    print("Archivo de líneas:", ruta_salida_lineas)
    print("Archivo de control:", ruta_salida_control)
else:
    print("Líneas sugeridas: 0")