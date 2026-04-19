import pandas as pd
from pathlib import Path

from procesamiento_facturas_contabilidad_match import ejecutar_pipeline


# =========================================================
# PIPELINE
# =========================================================

base = Path(r"C:/Users/eduar/OneDrive - TCP BAAS S.A.S/Automatización/facturas_bigquery")
carpeta_salida = base / "resultados"
carpeta_salida.mkdir(parents=True, exist_ok=True)

ruta_facturas = base / "base datos factura.xlsx"
ruta_contabilidad = base / "contabilidad.xlsx"

columnas_movimientos = {
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

resultados = ejecutar_pipeline(
    ruta_facturas=str(ruta_facturas),
    ruta_movimientos=str(ruta_contabilidad),
    hoja_facturas=0,
    hoja_movimientos=0,
    empresa="empresa_demo",
    columnas_movimientos=columnas_movimientos,
)


# =========================================================
# RESUMEN EN CONSOLA
# =========================================================

tablas_resumen = {
    "Facturas": "facturas_df",
    "Movimientos": "movimientos_df",
    "Movimientos FC": "movimientos_fc_df",
    "Asientos FC": "asientos_fc_df",
    "Candidatos": "candidatos_df",
    "Matches": "match_df",
    "Dataset modelo": "dataset_modelo_df",
    "Dataset AutoGluon": "dataset_autogluon_df",
    "Líneas históricas valores": "lineas_historicas_valores_df",
    "Perfil ratios valores": "perfil_ratios_valores_df",
}

for nombre, clave in tablas_resumen.items():
    print(f"{nombre}: {resultados[clave].shape}")

match_df = resultados["match_df"].copy()

print("\nResumen estado_match:")
print(match_df["estado_match"].value_counts(dropna=False))

print("\nPrimeros matches:")
print(match_df.head())


# =========================================================
# RESUMEN DE MATCH
# =========================================================

total_registros = len(match_df)

resumen_match = (
    match_df["estado_match"]
    .fillna("SIN_ESTADO")
    .value_counts(dropna=False)
    .rename_axis("estado_match")
    .reset_index(name="cantidad")
)

resumen_match["porcentaje"] = (
    resumen_match["cantidad"] / total_registros
).round(4)

resumen_match["porcentaje_label"] = (
    resumen_match["porcentaje"] * 100
).round(2).astype(str) + "%"

estados_ok = ["OK_UNICO", "OK_TEXTO", "OK_VALOR"]

cantidad_ok = match_df["estado_match"].isin(estados_ok).sum()
cantidad_no_ok = total_registros - cantidad_ok

resumen_general = pd.DataFrame({
    "categoria": ["MATCH_OK", "NO_MATCH_OK", "TOTAL"],
    "cantidad": [cantidad_ok, cantidad_no_ok, total_registros],
})

resumen_general["porcentaje"] = (
    resumen_general["cantidad"] / total_registros
).round(4)

resumen_general["porcentaje_label"] = (
    resumen_general["porcentaje"] * 100
).round(2).astype(str) + "%"

print("\nResumen general:")
print(resumen_general)


# =========================================================
# EXPORTAR RESULTADOS
# =========================================================

exportaciones = {
    "facturas_limpias.xlsx": "facturas_df",
    "movimientos_limpios.xlsx": "movimientos_df",
    "movimientos_fc.xlsx": "movimientos_fc_df",
    "asientos_fc.xlsx": "asientos_fc_df",
    "candidatos_match.xlsx": "candidatos_df",
    "dataset_modelo.xlsx": "dataset_modelo_df",
    "dataset_autogluon.xlsx": "dataset_autogluon_df",
    "lineas_historicas_valores.xlsx": "lineas_historicas_valores_df",
    "perfil_ratios_valores.xlsx": "perfil_ratios_valores_df",
}

for archivo, clave in exportaciones.items():
    resultados[clave].to_excel(carpeta_salida / archivo, index=False)

ruta_match_excel = carpeta_salida / "match_factura_asiento.xlsx"

with pd.ExcelWriter(ruta_match_excel, engine="openpyxl") as writer:
    match_df.to_excel(writer, sheet_name="detalle_match", index=False)
    resumen_match.to_excel(writer, sheet_name="resumen_match", index=False)
    resumen_general.to_excel(writer, sheet_name="resumen_general", index=False)

print("\nArchivos exportados correctamente en:")
print(carpeta_salida)