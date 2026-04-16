import pandas as pd

from procesamiento_facturas_contabilidad_match import ejecutar_pipeline

# =========================================================
# PRUEBA DE EJECUCIÓN DEL PIPELINE
# =========================================================

ruta_facturas = r"C:/Users/eduar/OneDrive - TCP BAAS S.A.S/Automatización/facturas_bigquery/base datos factura.xlsx"
ruta_contabilidad = r"C:/Users/eduar/OneDrive - TCP BAAS S.A.S/Automatización/facturas_bigquery/Contabilidad.xlsx"
carpeta_salida = r"C:/Users/eduar/OneDrive - TCP BAAS S.A.S/Automatización/facturas_bigquery/resultados"

resultados = ejecutar_pipeline(
    ruta_facturas=ruta_facturas,
    ruta_movimientos=ruta_contabilidad,
    hoja_facturas=0,
    hoja_movimientos="CPA",
    empresa="empresa_demo",
    columnas_movimientos={
        "fecha_mov": "FECHA",
        "tipo_doc": "TIPO DOC.",
        "numero_doc": "NÚMERO DOC.",
        "cuenta": "CUENTA",
        "nombre_cuenta": "NOMBRE CUENTA",
        "identidad": "IDENTIDAD",
        "nombre_tercero": "NOMBRE DEL TERCERO",
        "concepto": "CONCEPTO",
        "codigo_centro_costo": "C. DE COSTO",
        "centro_costo": "CENTRO DE COSTO",
        "usuario": "USUARIO",
        "numero_movil": "NÚMERO MÓVIL",
        "nombre_centro_costo": "NOMBRE C. DE COSTO",
        "debito": "DEBITO",
        "credito": "CREDITO",
    }
)

# =========================================================
# RESUMEN RÁPIDO EN CONSOLA
# =========================================================

print("Facturas:", resultados["facturas_df"].shape)
print("Movimientos:", resultados["movimientos_df"].shape)
print("Movimientos FC:", resultados["movimientos_fc_df"].shape)
print("Asientos FC:", resultados["asientos_fc_df"].shape)
print("Candidatos:", resultados["candidatos_df"].shape)
print("Matches:", resultados["match_df"].shape)
print("Dataset modelo:", resultados["dataset_modelo_df"].shape)
print("Dataset AutoGluon:", resultados["dataset_autogluon_df"].shape)
print("Líneas históricas valores:", resultados["lineas_historicas_valores_df"].shape)
print("Perfil ratios valores:", resultados["perfil_ratios_valores_df"].shape)

print("\nResumen estado_match:")
print(resultados["match_df"]["estado_match"].value_counts(dropna=False))

print("\nPrimeros matches:")
print(resultados["match_df"].head())

# =========================================================
# EXPORTAR RESULTADOS DE PRUEBA
# =========================================================

resultados["facturas_df"].to_excel(f"{carpeta_salida}\\01_facturas_limpias.xlsx", index=False)
resultados["movimientos_df"].to_excel(f"{carpeta_salida}\\02_movimientos_limpios.xlsx", index=False)
resultados["movimientos_fc_df"].to_excel(f"{carpeta_salida}\\03_movimientos_fc.xlsx", index=False)
resultados["asientos_fc_df"].to_excel(f"{carpeta_salida}\\04_asientos_fc.xlsx", index=False)
resultados["candidatos_df"].to_excel(f"{carpeta_salida}\\05_candidatos_match.xlsx", index=False)
resultados["dataset_modelo_df"].to_excel(f"{carpeta_salida}\\07_dataset_modelo.xlsx", index=False)
resultados["dataset_autogluon_df"].to_excel(f"{carpeta_salida}\\08_dataset_autogluon.xlsx", index=False)
resultados["lineas_historicas_valores_df"].to_excel(f"{carpeta_salida}\\09_lineas_historicas_valores.xlsx", index=False)
resultados["perfil_ratios_valores_df"].to_excel(f"{carpeta_salida}\\10_perfil_ratios_valores.xlsx", index=False)

# =========================================================
# RESUMEN DE MATCH
# =========================================================

match_df = resultados["match_df"].copy()
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

resumen_general = pd.DataFrame({
    "categoria": ["MATCH_OK", "NO_MATCH_OK", "TOTAL"],
    "cantidad": [
        match_df["estado_match"].isin(estados_ok).sum(),
        (~match_df["estado_match"].isin(estados_ok)).sum(),
        total_registros
    ]
})

resumen_general["porcentaje"] = (
    resumen_general["cantidad"] / total_registros
).round(4)

resumen_general["porcentaje_label"] = (
    resumen_general["porcentaje"] * 100
).round(2).astype(str) + "%"

# =========================================================
# EXPORTAR MATCH CON VARIAS PESTAÑAS
# =========================================================

ruta_match_excel = f"{carpeta_salida}\\06_match_factura_asiento.xlsx"

with pd.ExcelWriter(ruta_match_excel, engine="openpyxl") as writer:
    match_df.to_excel(writer, sheet_name="detalle_match", index=False)
    resumen_match.to_excel(writer, sheet_name="resumen_match", index=False)
    resumen_general.to_excel(writer, sheet_name="resumen_general", index=False)

# =========================================================
# IMPRESIÓN FINAL
# =========================================================

print("\nResumen general:")
print(resumen_general)

print("\nArchivos exportados correctamente en resultados.")