from pathlib import Path

from funciones_de_limpieza_base_datos_factura import construir_base_descripciones_xml


BASE = Path(__file__).resolve().parent / "facturas_bigquery"
CARPETA_XML = BASE / "XML definitivos"
RUTA_SALIDA = BASE / "resultados" / "descripciones_lineas_xml.xlsx"


def main() -> None:
    if not CARPETA_XML.exists():
        raise FileNotFoundError(f"No existe la carpeta de XML:\n{CARPETA_XML}")

    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    df = construir_base_descripciones_xml(CARPETA_XML, incluir_original=True)
    df.to_excel(RUTA_SALIDA, index=False)

    print(f"Facturas exportadas: {len(df)}")
    print(f"Salida: {RUTA_SALIDA}")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
