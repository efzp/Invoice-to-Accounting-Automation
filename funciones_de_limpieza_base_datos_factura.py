import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


NS_XML_FACTURA = {
    "cac": "urn:oasis:names:specification:ubl:schema:xsd:CommonAggregateComponents-2",
    "cbc": "urn:oasis:names:specification:ubl:schema:xsd:CommonBasicComponents-2",
}

UMBRAL_TOKEN_MUY_COMUN_XML = 0.35
MIN_FACTURAS_TOKEN_XML = 2
MIN_LARGO_TOKEN_XML = 3


def es_nulo(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, str) and x.strip().lower() in {
        "",
        "nan",
        "none",
        "null",
    }:
        return True
    try:
        return bool(pd.isna(x))
    except Exception:
        return False


def limpiar_espacios(x: Any) -> Optional[str]:
    if es_nulo(x):
        return None
    x = str(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x if x else None


def quitar_tildes(x: Any) -> Optional[str]:
    if es_nulo(x):
        return None
    x = limpiar_espacios(x)
    if x is None:
        return None
    x = unicodedata.normalize("NFKD", str(x))
    x = "".join(c for c in x if not unicodedata.combining(c))
    return x if x else None


def normalizar_texto_basico(x: Any) -> Optional[str]:
    return limpiar_espacios(quitar_tildes(x))


def normalizar_texto_modelo(x: Any) -> Optional[str]:
    if es_nulo(x):
        return None
    x = quitar_tildes(x)
    if x is None:
        return None
    x = x.lower()
    x = re.sub(r"\s+", " ", x).strip()
    return x if x else None


def normalizar_alfanumerico(x: Any) -> Optional[str]:
    if es_nulo(x):
        return None
    x = quitar_tildes(x)
    if x is None:
        return None
    x = x.lower()
    x = re.sub(r"[^a-z0-9]", "", x)
    return x if x else None


def normalizar_nit(x: Any) -> Optional[str]:
    if es_nulo(x):
        return None
    x = str(x)
    x = re.sub(r"[^0-9]", "", x)
    return x if x else None


def cargar_spacy_es():
    try:
        import spacy
    except ImportError as exc:
        raise ImportError(
            "Falta instalar spaCy. Ejecuta: python -m pip install spacy"
        ) from exc

    try:
        return spacy.load("es_core_news_sm")
    except OSError:
        return spacy.blank("es")


def texto_xml(elemento) -> str:
    if elemento is None:
        return ""
    return " ".join(" ".join(elemento.itertext()).split())


def es_token_numerico_spacy(token) -> bool:
    texto = token.text.strip()
    return token.like_num or bool(re.fullmatch(r"[\d.,/%\-]+", texto))


def tokenizar_descripcion_xml(
    texto: str,
    nlp,
    min_largo_token: int = MIN_LARGO_TOKEN_XML,
) -> list[str]:
    doc = nlp(str(texto).lower())
    tokens = []

    for token in doc:
        valor = token.lemma_ if token.lemma_ else token.text
        valor = valor.strip().lower()

        if (
            not valor
            or token.is_stop
            or token.is_punct
            or token.is_space
            or es_token_numerico_spacy(token)
            or len(valor) < min_largo_token
        ):
            continue

        tokens.append(valor)

    return tokens


def filtrar_tokens_xml_por_frecuencia(
    filas: list[dict],
    umbral_token_muy_comun: float = UMBRAL_TOKEN_MUY_COMUN_XML,
    min_facturas_token: int = MIN_FACTURAS_TOKEN_XML,
) -> list[dict]:
    total_facturas = len(filas)
    frecuencia_documental = {}

    for fila in filas:
        for token in set(fila["tokens_descripcion"]):
            frecuencia_documental[token] = frecuencia_documental.get(token, 0) + 1

    max_facturas_token = max(1, int(total_facturas * umbral_token_muy_comun))

    for fila in filas:
        tokens = [
            token
            for token in fila["tokens_descripcion"]
            if min_facturas_token <= frecuencia_documental[token] <= max_facturas_token
        ]
        fila["descripciones_lineas_limpia"] = " ".join(tokens)
        del fila["tokens_descripcion"]

    return filas


def llave_orden_codigo_xml(codigo: str):
    partes = re.split(r"(\d+)", str(codigo))
    return [int(x) if x.isdigit() else x.lower() for x in partes]


def ordenar_lineas_xml(lineas: list[dict]) -> list[dict]:
    return sorted(
        lineas,
        key=lambda x: (
            x["codigo"] == "",
            llave_orden_codigo_xml(x["codigo"]),
            x["descripcion"],
        ),
    )


def extraer_lineas_xml_factura(root: ET.Element) -> list[dict]:
    lineas = root.findall(".//cac:InvoiceLine", NS_XML_FACTURA) + root.findall(
        ".//cac:CreditNoteLine",
        NS_XML_FACTURA,
    )
    datos = []

    for linea in lineas:
        textos = [
            texto_xml(x)
            for x in linea.findall("./cac:Item/cbc:Description", NS_XML_FACTURA)
        ]

        if not any(textos):
            textos = [texto_xml(linea.find("./cac:Item/cbc:Name", NS_XML_FACTURA))]

        textos = [x for x in textos if x]
        if textos:
            datos.append(
                {
                    "codigo": texto_xml(
                        linea.find(
                            "./cac:Item/cac:StandardItemIdentification/cbc:ID",
                            NS_XML_FACTURA,
                        )
                    ),
                    "descripcion": " / ".join(textos),
                }
            )

    return ordenar_lineas_xml(datos)


def concatenar_unicos(valores: list[str]) -> str:
    return " | ".join(dict.fromkeys(x for x in valores if x))


def extraer_cufe_xml(root: ET.Element) -> str:
    return texto_xml(root.find("./cbc:UUID", NS_XML_FACTURA))


def construir_base_descripciones_xml(
    carpeta_xml: str | Path,
    incluir_original: bool = False,
    umbral_token_muy_comun: float = UMBRAL_TOKEN_MUY_COMUN_XML,
    min_facturas_token: int = MIN_FACTURAS_TOKEN_XML,
    min_largo_token: int = MIN_LARGO_TOKEN_XML,
) -> pd.DataFrame:
    carpeta_xml = Path(carpeta_xml)
    nlp = cargar_spacy_es()
    filas = []

    for ruta_xml in sorted(carpeta_xml.glob("*.xml")):
        root = ET.parse(ruta_xml).getroot()
        lineas = extraer_lineas_xml_factura(root)
        descripcion_original = concatenar_unicos([x["descripcion"] for x in lineas])
        fila = {
            "cufe": extraer_cufe_xml(root),
            "standard_item_identification": concatenar_unicos([x["codigo"] for x in lineas]),
            "tokens_descripcion": tokenizar_descripcion_xml(
                descripcion_original,
                nlp,
                min_largo_token=min_largo_token,
            ),
        }
        if incluir_original:
            fila["descripciones_lineas_original"] = descripcion_original
        filas.append(fila)

    df = pd.DataFrame(
        filtrar_tokens_xml_por_frecuencia(
            filas,
            umbral_token_muy_comun=umbral_token_muy_comun,
            min_facturas_token=min_facturas_token,
        )
    )
    if df.empty:
        return pd.DataFrame(
            columns=[
                "cufe",
                "standard_item_identification",
                "descripciones_lineas_limpia",
            ]
        )

    df = df[df["cufe"].astype(str).str.strip().ne("")]
    return df.drop_duplicates(subset="cufe", keep="first").reset_index(drop=True)


def safe_to_numeric(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(serie, errors="coerce")


def limpiar_texto_numerico(x: Any) -> Optional[str]:
    if es_nulo(x):
        return None
    x = str(x).strip()
    x = x.replace("\u00A0", " ")
    x = x.replace(" ", "")
    x = x.replace("$", "")
    x = x.replace("%", "")
    x = x.replace(",", "")
    return x if x else None


def safe_float(x: Any) -> float:
    if es_nulo(x):
        return np.nan
    x = limpiar_texto_numerico(x)
    if x is None:
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def normalizar_valor_monetario(x: Any) -> float:
    return safe_float(x)


def safe_to_datetime(serie: pd.Series) -> pd.Series:
    return pd.to_datetime(serie, errors="coerce")


def safe_datetime(x: Any) -> pd.Timestamp:
    if es_nulo(x):
        return pd.NaT
    try:
        return pd.to_datetime(x, errors="coerce")
    except Exception:
        return pd.NaT


def detectar_prefijo_factura(factura_norm: Any) -> Optional[str]:
    if es_nulo(factura_norm):
        return None
    factura_norm = str(factura_norm).strip().lower()
    m = re.match(r"^[a-z]+", factura_norm)
    return m.group(0) if m else None


def coalesce_texto(*args: Any) -> Optional[str]:
    for x in args:
        if not es_nulo(x):
            x = limpiar_espacios(x)
            if x:
                return x
    return None


def coalesce_alfa_num(*args: Any) -> Optional[str]:
    for x in args:
        y = normalizar_alfanumerico(x)
        if y:
            return y
    return None


def coalesce_fecha(*args: Any) -> pd.Timestamp:
    for x in args:
        y = pd.to_datetime(x, errors="coerce")
        if not pd.isna(y):
            return y
    return pd.NaT


def construir_descripcion_modelo(row: pd.Series) -> Optional[str]:
    posibles = [
        row.get("proveedor_nombre_limpio"),
        row.get("ciudad_proveedor_limpia"),
        row.get("item_descripcion_1_limpia"),
        row.get("factura_completa_limpia"),
    ]
    partes = [limpiar_espacios(x) for x in posibles if not es_nulo(x)]
    partes = [x for x in partes if x]
    return " | ".join(partes) if partes else None


def normalizar_nombre_columna(col: Any) -> str:
    col = "" if col is None else str(col)
    col = quitar_tildes(col) or ""
    col = col.lower().strip()
    col = re.sub(r"[^a-z0-9]+", "_", col)
    col = re.sub(r"_+", "_", col).strip("_")
    return col


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


def anio_a_texto(anio) -> str:
    if pd.isna(anio):
        return "-1"
    return str(int(anio))


def valor_a_texto_llave(x) -> str:
    if es_nulo(x):
        return ""
    return str(x)


def construir_llave_factura(
    nit_proveedor_norm: Any,
    factura_match_norm: Any,
) -> str:
    return (
        valor_a_texto_llave(nit_proveedor_norm)
        + "|"
        + valor_a_texto_llave(factura_match_norm)
    )


def construir_llave_asiento(
    anio: Any,
    tipo_doc_norm: Any,
    numero_doc_norm: Any,
) -> str:
    return (
        anio_a_texto(anio)
        + "|"
        + valor_a_texto_llave(tipo_doc_norm)
        + "|"
        + valor_a_texto_llave(numero_doc_norm)
    )


def normalizar_flag_binaria(x: Any) -> int:
    if pd.isna(x):
        return 0

    if isinstance(x, (bool, np.bool_)):
        return int(x)

    if isinstance(x, (int, float, np.integer, np.floating)):
        if pd.isna(x):
            return 0
        return int(float(x) != 0)

    x = str(x).strip().lower()

    if x in {"", "nan", "none", "<na>", "null"}:
        return 0

    if x in {"1", "true", "verdadero", "si", "sí", "yes", "y", "x"}:
        return 1

    if x in {"0", "false", "falso", "no", "n"}:
        return 0

    return 0


def normalizar_lista_cuentas(df_asiento: pd.DataFrame) -> tuple:
    cuentas = []

    for _, row in df_asiento.iterrows():
        cuenta = row.get("cuenta_limpia", row.get("cuenta"))

        if es_nulo(cuenta):
            continue

        cuenta = str(cuenta).strip()

        if cuenta:
            cuentas.append(cuenta)

    return tuple(sorted(cuentas))


def normalizar_lista_cuentas_dc(df_asiento: pd.DataFrame) -> tuple:
    pares = []

    for _, row in df_asiento.iterrows():
        cuenta = row.get("cuenta_limpia", row.get("cuenta"))

        if es_nulo(cuenta):
            continue

        cuenta = str(cuenta).strip()

        debito = row.get("debito", 0)
        credito = row.get("credito", 0)

        debito = 0 if pd.isna(debito) else debito
        credito = 0 if pd.isna(credito) else credito

        if debito > 0 and credito > 0:
            naturaleza = "DC"
        elif debito > 0:
            naturaleza = "D"
        elif credito > 0:
            naturaleza = "C"
        else:
            naturaleza = "N"

        pares.append(f"{cuenta}_{naturaleza}")

    return tuple(sorted(pares))


def texto_contiene_factura(concepto_norm: Any, factura_norm: Any) -> bool:
    if es_nulo(concepto_norm) or es_nulo(factura_norm):
        return False
    return str(factura_norm) in str(concepto_norm)
