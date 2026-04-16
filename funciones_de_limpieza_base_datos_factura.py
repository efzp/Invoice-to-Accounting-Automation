import re
import unicodedata
from typing import Any, Optional

import numpy as np
import pandas as pd


# =========================================================
# 1. VALIDACIÓN BÁSICA
# =========================================================

def es_nulo(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, str) and x.strip().lower() in {"", "nan", "none", "null"}:
        return True
    try:
        return bool(pd.isna(x))
    except Exception:
        return False


# =========================================================
# 2. LIMPIEZA DE TEXTO
# =========================================================

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


# =========================================================
# 3. CONVERSIONES NUMÉRICAS Y FECHAS
# =========================================================

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


# =========================================================
# 4. FUNCIONES DE APOYO
# =========================================================

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


# =========================================================
# 5. FUNCIONES PARA PLANTILLAS CONTABLES
# =========================================================

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