"""
funciones_airbnb.py
Funciones compartidas por los proyectos 1 y 4 del TP de Inside Airbnb (Buenos Aires).

Acá vive todo lo que se repite entre los dos proyectos:
- carga del archivo listings.csv.gz
- conversión del precio de texto a número
- limpieza (faltantes, registros imposibles, extremos)
- métricas de evaluación (MAE, RMSE, R²)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ruta del dataset: listings.csv.gz SIN descomprimir.
# Se busca primero en data/ (lo que sugiere la guía) y si no, al lado del código.
CARPETA_TP = Path(__file__).parent
RUTA_DATOS = next((r for r in [CARPETA_TP / "data" / "listings.csv.gz",
                               CARPETA_TP / "listings.csv.gz"] if r.exists()),
                  CARPETA_TP / "data" / "listings.csv.gz")

# Parámetros comunes que pide la consigna
TEST_SIZE = 0.25        # 75 % entrenamiento / 25 % prueba
RANDOM_STATE = 42       # semilla fija para que la partición sea reproducible
PERCENTIL_INF = 0.01    # recortamos el 1 % más barato...
PERCENTIL_SUP = 0.99    # ...y el 1 % más caro


def cargar_datos(ruta=RUTA_DATOS):
    """Lee el CSV comprimido. Si no está, corta con un mensaje claro."""
    if not Path(ruta).exists():
        raise FileNotFoundError(
            f"No se encontró {ruta}. Descargá listings.csv.gz de Inside Airbnb "
            "(Buenos Aires, Detailed Listings) y guardalo en la carpeta data/."
        )
    df = pd.read_csv(ruta, compression="gzip", low_memory=False)
    print(f"Archivo cargado: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    return df


def verificar_columnas(df, columnas):
    """Comprueba que existan las columnas que necesita el ejercicio."""
    faltantes = [c for c in columnas if c not in df.columns]
    if faltantes:
        raise KeyError(f"Faltan columnas en el archivo: {faltantes}")
    print(f"Columnas verificadas: {columnas}")


def convertir_precio(serie):
    """Convierte textos como '$106,393.00' a número (106393.0).

    Se borra todo lo que no sea dígito, punto o signo menos.
    Lo que no se pueda convertir queda como NaN (errors='coerce').
    """
    limpio = serie.astype(str).str.replace(r"[^0-9.-]", "", regex=True)
    return pd.to_numeric(limpio, errors="coerce")


def limpiar(df, columnas_x, reglas_minimas):
    """Limpieza en tres pasos, informando cuántas filas quedan en cada uno.

    columnas_x:      variables de entrada del ejercicio
    reglas_minimas:  dict {columna: valor mínimo válido} para descartar
                     registros imposibles (ej. 0 huéspedes)
    Devuelve el DataFrame limpio y una tabla con el conteo por paso.
    """
    pasos = [("Archivo completo", len(df))]

    datos = df[columnas_x + ["price"]].copy()
    datos["price"] = convertir_precio(datos["price"])
    for col in columnas_x:
        datos[col] = pd.to_numeric(datos[col], errors="coerce")

    # 1) Faltantes: se eliminan (no se imputan), solo en las columnas del caso
    datos = datos.dropna()
    pasos.append(("Sin faltantes en las columnas del caso", len(datos)))

    # 2) Registros imposibles: precio <= 0 y valores fuera de rango lógico
    mascara = datos["price"] > 0
    for col, minimo in reglas_minimas.items():
        mascara &= datos[col] >= minimo
    datos = datos[mascara]
    pasos.append(("Sin registros imposibles", len(datos)))

    # 3) Extremos: nos quedamos con el 98 % central del precio (percentiles 1 y 99).
    #    Hay anuncios con precios absurdos (errores de carga o lujo extremo)
    #    que tirarían la recta / los promedios de vecinos hacia arriba.
    p_inf = datos["price"].quantile(PERCENTIL_INF)
    p_sup = datos["price"].quantile(PERCENTIL_SUP)
    datos = datos[datos["price"].between(p_inf, p_sup)]
    pasos.append((f"Sin extremos (precio entre {p_inf:,.0f} y {p_sup:,.0f})", len(datos)))

    tabla_pasos = pd.DataFrame(pasos, columns=["Paso", "Filas"])
    print("\nLimpieza:")
    print(tabla_pasos.to_string(index=False))
    return datos, tabla_pasos


def metricas(y_real, y_pred):
    """Devuelve MAE, RMSE y R² en un diccionario."""
    return {
        "MAE": mean_absolute_error(y_real, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_real, y_pred)),
        "R2": r2_score(y_real, y_pred),
    }


def tabla_metricas(resultados):
    """Arma una tabla prolija a partir de {nombre_modelo: dict_metricas}."""
    tabla = pd.DataFrame(resultados).T
    return tabla.round({"MAE": 0, "RMSE": 0, "R2": 3})
