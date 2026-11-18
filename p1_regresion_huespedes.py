"""
p1_regresion_huespedes.py
Proyecto 1 — Capacidad del alojamiento y precio nocturno.

Caso: se quiere una regla SENCILLA y EXPLICABLE para estimar cuánto cambia
el precio cuando aumenta la cantidad máxima de huéspedes (accommodates).

Modelo elegido: REGRESIÓN LINEAL SIMPLE.
  - El objetivo (precio) es un número continuo -> problema de regresión.
  - Hay una sola variable de entrada -> regresión "simple".
  - Se pide una regla explicable de "cuánto cambia" -> la pendiente de la
    recta responde exactamente eso: pesos extra por cada huésped adicional.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # permite guardar gráficos sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from funciones_airbnb import (RANDOM_STATE, TEST_SIZE, cargar_datos, limpiar, metricas,
                   tabla_metricas, verificar_columnas)

CARPETA_FIG = Path(__file__).parent / "graficos"
CARPETA_FIG.mkdir(exist_ok=True)

X_COLS = ["accommodates"]
CASO_NUEVO = 4  # huéspedes


def correr_p1():
    print("=" * 60)
    print("PROYECTO 1 — Regresión lineal simple: accommodates -> price")
    print("=" * 60)

    # 1) Carga y verificación de columnas
    df = cargar_datos()
    verificar_columnas(df, X_COLS + ["price"])

    # 2-3) Limpieza: 0 huéspedes es imposible, por eso mínimo 1
    datos, _ = limpiar(df, X_COLS, reglas_minimas={"accommodates": 1})

    precio = datos["price"]
    print(f"\nPrecio medio: {precio.mean():,.0f} | mediana: {precio.median():,.0f} ARS")
    print(f"Rango de capacidad observado: {datos['accommodates'].min():.0f} "
          f"a {datos['accommodates'].max():.0f} huéspedes")

    # 4) División train/test 75/25 con semilla fija.
    #    X va con doble corchete -> DataFrame 2D, que es lo que espera sklearn.
    X = datos[X_COLS]
    y = datos["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # 5) Entrenamiento: SOLO con train
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    pendiente = modelo.coef_[0]
    intercepto = modelo.intercept_
    print(f"\nRecta: precio = {intercepto:,.0f} + {pendiente:,.0f} x huéspedes")
    print(f"  Pendiente: cada huésped extra se asocia a ~{pendiente:,.0f} ARS más por noche")

    # Modelo base: siempre predice el promedio de train. Sirve de referencia
    # mínima: si no le ganamos a esto, el modelo no aporta nada.
    base = DummyRegressor(strategy="mean").fit(X_train, y_train)

    # 6-7) Predicción y evaluación sobre test
    pred_test = modelo.predict(X_test)
    resultados = {
        "Regresión lineal (test)": metricas(y_test, pred_test),
        "Regresión lineal (train)": metricas(y_train, modelo.predict(X_train)),
        "Modelo base - promedio (test)": metricas(y_test, base.predict(X_test)),
    }
    tabla = tabla_metricas(resultados)
    print("\nMétricas:")
    print(tabla.to_string())

    # 8) Caso nuevo: 4 huéspedes
    nuevo = pd.DataFrame({"accommodates": [CASO_NUEVO]})
    estimacion = modelo.predict(nuevo)[0]
    print(f"\nEstimación para {CASO_NUEVO} huéspedes: {estimacion:,.0f} ARS por noche")

    # Tabla de valores reales vs predichos (primeras 10 filas de test)
    comparacion = pd.DataFrame({
        "huéspedes": X_test["accommodates"].values[:10].astype(int),
        "precio_real": y_test.values[:10].round(0),
        "precio_predicho": pred_test[:10].round(0),
    })
    comparacion["error"] = comparacion["precio_real"] - comparacion["precio_predicho"]
    print("\nReales vs predichos (10 primeras filas de test):")
    print(comparacion.to_string(index=False))

    # 9) Gráfico: dispersión + recta ajustada
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # Un pequeño "jitter" horizontal para que los puntos no se pisen
    # (accommodates es entero y se apilarían en columnas).
    jitter = np.random.default_rng(0).uniform(-0.25, 0.25, len(X_test))
    ax.scatter(X_test["accommodates"] + jitter, y_test, s=6, alpha=0.15,
               color="#4C72B0", label="Anuncios de prueba")
    medianas = datos.groupby("accommodates")["price"].median()
    ax.plot(medianas.index, medianas.values, "o", color="#55A868",
            label="Mediana por capacidad")
    xs = np.linspace(datos["accommodates"].min(), datos["accommodates"].max(), 100)
    ax.plot(xs, modelo.predict(pd.DataFrame({"accommodates": xs})),
            color="#C44E52", lw=2.5, label="Recta ajustada")
    ax.scatter([CASO_NUEVO], [estimacion], color="black", s=80, zorder=5,
               marker="X", label=f"Caso nuevo: {CASO_NUEVO} huéspedes")
    ax.set_xlabel("Capacidad (huéspedes)")
    ax.set_ylabel("Precio por noche (ARS)")
    ax.set_title("Proyecto 1 — Precio nocturno vs capacidad")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CARPETA_FIG / "p1_precio_vs_huespedes.png", dpi=130)
    plt.close(fig)
    print("\nGráfico guardado en graficos/p1_precio_vs_huespedes.png")

    return {"tabla": tabla, "pendiente": pendiente, "intercepto": intercepto,
            "estimacion": estimacion, "comparacion": comparacion}


if __name__ == "__main__":
    correr_p1()
