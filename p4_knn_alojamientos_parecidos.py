"""
p4_knn_alojamientos_parecidos.py
Proyecto 4 — Precio según alojamientos similares.

Caso: valorar un anuncio nuevo buscando alojamientos PARECIDOS
(capacidad, dormitorios, camas y noches mínimas) y promediando el precio
de los vecinos más cercanos.

Modelo elegido: k-NN PARA REGRESIÓN (KNeighborsRegressor).
  - "Buscar parecidos y promediar sus precios" es literalmente lo que hace k-NN.
  - El precio es numérico -> Regressor, no Classifier
    (Classifier serviría si el objetivo fuera "barato / medio / caro").
  - k-NN mide DISTANCIAS, así que hay que estandarizar: si no,
    minimum_nights (que puede valer 30, 90...) pesaría muchísimo más que
    bedrooms (1, 2, 3) solo por sus unidades.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from funciones_airbnb import (RANDOM_STATE, TEST_SIZE, cargar_datos, limpiar, metricas,
                   tabla_metricas, verificar_columnas)

CARPETA_FIG = Path(__file__).parent / "graficos"
CARPETA_FIG.mkdir(exist_ok=True)

X_COLS = ["accommodates", "bedrooms", "beds", "minimum_nights"]
CASO_NUEVO = {"accommodates": 4, "bedrooms": 2, "beds": 3, "minimum_nights": 2}
K = 5  # lo fija la consigna


def correr_p4():
    print("=" * 60)
    print("PROYECTO 4 — k-NN para regresión (alojamientos similares)")
    print("=" * 60)

    # 1) Carga y verificación
    df = cargar_datos()
    verificar_columnas(df, X_COLS + ["price"])

    # 2-3) Limpieza. Registros imposibles:
    #   - 0 huéspedes, 0 camas o 0 noches mínimas no tienen sentido
    #   - bedrooms puede ser 0 (monoambiente), así que se acepta
    #   - además se descartan estadías mínimas absurdas (> 365 noches)
    datos, _ = limpiar(df, X_COLS, reglas_minimas={
        "accommodates": 1, "bedrooms": 0, "beds": 1, "minimum_nights": 1})
    antes = len(datos)
    datos = datos[datos["minimum_nights"] <= 365]
    print(f"Estadías mínimas > 365 noches descartadas: {antes - len(datos)}")

    # 4) División 75/25 con semilla fija
    X = datos[X_COLS]
    y = datos["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # 5) Pipeline: escalador + modelo.
    #    Al hacer fit del Pipeline, el StandardScaler aprende media y desvío
    #    SOLO con train; en test solo aplica transform -> no hay fuga de información.
    modelo = Pipeline([
        ("escalador", StandardScaler()),
        ("knn", KNeighborsRegressor(n_neighbors=K)),
    ])
    modelo.fit(X_train, y_train)

    # Controles para comparar:
    #  - el mismo k-NN pero SIN escalar
    #  - el modelo base (siempre el promedio)
    sin_escalar = KNeighborsRegressor(n_neighbors=K).fit(X_train, y_train)
    #  - la variante que pondera por distancia (los vecinos más cercanos pesan más)
    ponderado = Pipeline([
        ("escalador", StandardScaler()),
        ("knn", KNeighborsRegressor(n_neighbors=K, weights="distance")),
    ]).fit(X_train, y_train)
    base = DummyRegressor(strategy="mean").fit(X_train, y_train)

    # 6-7) Evaluación
    pred_test = modelo.predict(X_test)
    resultados = {
        f"k-NN escalado, k={K} (test)": metricas(y_test, pred_test),
        f"k-NN escalado, k={K} (train)": metricas(y_train, modelo.predict(X_train)),
        f"k-NN escalado, k={K}, weights=distance (test)": metricas(y_test, ponderado.predict(X_test)),
        f"k-NN sin escalar, k={K} (test)": metricas(y_test, sin_escalar.predict(X_test)),
        "Modelo base - promedio (test)": metricas(y_test, base.predict(X_test)),
    }
    tabla = tabla_metricas(resultados)
    print("\nMétricas:")
    print(tabla.to_string())

    # 8) Caso nuevo [4 huéspedes, 2 dormitorios, 3 camas, 2 noches]
    nuevo = pd.DataFrame([CASO_NUEVO])
    estimacion = modelo.predict(nuevo)[0]
    print("\nEstimación para [4 huéspedes, 2 dormitorios, 3 camas, 2 noches mínimas]: "
          f"{estimacion:,.0f} ARS por noche")

    # Mostramos los 5 vecinos que usó el modelo para ese caso.
    # kneighbors() trabaja en el espacio ESCALADO, así que primero transformamos.
    escalador = modelo.named_steps["escalador"]
    knn = modelo.named_steps["knn"]
    distancias, indices = knn.kneighbors(escalador.transform(nuevo))
    vecinos = X_train.iloc[indices[0]].copy()
    vecinos["price"] = y_train.iloc[indices[0]].values
    vecinos["distancia_escalada"] = distancias[0].round(3)
    print("\nVecinos usados para el caso nuevo:")
    print(vecinos.to_string(index=False))

    # Cuántos anuncios de train son "gemelos exactos" del caso nuevo
    gemelos = (X_train == pd.Series(CASO_NUEVO)).all(axis=1)
    print(f"\nAnuncios de train idénticos al caso nuevo: {gemelos.sum()} "
          f"(precio promedio {y_train[gemelos].mean():,.0f} ARS)")

    # ¿Por qué escalar cambia tan poco? Porque casi todos los anuncios de test
    # tienen un "gemelo" en train con exactamente los mismos 4 valores:
    # la distancia a esos vecinos es 0 con o sin escalar.
    combos_train = set(map(tuple, X_train.values))
    con_gemelo = sum(tuple(fila) in combos_train for fila in X_test.values)
    print(f"Anuncios de test con un gemelo exacto en train: {con_gemelo:,} de "
          f"{len(X_test):,} ({con_gemelo / len(X_test):.0%})")

    # Tabla reales vs predichos
    comparacion = X_test.iloc[:10].astype(int).copy()
    comparacion["precio_real"] = y_test.values[:10].round(0)
    comparacion["precio_predicho"] = pred_test[:10].round(0)
    comparacion["error"] = comparacion["precio_real"] - comparacion["precio_predicho"]
    print("\nReales vs predichos (10 primeras filas de test):")
    print(comparacion.to_string(index=False))

    # Extra: ¿cómo cambia el error con otros k? Validación cruzada SOLO en train.
    # (El modelo final se queda con k=5 porque lo pide la consigna.)
    print("\nValidación cruzada (5 folds, solo train):")
    ks = [1, 3, 5, 7, 9, 11, 15, 25, 50]
    rmse_cv = []
    for k in ks:
        pipe = Pipeline([("escalador", StandardScaler()),
                         ("knn", KNeighborsRegressor(n_neighbors=k))])
        puntajes = cross_val_score(pipe, X_train, y_train, cv=5,
                                   scoring="neg_root_mean_squared_error")
        rmse_cv.append(-puntajes.mean())
        print(f"  k={k:>2}: RMSE medio = {rmse_cv[-1]:,.0f}")

    # 9) Gráficos
    # a) Real vs predicho
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, pred_test, s=6, alpha=0.2, color="#4C72B0")
    lim = [0, max(y_test.max(), pred_test.max())]
    ax.plot(lim, lim, color="#C44E52", lw=2, label="Predicción perfecta")
    ax.set_xlabel("Precio real (ARS)")
    ax.set_ylabel("Precio predicho (ARS)")
    ax.set_title(f"Proyecto 4 — Real vs predicho (k-NN, k={K})")
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CARPETA_FIG / "p4_reales_vs_predichos.png", dpi=130)
    plt.close(fig)

    # b) RMSE de validación cruzada según k
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ks, rmse_cv, "o-", color="#4C72B0")
    ax.axvline(K, color="#C44E52", ls="--", label=f"k={K} (consigna)")
    ax.set_xlabel("k (cantidad de vecinos)")
    ax.set_ylabel("RMSE medio en validación cruzada (ARS)")
    ax.set_title("Proyecto 4 — Error según k (solo train)")
    ax.yaxis.set_major_formatter(fmt)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CARPETA_FIG / "p4_error_segun_k.png", dpi=130)
    plt.close(fig)
    print("\nGráficos guardados en graficos/ (p4_reales_vs_predichos.png y p4_error_segun_k.png)")

    return {"tabla": tabla, "estimacion": estimacion, "vecinos": vecinos,
            "gemelos": int(gemelos.sum()), "comparacion": comparacion,
            "cv": dict(zip(ks, rmse_cv))}


if __name__ == "__main__":
    correr_p4()
