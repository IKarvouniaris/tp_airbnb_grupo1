# TP Airbnb — Grupo 1

Estimación del precio por noche de alojamientos de Buenos Aires con datos de [Inside Airbnb](https://insideairbnb.com/get-the-data/).
Python para Ciencia de Datos — UADE.

- **Proyecto 1:** regresión lineal simple, precio según capacidad (`accommodates`).
- **Proyecto 4:** k-NN para regresión (k=5, variables escaladas), precio según alojamientos parecidos (`accommodates`, `bedrooms`, `beds`, `minimum_nights`).

## Cómo correrlo

1. Instalar dependencias:
   ```
   pip install -r requirements.txt
   ```
2. Descargar `listings.csv.gz` de Buenos Aires (Detailed Listings data, extracción del 23/09/2026) desde Inside Airbnb y dejarlo **sin descomprimir** en la carpeta del proyecto o en `data/`. No se sube al repo porque pesa unos 18 MB.
3. Correr los dos proyectos:
   ```
   python correr_tp.py
   ```
   Tarda unos 10 segundos y guarda los gráficos en `graficos/`.

## Archivos

| Archivo | Qué hace |
|---|---|
| `funciones_airbnb.py` | Carga, limpieza y métricas que comparten los dos proyectos |
| `p1_regresion_huespedes.py` | Proyecto 1 |
| `p4_knn_alojamientos_parecidos.py` | Proyecto 4 |
| `correr_tp.py` | Corre los dos proyectos |
| `Informe_TP_Airbnb_P1_P4.docx` | Informe (introducción, código, resultados y conclusión) |
| `graficos/`, `capturas/`, `salida_consola.txt` | Resultados de la corrida |

## Resultados (conjunto de prueba)

| Modelo | MAE (ARS) | RMSE (ARS) | R² |
|---|---|---|---|
| P1 — Regresión lineal | 48.485 | 79.839 | 0,261 |
| P4 — k-NN escalado, k=5 | 54.157 | 83.877 | 0,128 |
