"""
correr_tp.py — Corre el Proyecto 1 y el Proyecto 4 del TP de Airbnb, uno atrás del otro.

Uso (parado en la carpeta TP_airbnb):
    python correr_tp.py
"""

from p1_regresion_huespedes import correr_p1
from p4_knn_alojamientos_parecidos import correr_p4

if __name__ == "__main__":
    correr_p1()
    print("\n")
    correr_p4()
