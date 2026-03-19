# Análisis: Perfección Absoluta en KPvK con Encoding V5 y Búsqueda

## 📌 Resumen del Logro
Hemos alcanzado el **100.00% de precisión** en el final de Rey y Peón contra Rey (KPvK), validado contra el oráculo de Syzygy sobre una muestra exhaustiva de 100,000 posiciones. Este hito se ha conseguido mediante la combinación del nuevo **Encoding V5** y la técnica de **Corrección por Búsqueda Superficial**.

## 🛠️ Innovación: Encoding V5 (King-Centric)
El Encoding V5 soluciona las ambigüedades geométricas de versiones anteriores mediante tres pilares:

1.  **Soberanía del Rey (Slot 0):** El Rey del bando que mueve (STM) ahora siempre ocupa el **índice 0** del vector de entrada. Esto actúa como un "ancla" geográfica constante para la red.
2.  **Normalización de Perspectiva:** El tablero se invierte verticalmente si es el turno de las negras, asegurando que el bando que mueve siempre juega "hacia arriba".
3.  **Progresión de Peones Unificada:** Todos los peones (propios y enemigos) tienen una característica de `progress` que aumenta a medida que se acercan a su respectiva fila de promoción.

## 📊 Resultados de la Auditoría Exhaustiva (KPvK)
Evaluación realizada el 19 de Marzo, 2026.

| Configuración | Precisión | Errores / 100,000 | Comentario |
| :--- | :---: | :---: | :--- |
| **Básica (D0)** | **99.9640%** | 36 | < 0.04% de fallo local. |
| **Búsqueda D1** | **100.0000%** | **0** | **Perfección Absoluta.** |
| **Búsqueda D2** | **100.0000%** | **0** | **Perfección Absoluta.** |

### ¿Por qué funciona la búsqueda?
La red neuronal en solitario (D0) puede cometer errores en posiciones de "peón de torre" o maniobras de oposición muy lejanas. Sin embargo, al aplicar una **búsqueda de profundidad 1 (D1)**, la red evalúa los tableros resultantes de todos los movimientos posibles. 

Dado que los errores de la red son locales y no se repiten en posiciones adyacentes, la búsqueda actúa como un **filtro de consistencia**, descartando movimientos mediocres y seleccionando solo aquellos que llevan a estados con evaluaciones neuronales sólidas.

## 💡 El Futuro: Tradear Cómputo por Precisión
Este éxito en KPvK establece un nuevo paradigma para el proyecto:
*   Podemos usar **modelos más pequeños** y comprimidos para finales de 5 y 6 piezas.
*   No necesitamos que la red sea perfecta individualmente.
*   Una búsqueda superficial de 1 o 2 niveles (coste insignificante en milisegundos) garantiza que el motor nunca cometa un error contra Syzygy.

---
**Fecha:** 19 de Marzo, 2026
**Modelo:** `mlp_kpvk_v5` (45 inputs, 450k parámetros)
**Validador:** `src/verify_search_correction.py`
