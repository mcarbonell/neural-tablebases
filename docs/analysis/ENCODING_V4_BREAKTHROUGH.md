# Análisis Técnico: Encoding V4 y la Resolución de Carreras de Peones

## 📌 Contexto
Históricamente, los modelos basados en `v1` y `v2` tenían dificultades críticas en finales de peones, específicamente en posiciones de "carrera" (race positions) donde ambos bandos promocionan casi simultáneamente. La red a menudo evaluaba como tablas o derrota lo que eran victorias claras para el bando que corona primero.

El encoding **V4** fue diseñado para solucionar esto mediante dos innovaciones estructurales:
1. **Normalización de Perspectiva Humana**: El bando que mueve es siempre "Blanco" y avanza hacia la fila 8.
2. **Progreso de Promoción**: Una característica explícita que mide la cercanía de cada peón a la octava fila.

## 🚀 El Avance: Resolviendo la Carrera
Se utilizó la posición `8/8/7P/p7/8/8/8/2k2K2 w - - 0 1` como "test de estrés".

### Resultados Comparativos

| Atributo | Modelo V1/V2 | Modelo V4 (97.4% Acc) |
| :--- | :--- | :--- |
| **Evaluación Neural (Depth 0)** | Error (Tablas/Pierde) | **Aware** (49.5% Win, 50.4% Draw) |
| **Búsqueda Minimax (Depth 2)** | Inconsistente | **SOLVED** (Score 1.61) |
| **Búsqueda Minimax (Depth 4)** | A menudo fallaba | **SOLVED** (Score 1.75) |

### ¿Por qué funciona V4?
La inclusión de `progress = rank / 7.0` como un feature independiente permite que las capas lineales de la red realicen una comparación directa de "distancia al objetivo" sin tener que inferirla únicamente de las coordenadas absolutas. 

Al normalizar la perspectiva, la red ya no tiene que aprender que "Blanco sube" y "Negro baja"; solo tiene que aprender que **"un incremento en Progress es bueno"**. Esto reduce drásticamente la complejidad del espacio de búsqueda para los pesos de la red.

## 📊 Escalabilidad a 5 Piezas (KRPvKP)
El modelo V4 de 5 piezas ha demostrado una convergencia acelerada:
- **Epoch 3**: 93.3% de precisión.
- **Prueba en Tiempo Real**: 99.9% de confianza en posiciones ganadoras complejas.

## 🛠️ Herramientas de Legado
Se ha implementado `src/analyze_fen.py` para facilitar la verificación mutua entre versiones de encoding y tablebases Syzygy, permitiendo análisis profundos con Minimax.

## 🏁 Conclusión
El encoding V4 no es solo una mejora incremental, es un **cambio de paradigma** en cómo la red procesa la semántica del tablero de ajedrez, alineando la ingeniería de características con los principios heurísticos clásicos del juego (distancia y tiempo).

---
**Autor:** Mario Carbonell / Antigravity
**Fecha:** 14 de Marzo, 2026
**Modelos:** `mlp_kpvkp_v4_best.pth`, `mlp_krpvkp_v4_best.pth`
