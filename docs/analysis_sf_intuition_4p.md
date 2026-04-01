# Análisis de Stockfish Intuition (EVAL Depth 0) - Finales de 4 Piezas

Este documento detalla el rendimiento de Stockfish 18 (EVAL) comparado contra las tablas Syzygy en posiciones aleatorias de 4 piezas. Este benchmark sirve como línea base para evaluar el progreso de la arquitectura **Vanguard V10.1**.

**Fecha**: 2026-04-01
**Configuración**: `threshold=150 centipawns`, `go nodes 1` como fallback para jaques.
**Perspectiva**: Normalizada a Blancas (White-side evaluation).
**Muestra**: 10,000 posiciones aleatorias legales por cada final.

---

## 1. Resumen de Precisión (Mapeo Final)

| Final | Precisión (WDL) | Error de Juicio (Wins as Draws) | Comentario |
| :--- | :--- | :--- | :--- |
| **KBBvK** | **58.48%** | 4,088 | **Ceguera Total**: Cree que es tablas casi siempre. |
| **KBNvK** | **71.54%** | 2,327 | Falla en el mate técnico de Alfil+Caballo. |
| **KRvKN** | **82.39%** | 1,671 | No ve victorias técnicas sin búsqueda. |
| **KPvKP** | **86.04%** | 575 | Errores en carreras de peones sutiles. |
| **KBvKB** | 100.00% | 0 | Tablas por material perfectamente identificadas. |
| **KQQvK** | 98.49% | 0 | Dama de ventaja es trivial para el PSQT. |
| **KQvKQ** | **74.66%** | 1,267 | Inestabilidad en finales de damas simétricos. |

---

## 2. Puntos Ciegos Tácticos (Blind Spots)

### 2.1. El Callejón sin Salida de los Dos Alfiles (`KBBvK`)
A pesar de tener ventaja ganadora absoluta, Stockfish 18 (NNUE) evalúa el **40% de las posiciones ganadoras como tablas (0.00)**. 
*   **Significado**: NNUE prioriza material y proximidad de peones en su evaluación estática. No tiene una "heurística" de coordinación de alfiles que puntúe positivamente el progreso táctico sin cálculo.

### 2.2. La Complejidad de `KRvKN` (Torre vs Caballo)
Se observó una tendencia masiva a evaluar posiciones de victoria técnica como tablas (1,671 casos). Sin búsqueda, la red no otorga suficiente valor a la restricción del caballo por la torre.

---

## 3. Metodología de Resolución: "Paradoja de los Jaques"

Se resolvió la asimetría detectada inicialmente implementando un **fallback automático**:
1. Si `eval` retorna `none (in check)`, el script lanza `go nodes 1`.
2. Se parsea el `score cp` o `score mate` de la línea UCI.
3. Se normaliza la perspectiva según el turno (side-to-move).

Esto elevó la precisión en finales defensivos (como `KvKQ`) del 74% a >99%, permitiendo un mapa realista.

---

## 4. Conclusión Estratégica para Vanguard
Vanguard (especialmente en su versión V10.1 con pesos relacionales geométricos) debe enfocarse en **superar el 90% en KBBvK y KRvKN**. Estos son los terrenos donde una representación basada en grafos espaciales puede capturar la "armonía" de las piezas que Stockfish solo logra ver mediante fuerza bruta.

---
