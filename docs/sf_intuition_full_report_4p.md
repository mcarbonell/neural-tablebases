# Reporte Final: Intuición de Stockfish 18 (4 Piezas)

Este reporte detalla la precisión de la red NNUE de Stockfish 18 (Depth 0) tras corregir los errores de perspectiva y jaques. Los finales están ordenados de **menor a mayor precisión**.

| Final | Precisión | Muestras | Wins as Draws | Comentario |
| :--- | :--- | :--- | :--- | :--- |
| **KBBvK** | 58.48% | 10000 | 4088 | 🔴 CRÍTICO |
| **KBNvK** | 71.54% | 10000 | 2327 | 🔴 CRÍTICO |
| **KQvKQ** | 74.66% | 10000 | 1267 | 🔴 CRÍTICO |
| **KNvKR** | 82.30% | 10000 | 1481 | 🟡 MEJORABLE |
| **KRvKN** | 82.39% | 10000 | 1671 | 🟡 MEJORABLE |
| **KBvKR** | 85.75% | 10000 | 1421 | 🟡 MEJORABLE |
| **KRvKB** | 85.96% | 10000 | 1396 | 🟡 MEJORABLE |
| **KPvKP** | 86.04% | 10000 | 577 | 🟡 MEJORABLE |
| **KPvKN** | 91.82% | 10000 | 801 | 🟡 MEJORABLE |
| **KNvKP** | 91.88% | 10000 | 794 | 🟡 MEJORABLE |
| **KNPvK** | 92.14% | 10000 | 128 | 🟡 MEJORABLE |
| **KQvKB** | 94.15% | 10000 | 9 | 🟡 MEJORABLE |
| **KPvKB** | 94.42% | 10000 | 547 | 🟡 MEJORABLE |
| **KBvKQ** | 94.66% | 10000 | 753 | 🟡 MEJORABLE |
| **KBPvK** | 94.73% | 10000 | 458 | 🟡 MEJORABLE |
| **KBvKP** | 94.83% | 10000 | 505 | 🟡 MEJORABLE |
| **KQvKN** | 95.74% | 10000 | 12 | 🟢 ROBUSTO |
| **KNvKQ** | 95.88% | 10000 | 0 | 🟢 ROBUSTO |
| **KQvKP** | 96.77% | 10000 | 13 | 🟢 ROBUSTO |
| **KPvKQ** | 97.03% | 10000 | 11 | 🟢 ROBUSTO |
| **KRBvK** | 97.77% | 10000 | 0 | 🟢 ROBUSTO |
| **KPPvK** | 97.77% | 10000 | 45 | 🟢 ROBUSTO |
| **KRNvK** | 97.76% | 10000 | 0 | 🟢 ROBUSTO |
| **KQQvK** | 98.49% | 10000 | 0 | 🟢 ROBUSTO |
| **KRPvK** | 99.23% | 10000 | 14 | 🟢 ROBUSTO |
| **KQRvK** | 99.29% | 10000 | 0 | 🟢 ROBUSTO |
| **KQPvK** | 99.36% | 10000 | 27 | 🟢 ROBUSTO |
| **KQBvK** | 99.41% | 10000 | 0 | 🟢 ROBUSTO |
| **KBvKB** | 100.00% | 10000 | 0 | 🟢 ROBUSTO |
| **KBvKN** | 100.00% | 10000 | 0 | 🟢 ROBUSTO |

---

## 5. Experimento Especial: "Daltonismo Geométrico" en `KBBvK`

Se realizó una prueba dirigida para entender por qué Stockfish 18 NNUE tiene una precisión tan baja (**~58.5%**) en el final de dos alfiles (`KBBvK`).

### Hipótesis:
La red NNUE a profundidad 0 solo cuenta material y no distingue el color de las casillas de los alfiles.

### Resultados de la Prueba (Muestra: 200 posiciones segregadas):
- **Alfiles de DISTINTO Color**: **100.00% de Precisión**. Stockfish sabe que gana siempre.
- **Alfiles del MISMO Color**: **0.00% de Precisión**. Stockfish evalúa (+7.00) y predice Victoria, cuando Syzygy marca Tablas Técnicas forzadas.

### Conclusión Técnica:
Este es un **Punto Ciego Estructural** de Stockfish NNUE. El motor compensa esta ceguera mediante la búsqueda, pero su "intuición" (Depth 0) es incapaz de ver la ineficiencia geométrica de dos alfiles en el mismo color.

**Oportunidad para Vanguard**: Nuestra GNN puede y debe resolver esta relación espacial desde la evaluación, superando la visión estática de Stockfish.

**Conclusión Técnica**: Los finales donde Stockfish flaquea son aquellos que requieren una coordinación geométrica de largo alcance sin intercambio de material inminente. Esta es la oportunidad de oro para la **GNN Relacional**.
