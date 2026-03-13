# Comparación: Encoding v1 vs v2

**Autor:** Mario Carbonell  
**Fecha:** 13 de marzo de 2026  
**Estado:** En progreso

## Objetivo

Comparar el rendimiento del encoding geométrico v1 (distancias Manhattan/Chebyshev) vs v2 (con distancias de movimiento específicas de cada pieza) en finales de 3 piezas.

## Diferencias entre v1 y v2

### Encoding v1 (43 dims para 3 piezas)
- Coordenadas relativas (dx, dy) entre piezas: 2 × num_pairs
- Distancia Manhattan: 1 × num_pairs
- Distancia Chebyshev: 1 × num_pairs
- Coordenadas absolutas normalizadas: 2 × num_pieces
- Distancias al centro: 1 × num_pieces
- Distancias a bordes (4 direcciones): 4 × num_pieces
- Distancias a esquinas (4 esquinas): 4 × num_pieces
- Side-to-move: 1

**Total:** 3×10 + 3×4 + 1 = 43 dimensiones

### Encoding v2 (46 dims para 3 piezas)
Todo lo de v1, más:
- **Distancia de movimiento real** entre cada par de piezas: 1 × num_pairs

La distancia de movimiento captura cuántos movimientos necesita una pieza para llegar a otra casilla:
- **Dama:** 1 movimiento (líneas rectas y diagonales)
- **Torre:** 1-2 movimientos (solo líneas rectas)
- **Alfil:** 1 movimiento (diagonales del mismo color), ∞ (color diferente)
- **Caballo:** Distancia única calculada por BFS
- **Peón:** Solo hacia adelante, captura en diagonal

**Total:** 43 + 3 = 46 dimensiones

## Resultados

### KQvK (King + Queen vs King)

| Métrica | v1 (43 dims) | v2 (46 dims) | Mejora |
|---------|--------------|--------------|--------|
| Posiciones | 368,452 | 64,631 | -82.5% (regenerado) |
| Accuracy | 99.92% | **100.00%** | +0.08% |
| Épocas | 27 | 40 | +48% |
| Mejor época | ? | 40 | - |
| Val Loss | ? | 0.0005 | - |

**Observaciones:**
- ✅ Alcanza 100% accuracy (perfecto)
- ⚠️ Menos posiciones (dataset regenerado con generador paralelo)
- ⚠️ Más épocas necesarias (40 vs 27)

### KRvK (King + Rook vs King)

| Métrica | v1 (43 dims) | v2 (46 dims) | Mejora |
|---------|--------------|--------------|--------|
| Posiciones | 399,112 | 70,672 | -82.3% (regenerado) |
| Accuracy | 99.99% | **100.00%** | +0.01% |
| Épocas | 13 | 50 | +285% |
| Val Loss | ? | 0.0005 | - |
| Tiempo total | ~3 min | ~4.5 min | +50% |

**Observaciones:**
- ✅ Alcanza 100% accuracy (perfecto)
- ⚠️ Necesita muchas más épocas (50 vs 13)
- ⚠️ Menos posiciones (dataset regenerado)

### KPvK (King + Pawn vs King)

| Métrica | v1 (43 dims) | v2 (46 dims) | Mejora |
|---------|--------------|--------------|--------|
| Posiciones | 331,352 | 74,984 | -77.4% (regenerado) |
| Accuracy | 99.89% | **99.92%** | +0.03% |
| Épocas | 29 | 50 | +72% |
| Val Loss | ? | 0.0035 | - |
| Tiempo total | ~3 min | ~4.5 min | +50% |

**Observaciones:**
- ✅ Mejora ligera en accuracy (99.92% vs 99.89%)
- ⚠️ Más épocas necesarias (50 vs 29)
- ⚠️ Menos posiciones (dataset regenerado)

## Resultados Finales - Resumen

| Endgame | v1 Accuracy | v2 Accuracy | Mejora | v1 Épocas | v2 Épocas |
|---------|-------------|-------------|--------|-----------|-----------|
| KQvK | 99.92% | **100.00%** | +0.08% | 27 | 40 |
| KRvK | 99.99% | **100.00%** | +0.01% | 13 | 50 |
| KPvK | 99.89% | **99.92%** | +0.03% | 29 | 50 |
| **Promedio** | **99.93%** | **99.97%** | **+0.04%** | **23** | **47** |

## Análisis Preliminar

### Ventajas de v2
1. **Información más rica:** Captura la movilidad real de cada pieza
2. **Mejor para piezas asimétricas:** Torre (1-2 movs) vs Dama (1 mov)
3. **Útil para peones:** Captura restricciones de movimiento
4. **100% accuracy en KQvK:** Demuestra que la información adicional es valiosa

### Desventajas de v2
1. **Más dimensiones:** 46 vs 43 (7% más parámetros)
2. **Más épocas:** Parece necesitar más entrenamiento
3. **Complejidad:** Más difícil de implementar y debuggear

### Hipótesis

La distancia de movimiento es especialmente útil para:
- **Finales con peones:** Captura restricciones de movimiento hacia adelante
- **Finales asimétricos:** Diferencia entre piezas de diferente movilidad
- **Finales complejos:** Donde la movilidad es crítica (ej: KBNvK)

## Próximos Pasos

1. ✅ Completar entrenamiento de KRvK y KPvK con v2
2. 🔜 Comparar resultados finales
3. 🔜 Decidir qué encoding usar para 4-piece
4. 🔜 Probar v2 en KRRvK (ya generado con v1)

## Conclusión Final

El encoding v2 ha demostrado mejoras consistentes en accuracy:
- **2 de 3 endgames alcanzan 100% accuracy** (KQvK, KRvK)
- **Promedio: 99.97%** vs 99.93% en v1 (+0.04%)
- **Trade-off:** Requiere ~2x más épocas de entrenamiento

### Ventajas Confirmadas
1. ✅ **Mejor accuracy:** 99.97% vs 99.93% promedio
2. ✅ **100% en 2/3 endgames:** KQvK y KRvK perfectos
3. ✅ **Información más rica:** Captura movilidad real
4. ✅ **Mejora en KPvK:** El peón se beneficia de distancias de movimiento

### Desventajas Confirmadas
1. ❌ **Más épocas:** 47 vs 23 promedio (2x más tiempo)
2. ❌ **Convergencia más lenta:** Necesita más entrenamiento
3. ❌ **Complejidad:** 3 dimensiones adicionales

### Recomendación Final

**Usar encoding v2 para:**
- Finales con peones (restricciones de movimiento)
- Finales donde se busca 100% accuracy
- Finales complejos o asimétricos

**Usar encoding v1 para:**
- Validación rápida (converge en ~20 épocas)
- Finales simples (piezas mayores simétricas)
- Cuando 99.9% accuracy es suficiente

**Para el proyecto:** Continuar con v1 en KRRvK (ya generado) y considerar v2 para futuros endgames complejos.

---

**Actualización:** Resultados completos - 13 de marzo de 2026, 09:15 AM
