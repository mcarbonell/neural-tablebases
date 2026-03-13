# Resultados Finales: Encoding v1 vs v2

**Fecha:** 13 de marzo de 2026  
**Autor:** Mario Carbonell

## Resumen Ejecutivo

Hemos completado la validación del encoding v2 (con distancias de movimiento) en los 3 finales de 3 piezas. Los resultados muestran una mejora consistente en accuracy, alcanzando **100% en 2 de 3 endgames**.

## Tabla Comparativa Completa

### Accuracy

| Endgame | Posiciones v1 | Posiciones v2 | v1 Accuracy | v2 Accuracy | Mejora |
|---------|---------------|---------------|-------------|-------------|--------|
| KQvK | 368,452 | 64,631 | 99.92% | **100.00%** | +0.08% |
| KRvK | 399,112 | 70,672 | 99.99% | **100.00%** | +0.01% |
| KPvK | 331,352 | 74,984 | 99.89% | **99.92%** | +0.03% |
| **Promedio** | **366,305** | **70,096** | **99.93%** | **99.97%** | **+0.04%** |

### Entrenamiento

| Endgame | v1 Épocas | v2 Épocas | v1 Tiempo | v2 Tiempo | v1 Val Loss | v2 Val Loss |
|---------|-----------|-----------|-----------|-----------|-------------|-------------|
| KQvK | 27 | 40 | ~3 min | ~4 min | ? | 0.0005 |
| KRvK | 13 | 50 | ~3 min | ~4.5 min | ? | 0.0005 |
| KPvK | 29 | 50 | ~3 min | ~4.5 min | ? | 0.0035 |
| **Promedio** | **23** | **47** | **~3 min** | **~4.3 min** | **-** | **0.0015** |

### Generación de Datasets

| Endgame | v1 Tiempo | v2 Tiempo (Paralelo) | Speedup |
|---------|-----------|----------------------|---------|
| KQvK | ~3 min | 13 seg | 13.8x |
| KRvK | ~3 min | 23 seg | 7.8x |
| KPvK | ~3 min | 11 seg | 16.4x |
| **Promedio** | **~3 min** | **~16 seg** | **~12x** |

## Análisis Detallado

### Mejoras de v2

1. **Accuracy Superior**
   - Promedio: 99.97% vs 99.93% (+0.04%)
   - 100% en 2/3 endgames (KQvK, KRvK)
   - Primera vez alcanzando perfección

2. **Información Más Rica**
   - Captura movilidad real de cada pieza
   - Diferencia entre torre (1-2 movs) y dama (1 mov)
   - Útil para restricciones de peones

3. **Generación Ultra-Rápida**
   - Generador paralelo: 13-23 segundos
   - 12x más rápido que v1 single-threaded
   - Permite iteración rápida

### Costos de v2

1. **Más Épocas de Entrenamiento**
   - 47 vs 23 épocas promedio (2x más)
   - ~4.3 min vs ~3 min por endgame (+43%)
   - Convergencia más lenta

2. **Más Dimensiones**
   - 46 vs 43 dims para 3 piezas (+7%)
   - 71 vs 65 dims para 4 piezas (+9%)
   - Más parámetros en el modelo

3. **Complejidad de Implementación**
   - Cálculo de distancias de movimiento por pieza
   - Manejo de casos especiales (alfil, peón)
   - Más código para mantener

## Encoding v2: Detalles Técnicos

### Dimensiones Adicionales

Para cada par de piezas (i, j), v2 añade:
- **Distancia de movimiento:** Cuántos movimientos necesita la pieza i para llegar a la casilla de j

### Cálculo por Tipo de Pieza

- **Dama (Q):** 1 movimiento (líneas rectas + diagonales)
- **Torre (R):** 1-2 movimientos (solo líneas rectas)
- **Alfil (B):** 1 movimiento (mismo color), ∞ (color diferente)
- **Caballo (N):** BFS desde casilla origen
- **Peón (P):** Solo hacia adelante, captura diagonal
- **Rey (K):** Distancia Chebyshev

### Fórmula de Dimensiones

```
v1: num_pieces × 10 + num_pairs × 4 + 1
v2: num_pieces × 10 + num_pairs × 5 + 1

Para 3 piezas:
  v1: 3×10 + 3×4 + 1 = 43
  v2: 3×10 + 3×5 + 1 = 46

Para 4 piezas:
  v1: 4×10 + 6×4 + 1 = 65
  v2: 4×10 + 6×5 + 1 = 71
```

## Recomendaciones

### Cuándo Usar v2

✅ **Usar v2 para:**
- Finales con peones (KPvK, KPvKP, KBPvK)
- Finales asimétricos (KQvKR, KRvKN)
- Cuando se busca 100% accuracy
- Finales complejos (5+ piezas)
- Producción final

### Cuándo Usar v1

✅ **Usar v1 para:**
- Validación rápida (converge en ~20 épocas)
- Finales simples (KQvK, KRvK, KRRvK)
- Cuando 99.9% accuracy es suficiente
- Experimentación rápida
- Datasets ya generados

## Impacto en el Proyecto

### Logros Alcanzados

1. ✅ **100% accuracy en 2/3 endgames** - Primera vez
2. ✅ **Generador paralelo** - 6-7x speedup
3. ✅ **Encoding v2 validado** - Mejora consistente
4. ✅ **Infraestructura escalable** - Lista para 4+ piezas

### Próximos Pasos

1. **Entrenar KRRvK** con dataset completo (21.89M posiciones)
2. **Comparar MLP vs SIREN** en 4-piece
3. **Generar más 4-piece** con generador paralelo:
   - KRvKP (asimétrico)
   - KPvKP (complejo)
   - KBPvK (fortress)
4. **Implementar canonical forms** (propuesta de Gemini)
5. **Probar QAT** (Quantization-Aware Training)

## Conclusión

El encoding v2 ha demostrado su valor al alcanzar **99.97% accuracy promedio** y **100% en 2/3 endgames**. Aunque requiere 2x más épocas de entrenamiento, la mejora en accuracy justifica el costo, especialmente para finales complejos donde cada décima de porcentaje importa.

**Decisión:** Usar v2 como estándar para futuros experimentos, con v1 como opción para validación rápida.

---

**Métricas Clave:**
- Accuracy promedio v2: **99.97%**
- 100% accuracy: **2/3 endgames**
- Generación: **13-23 segundos** (paralelo)
- Entrenamiento: **~4.3 minutos** promedio
- Speedup generación: **~12x** vs v1

**Estado:** ✅ Validación completa - Listo para 4-piece
