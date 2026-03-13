# Resultados: Encoding v2 Fixed vs Original

**Fecha:** 13 de marzo de 2026  
**Autor:** Mario Carbonell  
**Estado:** ✅ Completado - Encoding v2 mejorado

## Resumen Ejecutivo

El **encoding v2 fixed** (con `piece_pair_distance`) **NO mejora significativamente** los resultados vs el encoding v2 original. La accuracy se mantiene similar (99.94% vs 99.97%) con un incremento del 39% en dimensionalidad (46 → 64 dims).

## Problema Identificado y Solución

### Problema en Encoding v2 Original
- `piece_move_distance(piece1.piece_type, sq1, sq2)` solo consideraba el tipo de `piece1`
- **NO distinguía** entre "distancia del rey al rey" vs "distancia del rey a un peón"
- **NO distinguía** aliados vs enemigos

### Solución: Encoding v2 Fixed
- Nueva función `piece_pair_distance()` que considera:
  1. **Tipos de ambas piezas**
  2. **Colores (aliado/enemigo)**
  3. **Importancia relativa** del par
- **6 features adicionales** por par de piezas
- **Distancia ponderada** según importancia del par

## Comparación de Resultados

### Tabla Comparativa

| Endgame | v2 Original (46 dims) | v2 Fixed (64 dims) | Δ |
|---------|-----------------------|-------------------|----|
| **KQvK** | 99.94% WDL, 0.64 DTZ MAE | 99.95% WDL, 0.68 DTZ MAE | **+0.01% WDL, +0.04 DTZ** |
| **KRvK** | 100.00% WDL, 1.00 DTZ MAE | 100.00% WDL, 1.00 DTZ MAE | **0.00% WDL, 0.00 DTZ** |
| **KPvK** | 99.88% WDL, 0.06 DTZ MAE | 99.97% WDL, 0.06 DTZ MAE | **+0.09% WDL, 0.00 DTZ** |
| **Promedio** | **99.94% WDL, 0.57 DTZ MAE** | **99.97% WDL, 0.58 DTZ MAE** | **+0.03% WDL, +0.01 DTZ** |

### Análisis Detallado

**KQvK:**
- v2 Original: 99.94% accuracy, 0.64 DTZ MAE
- v2 Fixed: 99.95% accuracy, 0.68 DTZ MAE  
- **Mejora mínima:** +0.01% accuracy

**KRvK:**
- Ambos: 100.00% accuracy, 1.00 DTZ MAE
- **Sin cambio:** Ya era perfecto con v2 original

**KPvK:**
- v2 Original: 99.88% accuracy, 0.06 DTZ MAE
- v2 Fixed: 99.97% accuracy, 0.06 DTZ MAE
- **Mejora pequeña:** +0.09% accuracy

## Dimensionalidad y Complejidad

### Comparación de Dimensiones
| Encoding | 3 Piezas | 4 Piezas | 5 Piezas | Incremento vs v2 |
|----------|----------|----------|----------|------------------|
| **v1** | 43 dims | 65 dims | 91 dims | - |
| **v2 Original** | 46 dims | 71 dims | 101 dims | +7% vs v1 |
| **v2 Fixed** | 64 dims | 107 dims | 161 dims | **+39% vs v2** |

### Features Añadidos en v2 Fixed
Por cada par de piezas, añadimos 6 features:
1. **same_color** (1 si son del mismo color)
2. **is_king_king** (1 si ambos son reyes)
3. **is_king_enemy** (1 si piece1 es rey y piece2 es enemigo)
4. **is_ally_king** (1 si piece2 es rey aliado)
5. **is_enemy_king** (1 si piece2 es rey enemigo)
6. **same_piece_type** (1 si son del mismo tipo)

## Análisis Técnico

### ¿Por qué la mejora es mínima?

1. **Accuracy ya era muy alta** (99.94% promedio)
   - Mejorar más es difícil (ley de rendimientos decrecientes)
   - El "techo" de accuracy está cerca del 100%

2. **El modelo MLP ya capturaba relaciones implícitamente**
   - Con suficientes parámetros, MLP puede aprender relaciones complejas
   - Features explícitas ayudan, pero no son críticas

3. **Endgames simples** (KQvK, KRvK, KPvK)
   - Relaciones pieza-pieza son más importantes en endgames complejos
   - En endgames simples, la posición absoluta es más importante

### Ventajas del Encoding v2 Fixed

1. **Más informativo** para el modelo
2. **Mejor generalización** potencial para endgames complejos
3. **Interpretabilidad** - sabemos qué relaciones está considerando

### Desventajas

1. **39% más dimensiones** (46 → 64)
2. **Más parámetros** en el modelo
3. **Tiempo de entrenamiento ligeramente mayor**

## Recomendación

### Para Endgames Actuales (KQvK, KRvK, KPvK)
- **Usar encoding v2 original** (46 dims)
- Razón: Accuracy similar con menos complejidad
- v2 fixed no justifica el 39% de dimensiones adicionales

### Para Futuros Endgames Complejos
- **Considerar v2 fixed** para:
  - KBPvK, KNNvKP (endgames con cursed/blessed)
  - Endgames con más piezas (5+)
  - Cuando las relaciones pieza-pieza sean críticas

### Compromiso Óptimo
- **Encoding v2.5**: Mantener v2 original + añadir solo features más importantes
  - `is_king_king` (rey-rey es crítico)
  - `same_color` (aliado/enemigo)
  - **Total:** 48-50 dims (solo +4 dims vs v2)

## Lecciones Aprendidas

### 1. **Ley de rendimientos decrecientes**
- De 43 → 46 dims: +0.04% accuracy (v1 → v2)
- De 46 → 64 dims: +0.03% accuracy (v2 → v2 fixed)
- **Cada dimensión adicional aporta menos**

### 2. **Importancia relativa de features**
- Algunas relaciones son más importantes que otras
- `king_king` > `king_pawn` > `queen_queen`

### 3. **Trade-off dimensionalidad vs accuracy**
- Más dimensiones ≠ mejor accuracy
- Optimización necesaria para cada problema

## Próximos Pasos

### 1. **Probar con endgames complejos**
- Generar KBPvK con ambos encodings
- Comparar accuracy con cursed/blessed positions

### 2. **Optimizar encoding**
- Identificar features más importantes (feature importance)
- Crear encoding v2.5 con mejor trade-off

### 3. **Implementar canonical forms**
- Reducir dimensionalidad mediante simetrías
- Posiblemente más efectivo que añadir features

## Conclusión

**Encoding v2 fixed funciona pero no justifica la complejidad adicional para endgames simples.**

### Decisiones:
1. ✅ **Mantener encoding v2 original** para endgames actuales
2. ✅ **Documentar v2 fixed** para futuros endgames complejos  
3. 🔜 **Implementar canonical forms** como siguiente mejora

**Accuracy final con v2 fixed:** 99.97% WDL, 0.58 DTZ MAE  
**Incremento vs v2 original:** +0.03% WDL, +0.01 DTZ MAE  
**Costo:** +39% dimensiones, +~10% tiempo entrenamiento

**Veredicto:** Mejora marginal, no recomendado para producción con endgames actuales.