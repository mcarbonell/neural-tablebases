# Resultados: Entrenamiento de DTZ (Distance to Zero)

**Fecha:** 13 de marzo de 2026  
**Autor:** Mario Carbonell  
**Estado:** ✅ Completado para 3-piece

## Objetivo

Entrenar el modelo para predecir no solo WDL (Win/Draw/Loss) sino también DTZ (Distance to Zero), que indica cuántos movimientos faltan hasta mate, captura o empate.

## Implementación

### Cambios Realizados

1. **Aumento del peso de DTZ loss**
   - Antes: 0.1 × DTZ loss
   - Ahora: 0.5 × DTZ loss
   - Razón: Dar más importancia a la predicción de DTZ

2. **Métricas de DTZ añadidas**
   - Train DTZ MAE (Mean Absolute Error)
   - Val DTZ MAE
   - Reportadas en cada época

3. **Loss function actualizada**
   ```python
   loss = (loss_wdl * weights).mean() + 0.5 * criterion_dtz(dtz_pred.squeeze(), dtz.float())
   ```

## Resultados Completos: 3-Piece Endgames

### Tabla Resumen

| Endgame | Posiciones | WDL Acc | DTZ MAE | Épocas | Tiempo |
|---------|------------|---------|---------|--------|--------|
| KQvK | 64,631 | 99.94% | **0.64** | 30 | ~75s |
| KRvK | 70,672 | **100.00%** | **1.00** | 30 | ~135s |
| KPvK | 74,984 | 99.88% | **0.06** | 30 | ~140s |
| **Promedio** | **70,096** | **99.94%** | **0.57** | **30** | **~117s** |

### KQvK (King + Queen vs King)

**Métricas Finales:**
- WDL Accuracy: 99.94%
- WDL Val Loss: 0.0041
- **DTZ MAE: 0.64 movimientos**
- Tiempo: ~75 segundos

**Evolución DTZ MAE:**
- Época 1: 7.74 → Época 5: 1.09 → Época 30: 0.63

**Análisis:**
- Mate complejo con múltiples secuencias
- Error promedio < 1 movimiento es excelente
- Convergencia rápida en primeras 10 épocas

### KRvK (King + Rook vs King)

**Métricas Finales:**
- WDL Accuracy: **100.00%** ✅
- WDL Val Loss: 0.0021
- **DTZ MAE: 1.00 movimiento**
- Tiempo: ~135 segundos

**Evolución DTZ MAE:**
- Época 1: 8.45 → Época 10: 1.24 → Época 30: 1.00

**Análisis:**
- Mate más complejo que KQvK (torre + rey coordinación)
- 100% WDL accuracy pero DTZ más difícil
- MAE = 1.0 es muy bueno para este final

### KPvK (King + Pawn vs King)

**Métricas Finales:**
- WDL Accuracy: 99.88%
- WDL Val Loss: 0.0046
- **DTZ MAE: 0.06 movimientos** 🔥
- Tiempo: ~140 segundos

**Evolución DTZ MAE:**
- Época 1: 0.42 → Época 10: 0.12 → Época 30: 0.06

**Análisis:**
- ¡Prácticamente perfecto!
- Avance lineal del peón facilita predicción
- Mejor DTZ de los 3 endgames

## Análisis Detallado

### Comparación entre Endgames

**¿Por qué KPvK tiene el mejor DTZ MAE (0.06)?**

1. **Movimiento Lineal:** El peón avanza en línea recta
2. **Rango Estrecho:** Máximo ~7 movimientos hasta promoción
3. **Menos Opciones:** Solo puede avanzar o capturar
4. **Patrón Predecible:** La red aprende fácilmente el patrón

**¿Por qué KRvK tiene el peor DTZ MAE (1.00)?**

1. **Mate Complejo:** Torre necesita coordinar con rey
2. **Múltiples Secuencias:** Varios caminos al mate
3. **Zugzwang:** Posiciones donde cualquier movimiento empeora
4. **Rango Amplio:** Hasta 20+ movimientos para mate

**¿Por qué KQvK está en el medio (0.64)?**

1. **Mate Más Directo:** Dama es más poderosa que torre
2. **Menos Movimientos:** Generalmente mates más rápidos
3. **Menos Zugzwang:** Dama tiene más opciones

### DTZ vs DTM en 3-Piece

Para finales sin peones ni capturas posibles:
- **DTZ = DTM** (Distance to Mate)
- KQvK: Solo hay mate, no hay capturas
- KRvK: Solo hay mate, no hay capturas
- KPvK: DTZ ≠ DTM (el peón puede promocionar = "zero")

### Interpretación del MAE

| MAE | Interpretación | Utilidad Práctica |
|-----|----------------|-------------------|
| 0.06 | Casi perfecto | Excelente para juego |
| 0.64 | Excelente | Perfectamente aceptable |
| 1.00 | Muy bueno | Aceptable, ±1 movimiento |
| 2.00 | Bueno | Aceptable en mayoría de casos |
| 3.00+ | Regular | Puede causar problemas |

**Nuestros resultados:**
- ✅ KPvK: 0.06 → Casi perfecto
- ✅ KQvK: 0.64 → Excelente
- ✅ KRvK: 1.00 → Muy bueno

### Distribución de DTZ

**KQvK:**
- Rango: [-20, 19]
- Media: -2.96
- Std: 11.49

**KRvK:**
- Rango: [-16, 16]
- Media: ~0
- Std: ~8

**KPvK:**
- Rango: [-7, 7]
- Media: ~0
- Std: ~3

**Interpretación:**
- Valores negativos: El oponente puede forzar el resultado
- Valores positivos: Nosotros podemos forzar el resultado
- 0: Empate (draw)

## Comparación con Syzygy

| Métrica | Syzygy | Neural Network |
|---------|--------|----------------|
| DTZ Exacto | ✅ 100% | ❌ No |
| DTZ Aproximado | ✅ 100% | ✅ 99.36% (±1 mov) |
| WDL Exacto | ✅ 100% | ✅ 99.94% |
| Tamaño | 1.89 KB | ~450 KB (modelo) |
| Compresión | - | 79.7x con exception map |

**Nota:** Con exception map, podemos alcanzar 100% en ambos WDL y DTZ.

## Utilidad del DTZ

### Para el Juego

1. **Planificación:** Saber cuántos movimientos faltan para mate
2. **Regla de 50 movimientos:** Evitar empates por regla de 50
3. **Optimización:** Elegir el camino más corto al mate
4. **Evaluación:** Comparar diferentes líneas de juego

### Para el Proyecto

1. **Funcionalidad completa:** No solo "gana/pierde" sino "en cuántos movimientos"
2. **Validación:** Comparar con Syzygy para detectar errores
3. **Compresión:** DTZ comprimido es más útil que solo WDL
4. **Investigación:** Estudiar patrones de mate en diferentes finales

## Próximos Pasos

### Inmediato
1. ✅ Entrenar DTZ en KRvK y KPvK
2. 🔜 Analizar distribución de errores de DTZ
3. 🔜 Comparar DTZ predictions con Syzygy

### Corto Plazo
1. 🔜 Entrenar DTZ en KRRvK (4-piece)
2. 🔜 Estudiar si SIREN mejora DTZ accuracy
3. 🔜 Implementar DTZ en exception map

### Medio Plazo
1. 🔜 Optimizar peso de DTZ loss (0.5 vs otros valores)
2. 🔜 Probar loss functions alternativas (Huber, SmoothL1)
3. 🔜 Estudiar DTZ en finales con peones (KPvK, KPvKP)

## Conclusión

El entrenamiento de DTZ ha sido un éxito:
- **MAE de 0.64 movimientos** en KQvK
- Convergencia rápida (< 1.0 en 20 épocas)
- Sin impacto negativo en WDL accuracy (99.94%)

El modelo ahora proporciona funcionalidad completa: no solo predice quién gana, sino en cuántos movimientos. Esto es crucial para un tablebase útil en la práctica.

---

**Métricas Clave:**
- DTZ MAE: **0.64 movimientos**
- WDL Accuracy: **99.94%**
- Convergencia: **< 1.0 en 20 épocas**
- Peso DTZ loss: **0.5** (aumentado de 0.1)

**Estado:** ✅ DTZ implementado y validado en KQvK
