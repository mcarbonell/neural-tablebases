# SIREN vs MLP: Comparación Final

**Fecha:** 13 de marzo de 2026  
**Estado:** ✅ Completado  
**Conclusión:** MLP es claramente superior

## Resultados Completos

### Tabla Comparativa

| Endgame | MLP WDL | SIREN WDL | Δ | MLP DTZ | SIREN DTZ | Δ |
|---------|---------|-----------|---|---------|-----------|---|
| KQvK | 99.94% | 98.70% | **-1.24%** | 0.64 | 2.21 | **+1.57** |
| KRvK | 100.00% | 98.34% | **-1.66%** | 1.00 | 4.32 | **+3.32** |
| KPvK | 99.88% | 92.92% | **-6.96%** | 0.06 | 0.40 | **+0.34** |
| **Promedio** | **99.94%** | **96.65%** | **-3.29%** | **0.57** | **2.31** | **+1.74** |

### Detalles por Endgame

**KQvK:**
- MLP: 99.94% WDL, 0.64 DTZ MAE
- SIREN: 98.70% WDL, 2.21 DTZ MAE
- **Ganador:** MLP (+1.24% WDL, -1.57 DTZ MAE)

**KRvK:**
- MLP: 100.00% WDL, 1.00 DTZ MAE
- SIREN: 98.34% WDL, 4.32 DTZ MAE
- **Ganador:** MLP (+1.66% WDL, -3.32 DTZ MAE)

**KPvK:**
- MLP: 99.88% WDL, 0.06 DTZ MAE
- SIREN: 92.92% WDL, 0.40 DTZ MAE
- **Ganador:** MLP (+6.96% WDL, -0.34 DTZ MAE)

## Análisis

### ¿Por qué SIREN falla?

#### 1. Naturaleza del Problema
- **WDL es discreto:** Solo 3 valores (-2, 0, 2)
- **DTZ es discreto:** Valores enteros
- **SIREN está diseñado para funciones continuas suaves**
- Las transiciones sharp de tablebases no son ideales para SIREN

#### 2. Arquitectura
- SIREN usa `sin(ω₀ × x)` en lugar de ReLU
- Funciones periódicas no son apropiadas para clasificación discreta
- La periodicidad introduce ambigüedad

#### 3. Convergencia
- SIREN converge mucho más lento que MLP
- Requiere más épocas para resultados comparables
- Más difícil de entrenar (sensible a hiperparámetros)

#### 4. Hiperparámetros
- `omega_0 = 30` puede ser demasiado alto
- Learning rate 0.001 puede ser demasiado alto
- Arquitectura (128 hidden, 3 layers) puede ser insuficiente

### Ventajas Teóricas de SIREN (No Aplicables Aquí)

1. ✅ Mejor para funciones continuas → ❌ Tablebases son discretos
2. ✅ Captura derivadas → ❌ No necesitamos derivadas
3. ✅ Representación implícita → ❌ Clasificación explícita es mejor
4. ✅ Funciones periódicas → ❌ No hay periodicidad en tablebases

### Ventajas de MLP para Tablebases

1. ✅ **Diseñado para clasificación discreta**
2. ✅ **Convergencia rápida** (30 épocas suficientes)
3. ✅ **Fácil de entrenar** (hiperparámetros estándar)
4. ✅ **Mejor accuracy** (99.94% vs 96.65%)
5. ✅ **Mejor DTZ** (0.57 vs 2.31 MAE)

## Conclusión

**MLP es claramente superior para tablebases de ajedrez:**

### Accuracy
- MLP: 99.94% promedio
- SIREN: 96.65% promedio
- **Diferencia: +3.29% a favor de MLP**

### DTZ MAE
- MLP: 0.57 movimientos promedio
- SIREN: 2.31 movimientos promedio
- **Diferencia: -1.74 movimientos a favor de MLP**

### Convergencia
- MLP: Rápida (98%+ en 5 épocas)
- SIREN: Lenta (95%+ en 20 épocas)

### Facilidad de Entrenamiento
- MLP: Hiperparámetros estándar funcionan bien
- SIREN: Requiere optimización cuidadosa

## Recomendación

**Usar MLP como arquitectura estándar para el proyecto:**

1. ✅ Mejor accuracy en todos los endgames
2. ✅ Mejor DTZ MAE en todos los endgames
3. ✅ Convergencia más rápida
4. ✅ Más fácil de entrenar
5. ✅ Más apropiado para el problema

**No continuar con SIREN:**
- No justifica la complejidad adicional
- Resultados consistentemente inferiores
- No hay ventaja teórica para este problema

## Lecciones Aprendidas

### Para Arquitecturas Neuronales

1. **Matching problem to architecture:** SIREN es para funciones continuas, tablebases son discretos
2. **Simplicidad gana:** MLP simple supera a SIREN complejo
3. **Convergencia importa:** Entrenamiento rápido permite iteración rápida

### Para el Proyecto

1. **MLP es suficiente:** No necesitamos arquitecturas exóticas
2. **Focus en datos:** Mejor encoding (v2) > mejor arquitectura
3. **Validación rápida:** 3-piece permite probar ideas rápidamente

## Próximos Pasos

1. ✅ Continuar con MLP para 4+ piezas
2. ✅ Implementar WDL 5 clases (cursed wins)
3. ✅ Implementar canonical forms
4. ✅ Entrenar KRRvK con dataset completo

---

**Decisión Final:** MLP es la arquitectura estándar para neural tablebases  
**SIREN:** No recomendado para este problema  
**Próximo experimento:** WDL 5 clases con MLP
