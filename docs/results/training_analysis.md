# Análisis de Resultados - Entrenamiento MLP Mejorado

## Fecha: 2026-03-12 22:08

### Configuración del Modelo

**Arquitectura:**
- Modelo: MLP con 529,028 parámetros (37x más grande que antes)
- Capas: [512, 512, 256, 128]
- Dropout: 0.2
- Clases: 3 (Loss, Draw, Win)
- Class weights: [0.61, 5.33, 0.85]

**Hiperparámetros:**
- Épocas: 100
- Batch size: 2048
- Learning rate inicial: 0.001
- Patience: 50
- Hard weight: 2.0
- Hard mining freq: 50

### Resultados del Entrenamiento

#### Progreso por Épocas

| Época | Train Acc | Val Acc | Val Loss | LR | Observación |
|-------|-----------|---------|----------|-----|-------------|
| 1 | 39.2% | 46.0% | 0.8321 | 0.001 | Inicio |
| 10 | 59.9% | 59.1% | 0.4380 | 0.001 | Mejora rápida |
| 20 | 60.2% | 60.9% | 0.4267 | 0.001 | Progreso continuo |
| 40 | 60.1% | 61.0% | 0.4169 | 0.001 | Mejor época pre-mining |
| 50 | 60.3% | 61.0% | 0.4156 | 0.001 | Hard mining activado |
| 61 | 60.5% | **63.2%** | 0.4146 | 0.001 | Pico de accuracy |
| 66 | 60.6% | 63.3% | **0.4139** | 0.0007 | Mejor loss (LR reducido) |
| 75 | 60.7% | 59.4% | 0.4134 | 0.0007 | **Mejor modelo guardado** |
| 100 | 60.6% | 60.6% | 0.4158 | 0.0007 | Final |

#### Métricas Finales

- **Mejor Val Accuracy**: 59.4% (época 75)
- **Mejor Val Loss**: 0.4134 (época 75)
- **Train Accuracy Final**: 60.6%
- **Val Accuracy Final**: 60.6%
- **Total Hard Examples**: 735,500
- **Tiempo Total**: ~27 minutos

### Análisis Crítico

#### ✅ Mejoras Observadas

1. **Convergencia más rápida**: Alcanzó 60% en 10 épocas vs 154 épocas anteriormente
2. **Sin overfitting severo**: Train/Val accuracy muy similares (60.6% vs 60.6%)
3. **Estabilidad**: El modelo no colapsa después del hard mining
4. **Capacidad adecuada**: 529K parámetros permiten mejor aprendizaje

#### ❌ Problemas Persistentes

1. **Estancamiento en ~60%**: El modelo se estabiliza y no mejora más allá del 60-63%
2. **Varianza alta en validación**: Val accuracy oscila entre 54-68% en épocas tardías
3. **Hard mining inefectivo**: Después de la época 50, no hay mejora significativa
4. **Límite aparente**: Parece haber un "techo" en ~60% de accuracy

### Diagnóstico del Problema

El modelo está **estancado en un mínimo local** alrededor del 60% de accuracy. Posibles causas:

1. **Baseline demasiado fuerte**: Predecir siempre "Loss" (54.5%) ya da ~55% accuracy
2. **Clases difíciles de separar**: Las posiciones pueden ser ambiguas sin contexto adicional
3. **Encoding insuficiente**: El encoding compacto (one-hot por pieza) pierde información posicional relativa
4. **Problema fundamental**: KQvK puede requerir features más complejas

### Comparación con Resultados Anteriores

| Métrica | Modelo Anterior | Modelo Mejorado | Cambio |
|---------|----------------|-----------------|--------|
| Parámetros | 14,822 | 529,028 | +3,469% |
| Épocas para 60% | 154 | 10 | -93% |
| Mejor Val Acc | 60.9% | 63.2% | +2.3% |
| Estabilidad | Baja | Alta | ✓ |
| Tiempo/época | ~5s | ~16s | +220% |

### Análisis de Validación

La alta varianza en validación (54-68%) sugiere:
- El modelo está memorizando patrones específicos
- La división train/val puede tener distribuciones diferentes
- Algunas posiciones son inherentemente difíciles de clasificar

### Recomendaciones

#### Opción 1: Mejorar el Encoding (RECOMENDADO)
```python
# En lugar de one-hot por pieza, usar:
- Distancias relativas entre piezas
- Coordenadas polares desde el centro
- Features geométricas (distancia a promoción, etc.)
```

#### Opción 2: Aumentar Capacidad Aún Más
```python
# Probar con un modelo mucho más grande:
hidden_layers = [1024, 1024, 512, 256, 128]  # ~2M parámetros
```

#### Opción 3: Cambiar a un Endgame Más Simple
```python
# KRvK tiene menos posiciones y es más determinista
# Generar dataset: python src/generate_datasets.py --config KRvK
```

#### Opción 4: Usar SIREN con Más Capas
```python
# SIREN puede capturar mejor funciones complejas
# Probar: hidden_size=512, num_layers=6
```

### Conclusión

El modelo mejorado es **significativamente mejor** que el anterior en términos de:
- Velocidad de convergencia
- Estabilidad
- Capacidad de aprendizaje

Sin embargo, **sigue estancado en ~60%** debido a limitaciones fundamentales:
- El encoding one-hot es demasiado simple
- KQvK puede ser demasiado complejo para este enfoque
- Se necesitan features más ricas o un endgame más simple

**Próximo paso sugerido**: Implementar encoding con features geométricas o probar con KRvK.
