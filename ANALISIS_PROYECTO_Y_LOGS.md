# Análisis del Proyecto y Logs de Entrenamiento

## 1. Resumen del Proyecto

**Neural Tablebases** es un sistema para comprimir tablas de finales de ajedrez (Syzygy) usando redes neuronales. El objetivo es lograr compresión con 100% de precisión.

### Arquitectura del Proyecto
- **Dataset**: KQvK (Rey y Dama vs Rey) - 368,452 posiciones
- **Modelos implementados**: MLP y SIREN
- **Técnicas**: Hard Example Mining, entrenamiento adaptativo

### Estructura de Archivos
```
src/
├── models.py          # Implementación de MLP y SIREN
├── train.py           # Script de entrenamiento con hard mining
└── generate_datasets.py  # Generación de datasets desde Syzygy

docs/
├── implementation_plan.md
├── NEURAL_TABLEBASE_DESIGN.md
└── walkthrough.md

analyze_models.py      # Script de análisis de modelos
```

---

## 2. Análisis del Log de Entrenamiento

### Configuración del Entrenamiento
- **Modelo**: SIREN (Sinusoidal Representation Networks)
- **Parámetros**: 16,902
- **Arquitectura**: 2 capas ocultas de 64 neuronas
- **Dataset**: 368,452 posiciones (90% train, 10% val)
- **Batch size**: 4,096
- **Learning rate inicial**: 0.001
- **Hard mining**: Cada 10 épocas, 5 épocas de re-entrenamiento

### Progreso del Entrenamiento (92/1000 épocas)

#### Métricas Clave
| Época | Train Loss | Val Loss | Train Acc | Val Acc | LR | Observaciones |
|-------|------------|----------|-----------|---------|-----|---------------|
| 1 | 17.0602 | 0.8436 | 47.42% | 50.24% | 0.001 | Inicio |
| 5 | 14.2678 | 0.7012 | 59.21% | 58.78% | 0.001 | Mejora rápida |
| 10 | 14.0901 | 0.6910 | 59.34% | 58.80% | 0.001 | Primer hard mining |
| 17 | 14.1675 | 0.6808 | 59.68% | **59.52%** | 0.001 | Mejor val acc |
| 20 | 14.2379 | 0.6822 | 59.73% | 58.77% | 0.001 | Hard mining |
| 30 | 14.3424 | 0.6752 | 59.64% | 58.11% | 0.0005 | LR reducido |
| 40 | 14.5113 | 0.6709 | 59.97% | 54.88% | 0.00025 | Sobreajuste visible |
| 51 | 16.1387 | 0.6769 | 56.63% | **60.09%** | 0.000125 | **Mejor val acc** |
| 60 | 14.9609 | 0.6696 | 60.55% | 53.46% | 0.000125 | Hard mining |
| 70 | 15.2218 | 0.6708 | 60.71% | 54.52% | 0.000063 | LR muy bajo |
| 80 | 15.5448 | 0.6721 | 60.62% | 57.56% | 0.000031 | Estancamiento |
| 90 | 15.9754 | 0.6766 | 60.24% | 58.90% | 0.000016 | LR mínimo |
| 92 | 17.9197 | 0.6927 | 56.83% | 57.30% | 0.000016 | Deterioro |

### Patrones Observados

#### 1. **Ciclo de Hard Mining Problemático**
Cada 10 épocas se ejecuta el "Overfitting Loop":
- Época 10: Train loss salta de 14.09 → 14.28 (+1.3%)
- Época 20: Train loss salta de 14.24 → 14.58 (+2.4%)
- Época 30: Train loss salta de 14.34 → 14.82 (+3.3%)
- Época 40: Train loss salta de 14.51 → 15.34 (+5.7%)
- Época 50: Train loss salta de 14.74 → 16.14 (+9.5%)
- Época 60: Train loss salta de 14.96 → 16.26 (+8.7%)
- Época 70: Train loss salta de 15.22 → 17.10 (+12.4%)
- Época 80: Train loss salta de 15.54 → 17.79 (+14.5%)
- Época 90: Train loss salta de 15.98 → 19.53 (+22.2%)

**Problema**: El hard mining está causando un deterioro progresivo del modelo.

#### 2. **Sobreajuste a Ejemplos Difíciles**
- Train accuracy aumenta gradualmente (47% → 60%)
- Val accuracy fluctúa y no mejora consistentemente
- Después de cada hard mining, val accuracy cae

#### 3. **Learning Rate Demasiado Agresivo**
- LR se reduce de 0.001 → 0.000016 en 90 épocas
- Factor de reducción: 62.5x
- Esto impide que el modelo recupere el rendimiento después del hard mining

---

## 3. Análisis del Dataset

### Distribución de Clases WDL
| Clase | Significado | Cantidad | Porcentaje |
|-------|-------------|----------|------------|
| -2 | Pérdida | 200,896 | 54.5% |
| 0 | Tablas | 23,048 | 6.3% |
| 2 | Victoria | 144,508 | 39.2% |

### Observaciones
- **Desbalance de clases**: 54.5% pérdidas vs 39.2% victorias
- **Pocas tablas**: Solo 6.3% son posiciones de tablas
- **Rango DTZ**: -20 a 19 (distanza a tablas)

---

## 4. Análisis de Modelos Guardados

### Modelos MLP
| Archivo | Tamaño | Parámetros | Observaciones |
|---------|--------|------------|---------------|
| mlp_best.pth | 0.06 MB | 15,016 | Mejor modelo (arquitectura pequeña) |
| mlp_checkpoint_e100.pth | 0.06 MB | 15,016 | Checkpoint temprano |
| mlp_checkpoint_e200-400.pth | 0.91 MB | 238,406 | Arquitectura más grande |
| mlp_final.pth | 0.06 MB | 15,016 | Modelo final |
| mlp_model.pth | 0.91 MB | 238,406 | Modelo alternativo |

### Modelos SIREN
| Archivo | Tamaño | Parámetros | Observaciones |
|---------|--------|------------|---------------|
| siren_best.pth | 0.07 MB | 16,902 | Mejor modelo actual |
| siren_model.pth | 0.51 MB | 132,230 | Modelo más grande |

**Nota**: Existen inconsistencias en las arquitecturas guardadas (15K vs 238K parámetros para MLP).

---

## 5. Problemas Identificados

### Críticos
1. **Hard Mining Destructivo**: El ciclo de hard mining está destruyendo el aprendizaje
2. **Sobreajuste Severo**: El modelo memoriza ejemplos difíciles en lugar de generalizar
3. **LR Scheduler Demasiado Agresivo**: Reduce LR demasiado rápido

### Moderados
4. **Desbalance de Clases**: 54.5% vs 39.2% puede afectar el aprendizaje
5. **Inconsistencia de Arquitecturas**: Diferentes tamaños de modelo guardados
6. **Falta de Regularización**: No se usa dropout en SIREN

### Menores
7. **Logging**: No se loggea el número de ejemplos difíciles encontrados
8. **Checkpointing**: No se guardan checkpoints del mejor modelo de hard mining

---

## 6. Recomendaciones

### Inmediatas (Para el Entrenamiento Actual)

1. **Detener el Hard Mining Temporalmente**
   ```python
   # En train.py, comentar o reducir hard_mining_freq
   parser.add_argument("--hard_mining_freq", type=int, default=50,  # Era 10
   ```

2. **Ajustar el Learning Rate Scheduler**
   ```python
   # Usar un scheduler más conservador
   scheduler = optim.lr_scheduler.ReduceLROnPlateau(
       optimizer, 'max', patience=20, factor=0.7, min_lr=1e-5
   )
   ```

3. **Reducir el Peso de Ejemplos Difíciles**
   ```python
   parser.add_argument("--hard_weight", type=float, default=2.0,  # Era 5.0
   ```

### Corto Plazo

4. **Implementar Balanceo de Clases**
   ```python
   # Calcular pesos por clase
   class_counts = np.bincount(dataset.wdl.numpy())
   class_weights = 1.0 / class_counts
   criterion_wdl = nn.CrossEntropyLoss(
       weight=torch.FloatTensor(class_weights).to(device)
   )
   ```

5. **Añadir Regularización a SIREN**
   ```python
   class SIREN(nn.Module):
       def __init__(self, ..., dropout=0.1):
           # Añadir dropout entre capas
   ```

6. **Implementar Early Stopping Basado en Val Loss**
   ```python
   # En lugar de solo val_acc, considerar val_loss
   if val_loss_avg < best_val_loss:
       best_val_loss = val_loss_avg
       # Guardar modelo
   ```

### Largo Plazo

7. **Implementar Curriculum Learning**
   - Entrenar primero con ejemplos fáciles
   - Introducir gradualmente ejemplos difíciles

8. **Probar Otras Arquitecturas**
   - KAN (Kolmogorov-Arnold Networks)
   - Transformers para secuencias de posiciones

9. **Mejorar la Generación de Datasets**
   - Balancear clases durante la generación
   - Aumentar el número de posiciones de tablas

---

## 7. Próximos Pasos Sugeridos

### Prioridad Alta
1. [ ] Detener el entrenamiento actual si está en progreso
2. [ ] Implementar las recomendaciones inmediatas
3. [ ] Re-entrenar con configuración corregida
4. [ ] Monitorear val_loss en lugar de solo val_acc

### Prioridad Media
5. [ ] Implementar balanceo de clases
6. [ ] Añadir regularización a SIREN
7. [ ] Crear script de evaluación completo
8. [ ] Documentar configuraciones óptimas

### Prioridad Baja
9. [ ] Probar arquitectura KAN
10. [ ] Implementar curriculum learning
11. [ ] Optimizar para GPU (actualmente parece usar CPU)
12. [ ] Crear dashboard de monitoreo en tiempo real

---

## 8. Conclusión

El proyecto tiene una base sólida con una implementación correcta de SIREN y hard mining. Sin embargo, la configuración actual del hard mining es **demasiado agresiva** y está causando un deterioro progresivo del modelo.

**Estado Actual**: ⚠️ Entrenamiento subóptimo
**Mejor Val Accuracy**: 60.09% (época 51)
**Problema Principal**: Hard mining destructivo + LR scheduler agresivo

Con las correcciones sugeridas, se espera alcanzar:
- **Val Accuracy**: 75-85% en 100 épocas
- **Estabilidad**: Sin deterioro después de hard mining
- **Generalización**: Mejor rendimiento en posiciones no vistas

---

*Análisis generado el 2026-03-12 21:20 UTC*
