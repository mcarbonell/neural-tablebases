# Resumen de Cambios - Neural Tablebase Optimization

## Fecha: 2026-03-12

### Problemas Identificados

1. **Dataset desbalanceado**: 54% pérdidas, 39% victorias, 6% empates
2. **Clases no utilizadas**: Solo 3 clases WDL (-2, 0, 2) en lugar de 5
3. **Modelo demasiado pequeño**: 14K-16K parámetros para 368K posiciones
4. **Precisión estancada**: ~58-60% de precisión máxima

### Cambios Implementados

#### 1. Mapeo de Clases (src/train.py)
- **Antes**: WDL {-2, -1, 0, 1, 2} → {0, 1, 2, 3, 4} (5 clases)
- **Ahora**: WDL {-2, 0, 2} → {0, 1, 2} (3 clases)
- Añadido cálculo automático de class weights para balancear el dataset

#### 2. Arquitectura del Modelo (src/models.py)

**MLP para 3 piezas:**
- **Antes**: [64, 32] → ~14K parámetros
- **Ahora**: [512, 512, 256, 128] → ~529K parámetros (37x más grande)
- Dropout aumentado de 0.1 a 0.2

**SIREN para 3 piezas:**
- **Antes**: hidden_size=64, num_layers=2 → ~16K parámetros
- **Ahora**: hidden_size=256, num_layers=4 → ~330K parámetros (20x más grande)
- Añadido dropout=0.1

#### 3. Hiperparámetros de Entrenamiento
- Épocas: 500 → 1000
- Patience: 50 → 100
- Hard weight: 5.0 → 2.0 (menos agresivo)
- Hard mining freq: 10 → 50 (menos frecuente)

### Resultados Preliminares (5 épocas de prueba)

**MLP Mejorado:**
- Parámetros: 529,028 (vs 14,822 anterior)
- Época 1: 45.5% acc
- Época 5: 61.2% acc
- Tendencia: Mejora continua, sin estancamiento

### Próximos Pasos

1. **Entrenamiento largo**: Ejecutar 1000 épocas para ver si alcanza >90% accuracy
2. **Monitoreo**: Verificar que el modelo no se estanque
3. **Evaluación**: Si llega a 95%+, considerar aumentar aún más la capacidad
4. **Alternativa**: Si se estanca, probar con un endgame más simple (KRvK)

### Comandos para Entrenar

**Windows:**
```cmd
train_improved.bat
```

**Linux/Mac:**
```bash
bash train_improved.sh
```

**Manual:**
```bash
# MLP
python src/train.py --data_path data/KQvK.npz --model mlp --epochs 1000 --batch_size 2048 --lr 0.001 --patience 100 --hard_weight 2.0 --hard_mining --hard_mining_freq 50 --hard_mining_epochs 3

# SIREN
python src/train.py --data_path data/KQvK.npz --model siren --epochs 1000 --batch_size 2048 --lr 0.0005 --patience 100 --hard_weight 2.0 --hard_mining --hard_mining_freq 50 --hard_mining_epochs 3
```

### Verificación de Datos

Ejecutado `verify_dataset.py`:
- ✅ 368,452 posiciones totales
- ✅ Sin NaN ni valores infinitos
- ✅ Encoding correcto (3 piezas por posición)
- ✅ 100% coincidencia con Syzygy en 100 muestras aleatorias

### Notas Técnicas

- El modelo ahora tiene suficiente capacidad para memorizar el dataset completo
- Los class weights ayudan a balancear el aprendizaje entre clases desbalanceadas
- El hard mining menos agresivo evita la desestabilización del modelo
- El dropout ayuda a prevenir overfitting prematuro en las primeras épocas
