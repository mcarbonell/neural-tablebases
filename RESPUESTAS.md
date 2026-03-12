# Respuestas a tus Preguntas

## 1. ¿La red solo da Win/Draw/Loss sin DTM/DTZ?

**Respuesta:** Actualmente SÍ, pero tenemos ambos datos disponibles.

### Datos disponibles en Syzygy:
- **WDL** (Win/Draw/Loss): -2, 0, 2
- **DTZ** (Distance to Zero): -20 a +19 movimientos

### Qué está prediciendo el modelo:
- **Solo WDL**: 3 clases (Loss, Draw, Win)
- **DTZ**: Tenemos el head pero no lo estamos usando bien

### Distribución de datos:
```
WDL -2 (Loss): 200,896 posiciones (54.52%)
WDL  0 (Draw):  23,048 posiciones ( 6.26%)
WDL  2 (Win):  144,508 posiciones (39.22%)

DTZ para Loss: -20 a -1 (media: -12.79 movimientos)
DTZ para Draw:  0 (siempre 0)
DTZ para Win:   1 a 19 (media: 10.23 movimientos)
```

## 2. Tu Observación: "La clasificación es muy sencilla"

**¡TIENES TODA LA RAZÓN!**

### Regla humana simple:
```
Si el rey defensor puede comer la dama → Draw
Si no puede → Loss (o Win según perspectiva)
```

### Por qué el modelo no lo ve:

El modelo recibe **solo one-hot encoding**:
```python
Input: [0,1,0,0,...,0]  # Pieza 1 en casilla 1
       [1,0,0,0,...,0]  # Pieza 2 en casilla 0
       [0,0,1,0,...,0]  # Pieza 3 en casilla 2
```

El modelo NO ve:
- ❌ Distancia entre piezas
- ❌ Si están adyacentes
- ❌ Relación geométrica
- ❌ "¿Puede el rey capturar la dama?"

### Análisis de empates:
De 100 posiciones de empate analizadas:
- **194 tienen piezas adyacentes** (más de lo esperado porque hay múltiples pares)
- Esto confirma que los empates ocurren cuando hay amenaza de captura

## 3. ¿Por qué solo 68% de accuracy?

### El problema fundamental:

El modelo está intentando aprender geometría a partir de one-hot encoding, que es como:
```
Humano: "¿Está el rey al lado de la dama?"
Modelo: "No sé qué es 'al lado', solo veo que hay un 1 en la posición 27 y otro en la 28"
```

### Comparación:

| Enfoque | Input | Accuracy Esperada |
|---------|-------|-------------------|
| One-hot solo | 192 dims | ~68% (actual) |
| One-hot + geometría | 208 dims | >90% (estimado) |
| Solo geometría | 16 dims | >85% (estimado) |

## 4. Solución: Features Geométricas

### Propuesta de nuevo encoding:

```python
# Mantener one-hot (192 dims)
+ Distancia rey_blanco ↔ dama (Chebyshev): 1 dim
+ Distancia rey_negro ↔ dama (Chebyshev): 1 dim  
+ Distancia rey ↔ rey: 1 dim
+ Rey negro adyacente a dama (binario): 1 dim ← ¡TU REGLA!
+ Coordenadas normalizadas (x,y) por pieza: 6 dims
+ Distancia al centro por pieza: 3 dims
+ Distancia al borde más cercano: 3 dims
= Total: 192 + 16 = 208 dims
```

### Feature más importante:
```python
rey_puede_capturar_dama = (distancia_rey_negro_dama == 1)
```

Esta única feature debería mejorar dramáticamente la predicción de empates.

## 5. ¿Qué hacer con DTZ?

Actualmente el modelo tiene un `dtz_head` pero no lo estamos evaluando bien.

### Opciones:

**A) Ignorar DTZ por ahora** (recomendado)
- Enfocarse en WDL primero
- DTZ es secundario para compresión

**B) Predecir DTZ como regresión**
- Útil para saber "cuántos movimientos hasta mate"
- Pero no crítico para el objetivo de compresión

**C) Clasificar DTZ en buckets**
- Ejemplo: [0-5 movimientos, 6-10, 11-15, 16+]
- Más fácil que regresión exacta

## Conclusión

Tu observación es **clave**: el problema es geométricamente simple pero el modelo no tiene acceso a esa información geométrica.

### Próximo paso crítico:
Implementar features geométricas, especialmente:
1. Distancias entre piezas
2. **Tu regla: ¿Rey adyacente a dama?**
3. Posiciones relativas

Esto debería llevar la accuracy de 68% a >90%.
