# KRvKP: Rey+Torre vs Rey+Peón

## El Reto Asimétrico

**Configuración:** Rey+Torre (blanco) vs Rey+Peón (negro)

**Dificultad:** ⭐⭐⭐ Media-Alta (más difícil que KRRvK)

**Por qué es interesante:**
1. **Asimetría total:** Torre vs Peón (material muy diferente)
2. **Amenaza de promoción:** El peón puede convertirse en Dama
3. **Táctica pura:** La torre debe parar el peón en el momento exacto
4. **Posiciones críticas:** Un movimiento puede cambiar Win → Draw

## Teoría del Endgame

### Reglas Generales:

**Blanco gana si:**
1. Torre captura el peón antes de promocionar
2. Rey blanco ayuda a capturar el peón
3. Torre da jaques perpetuos después de promoción

**Negro empata/gana si:**
1. Peón promociona con rey protegiéndolo
2. Rey negro bloquea la torre
3. Ahogado del rey blanco

### Posiciones Críticas:

```
1. Peón en rank 6, torre lejos:
   ♜ . . . . . . .
   . . . . . . . .
   . . . ♟ . . . .
   . . . ♚ . . . .
   . . . . . . . .
   . . . . . . . .
   . . . . . . . .
   ♖ . . ♔ . . . .
   
   Resultado: Draw (peón promociona)

2. Peón en rank 6, torre cerca:
   . . . ♜ . . . .
   . . . . . . . .
   . . . ♟ . . . .
   . . . ♚ . . . .
   . . . . . . . .
   . . . . . . . .
   . . . . . . . .
   . . . ♔ . . . .
   
   Resultado: Win (torre captura peón)

3. Peón en rank 2, rey protege:
   . . . . . . . .
   . . . . . . . .
   . . . . . . . .
   . . . . . . . .
   . . . . . . . .
   . . . ♚ . . . .
   . . . ♟ . . . .
   ♖ . . ♔ . . . .
   
   Resultado: Win (torre captura fácilmente)
```

## Distribución WDL Esperada

### Estimación:

```
Win (blanco):  55-65%  (Torre captura peón o da jaques)
Draw:          25-35%  (Peón promociona, ahogado)
Loss (blanco): 5-10%   (Peón promociona y gana)
```

**Comparación con otros endgames:**

| Endgame | Win | Draw | Loss |
|---------|-----|------|------|
| KQvK | 39% | 6% | 55% |
| KRvK | 39% | 6% | 55% |
| KPvK | 38% | 33% | 29% |
| KRRvK | ~40% | ~5% | ~55% |
| **KRvKP** | **60%** | **30%** | **10%** |

**Más empates que KRRvK** porque el peón puede promocionar.

## Encoding Relativo: 65 Dimensiones

### Piezas (4 × 10 = 40 dims):

```python
Pieza 1: Rey blanco
  - Coords (x, y): 2 dims
  - Tipo [K,Q,R,B,N,P]: [1,0,0,0,0,0]
  - Color [W,B]: [1,0]

Pieza 2: Torre blanca
  - Coords (x, y): 2 dims
  - Tipo: [0,0,1,0,0,0]  # Rook
  - Color: [1,0]

Pieza 3: Rey negro
  - Coords (x, y): 2 dims
  - Tipo: [1,0,0,0,0,0]
  - Color: [0,1]

Pieza 4: Peón negro
  - Coords (x, y): 2 dims
  - Tipo: [0,0,0,0,0,1]  # Pawn
  - Color: [0,1]
```

### Pares (6 × 4 = 24 dims):

```python
Par 1: Rey blanco ↔ Torre blanca
  - Manhattan, Chebyshev, dx, dy

Par 2: Rey blanco ↔ Rey negro
  - Manhattan, Chebyshev, dx, dy

Par 3: Rey blanco ↔ Peón negro
  - Manhattan, Chebyshev, dx, dy

Par 4: Torre blanca ↔ Rey negro
  - Manhattan, Chebyshev, dx, dy

Par 5: Torre blanca ↔ Peón negro  ← CRÍTICO
  - Manhattan, Chebyshev, dx, dy
  - Distancia torre-peón determina si puede parar

Par 6: Rey negro ↔ Peón negro  ← CRÍTICO
  - Manhattan, Chebyshev, dx, dy
  - Rey protege el peón
```

### Global (1 dim):

```python
Side to move: 1 dim
```

**Total: 40 + 24 + 1 = 65 dims**

## Features Críticas para KRvKP

### 1. Distancia Torre-Peón:

```python
chebyshev_dist(torre, peon) < 2 → Torre puede capturar
chebyshev_dist(torre, peon) > 3 → Peón puede escapar
```

### 2. Distancia Peón-Promoción:

```python
pawn_rank = peon.y * 7  # 0-7
dist_to_promotion = 7 - pawn_rank  # Para peón negro

dist_to_promotion == 1 → Crítico (1 movimiento)
dist_to_promotion == 2 → Urgente (2 movimientos)
dist_to_promotion >= 4 → Tiempo suficiente
```

### 3. Rey Negro Protege:

```python
chebyshev_dist(rey_negro, peon) <= 1 → Protegido
chebyshev_dist(rey_negro, peon) > 2 → Desprotegido
```

### 4. Coordinación Blanca:

```python
# Rey blanco ayuda a torre
chebyshev_dist(rey_blanco, peon) < 3 → Ayuda
chebyshev_dist(rey_blanco, peon) > 5 → No ayuda
```

## El Modelo Aprenderá:

### Regla 1: Torre Cerca del Peón
```python
if chebyshev_dist(torre, peon) <= 1:
    if dist_to_promotion > 1:
        return WIN  # Torre captura
```

### Regla 2: Peón Cerca de Promoción
```python
if dist_to_promotion == 1:
    if chebyshev_dist(torre, peon) > 2:
        return DRAW/LOSS  # Peón promociona
```

### Regla 3: Rey Negro Protege
```python
if chebyshev_dist(rey_negro, peon) <= 1:
    if chebyshev_dist(torre, peon) > 2:
        return DRAW  # Difícil capturar
```

### Regla 4: Coordinación
```python
if chebyshev_dist(rey_blanco, peon) < 3:
    if chebyshev_dist(torre, peon) < 3:
        return WIN  # Coordinación perfecta
```

## Accuracy Esperado

### Por Época:

| Época | Train Acc | Val Acc | Razón |
|-------|-----------|---------|-------|
| 1 | 92-95% | 94-96% | Más complejo que KPvK |
| 5 | 97-98% | 98-99% | Aprende tácticas básicas |
| 10 | 98-99% | 99-99.5% | Domina mayoría de casos |
| 20 | 99%+ | 99.5%+ | Casos difíciles |
| 30 | 99.5%+ | 99.7%+ | Cerca de perfecto |

### Comparación:

| Endgame | Época 1 | Época 10 | Mejor | Dificultad |
|---------|---------|----------|-------|------------|
| KQvK | 98.07% | 99.77% | 99.92% | Fácil |
| KRvK | 99.68% | 99.98% | 99.99% | Fácil |
| KPvK | 96.59% | 99.43% | 99.89% | Media |
| KRRvK | ~99%? | ~99.9%? | ~99.95%? | Fácil |
| **KRvKP** | **95%** | **99%** | **99.7%** | **Media-Alta** |

## Hard Examples Esperados

### Casos Difíciles:

1. **Peón en rank 6, torre a 2 casillas:**
   - ¿Llega a tiempo?
   - Depende del turno

2. **Rey negro en casilla de promoción:**
   - Bloquea la torre
   - Puede ser ahogado

3. **Torre da jaque, peón promociona:**
   - Jaque perpetuo vs Dama nueva
   - Complejo de evaluar

4. **Peón pasado protegido:**
   - Rey negro detrás del peón
   - Torre no puede capturar

**Hard examples esperados:** 500-1,000 (vs 66 en KPvK)

## Tiempo de Entrenamiento

### Estimación:

```
Posiciones esperadas: ~2-3M
Batch size: 2,048
Batches por época: ~1,000-1,500
Tiempo por época: ~2 min
Épocas necesarias: 20-30

Total: 40-60 minutos
```

## Métricas de Éxito

### Mínimo Aceptable:
- Accuracy: >99%
- Épocas: <40
- Hard examples: <1,000

### Objetivo:
- Accuracy: >99.5%
- Épocas: <30
- Hard examples: <500

### Excelente:
- Accuracy: >99.7%
- Épocas: <20
- Hard examples: <200

## Posibles Problemas

### 1. Casos Tácticos Complejos
**Síntoma:** Accuracy se estanca en 97-98%  
**Solución:** Más épocas, modelo más grande

### 2. Promoción Difícil de Aprender
**Síntoma:** Errores en peones en rank 6-7  
**Solución:** Hard mining más agresivo

### 3. Asimetría Confunde al Modelo
**Síntoma:** Confunde Win/Draw  
**Solución:** Class weights ajustados

## Comparación con Syzygy

```
Syzygy KRvKP: ~15-20 KB (estimado)
Neural KRvKP: Compartido con otros 4-piezas (~500 KB)

Si es archivo individual:
  Compresión: 0.03x (más grande)
  
Si es parte de modelo universal:
  Compresión: 40x (30 endgames / 500 KB)
```

## Plan de Ejecución

### 1. Generar Dataset
```bash
python src/generate_datasets.py --config KRvKP --relative
```
Tiempo: ~10-15 min

### 2. Verificar Dataset
```python
- Posiciones: ~2-3M
- Encoding: 65 dims
- WDL: ~60% Win, ~30% Draw, ~10% Loss
- Peones en ranks 1-6
```

### 3. Entrenar Modelo
```bash
python src/train.py --data_path data/KRvKP.npz --model mlp --epochs 30
```
Tiempo: ~40-60 min

### 4. Analizar Resultados
- Accuracy por época
- Hard examples
- Casos difíciles específicos

## Conclusión

**KRvKP es el reto perfecto** para validar el encoding relativo:

✅ **Asimetría:** Torre vs Peón (material muy diferente)  
✅ **Táctica:** Timing crítico para parar promoción  
✅ **Complejidad:** Más difícil que KPvK  
✅ **Realista:** Endgame común en partidas reales

**Predicción:**
- Accuracy: 99.5-99.7%
- Épocas: 20-30
- Hard examples: 200-500

Si funciona bien, confirma que el encoding relativo maneja:
- Asimetría de material
- Tácticas complejas
- Promoción de peones
- Coordinación de piezas

**¡Será una prueba excelente!** 🎯

---

**Próximo paso:** Esperar a que termine KRRvK, luego generar y entrenar KRvKP
