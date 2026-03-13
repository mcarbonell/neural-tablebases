# Optimización del Generador de Datasets

## Problema Actual

**Generador secuencial:**
- Usa 1 solo núcleo (12.5% CPU en 8 núcleos)
- Tarda ~10-15 minutos para 3 piezas
- Tarda ~8+ horas para 4 piezas (estimado)
- Consume ~316 MB RAM

**Para KRRvK:**
- Total combinaciones: 16.7M
- Progreso actual: 9M (54%)
- Tiempo transcurrido: ~8 horas
- Tiempo estimado total: ~15 horas

## Soluciones Posibles

### Opción 1: Multiprocessing (Difícil)

**Problema:** `itertools.permutations` no se puede dividir fácilmente.

**Solución:** Generar índices y convertir a permutaciones.

```python
def index_to_permutation(index, n, k):
    """Convert index to k-permutation of n items"""
    # Complex algorithm
    pass

# Dividir por índices
chunk_size = total_perms // num_workers
for worker in range(num_workers):
    start = worker * chunk_size
    end = (worker + 1) * chunk_size
    # Procesar índices [start, end)
```

**Ventaja:** Usa todos los núcleos (8x más rápido)  
**Desventaja:** Complejo de implementar correctamente

### Opción 2: Sampling en lugar de Exhaustivo

**Idea:** No generar TODAS las posiciones, sino una muestra representativa.

```python
import random

# En lugar de todas las permutaciones
total_samples = 5_000_000  # 5M en lugar de 24M
for _ in range(total_samples):
    # Generar posición aleatoria
    squares = random.sample(chess.SQUARES, num_pieces)
    # Procesar...
```

**Ventaja:** Mucho más rápido (minutos en lugar de horas)  
**Desventaja:** No es exhaustivo, podría perder posiciones importantes

### Opción 3: Usar Syzygy Directamente

**Idea:** Iterar sobre las posiciones que Syzygy ya tiene.

```python
# Syzygy tiene índices internos
# Podríamos iterar sobre ellos directamente
for position_index in range(syzygy_size):
    board = syzygy.get_position(position_index)
    # Procesar...
```

**Ventaja:** Más rápido, garantiza cobertura completa  
**Desventaja:** Requiere acceso interno a Syzygy (no disponible en python-chess)

### Opción 4: Dejar que Termine + Cachear

**Idea:** Dejar que termine esta vez, luego reutilizar el dataset.

```python
# Una vez generado KRRvK.npz, no necesitamos regenerarlo
# Solo entrenar con diferentes hiperparámetros
```

**Ventaja:** Simple, no requiere cambios  
**Desventaja:** Primera generación sigue siendo lenta

## Recomendación

### Para Este Experimento:

**Opción 4:** Dejar que termine KRRvK (ya va por 54%).

**Razones:**
1. Ya llevamos 8 horas, faltan ~7 horas más
2. Solo necesitamos generar cada dataset una vez
3. Podemos entrenar múltiples veces con el mismo dataset
4. Los datasets se pueden reutilizar para diferentes experimentos

### Para el Futuro:

**Opción 2:** Sampling para datasets grandes (5+ piezas).

```python
# Para 5 piezas: 64^5 = 1,073M combinaciones
# Sampling de 10M posiciones es suficiente
# Tiempo: ~30 minutos en lugar de días
```

**Ventaja:** Escalable a cualquier número de piezas.

## Estimación de Tiempos

### Actual (Secuencial):

| Piezas | Combinaciones | Posiciones | Tiempo |
|--------|---------------|------------|--------|
| 3 | 262K | ~350K | 2 min |
| 4 | 16.7M | ~24M | 15 horas |
| 5 | 1,073M | ~500M | 40 días |
| 6 | 68,719M | ~5,000M | 7 años |

### Con Multiprocessing (8 núcleos):

| Piezas | Combinaciones | Posiciones | Tiempo |
|--------|---------------|------------|--------|
| 3 | 262K | ~350K | 15 seg |
| 4 | 16.7M | ~24M | 2 horas |
| 5 | 1,073M | ~500M | 5 días |
| 6 | 68,719M | ~5,000M | 320 días |

### Con Sampling (5M muestras):

| Piezas | Samples | Tiempo |
|--------|---------|--------|
| 3 | 350K (all) | 2 min |
| 4 | 5M | 10 min |
| 5 | 5M | 15 min |
| 6 | 5M | 20 min |

## Consumo de RAM

**Actual:**
- Acumula todas las posiciones en memoria
- KRRvK: ~24M posiciones × 71 dims × 4 bytes = 6.8 GB (estimado)
- **Problema:** Podría quedarse sin RAM

**Solución:** Guardar en chunks

```python
positions = []
chunk_size = 1_000_000

for ...:
    positions.append(...)
    
    if len(positions) >= chunk_size:
        # Guardar chunk
        save_chunk(positions)
        positions = []  # Liberar memoria
```

## Conclusión

**Para KRRvK actual:**
- Dejar que termine (~7 horas más)
- Monitorear RAM (si llega a 8GB, podría fallar)

**Para futuros datasets:**
- Implementar sampling para 5+ piezas
- Considerar multiprocessing para 4 piezas
- Guardar en chunks para evitar problemas de RAM

**Prioridad:**
1. ✅ Dejar terminar KRRvK
2. ⏭️ Entrenar KRRvK (validar 4 piezas)
3. ⏭️ Implementar sampling para KRvKP (más rápido)
4. ⏭️ Comparar exhaustivo vs sampling

---

**Estado actual:** KRRvK generando, 54% completo, ~7 horas restantes
