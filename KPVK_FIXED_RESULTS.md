# KPvK Corregido - Resultados

## Problema Encontrado y Solucionado

### Bug en el Encoding Relativo

**Problema:** El vector de tipo de pieza estaba mal ordenado.

```python
# ANTES (INCORRECTO):
piece_type[piece.piece_type - 1] = 1.0
# Orden resultante: [P, N, B, R, Q, K]
# KING (type=6) → índice 5 (posición de P) ✗

# DESPUÉS (CORRECTO):
type_to_idx = {
    chess.KING: 0,    # K en posición 0
    chess.QUEEN: 1,   # Q en posición 1
    chess.ROOK: 2,    # R en posición 2
    chess.BISHOP: 3,  # B en posición 3
    chess.KNIGHT: 4,  # N en posición 4
    chess.PAWN: 5     # P en posición 5
}
# Orden: [K, Q, R, B, N, P] ✓
```

### Impacto del Bug

**Datasets anteriores (KQvK, KRvK):**
- Tenían el encoding incorrecto
- Pero el modelo aprendió de todas formas (99.92%, 99.99%)
- El modelo es robusto al orden de las features

**Dataset KPvK anterior:**
- Encoding incorrecto: Rey → Peón
- Generó KPvKP en lugar de KPvK
- Aún así alcanzó 99.75% accuracy

## Dataset KPvK Correcto

### Verificación:

```
Total posiciones: 331,352
Encoding: Relativo (43 dims)

Piezas por posición:
  - Rey blanco (WK): 1
  - Peón blanco (WP): 1
  - Rey negro (BK): 1

Total peones: 331,352 (1.00 por posición) ✓

Distribución de ranks de peones:
  Rank 1: 55,232 (16.7%)
  Rank 2: 55,232 (16.7%)
  Rank 3: 55,232 (16.7%)
  Rank 4: 55,232 (16.7%)
  Rank 5: 55,232 (16.7%)
  Rank 6: 55,192 (16.7%)
  
✓ Todos los peones en ranks válidos (1-6)
✗ NO hay peones en rank 0 o 7
```

### Distribución WDL:

```
Loss  (-2): 97,604  (29.46%)
Draw  ( 0): 108,788 (32.83%)
Win   ( 2): 124,960 (37.71%)
```

**Más empates que KQvK/KRvK:**
- KQvK: 6.26% empates
- KRvK: 5.57% empates
- KPvK: 32.83% empates ← Rey puede bloquear el peón

## Resultados del Entrenamiento

### Progreso por Época:

| Época | Train Acc | Val Acc | Hard Examples | Mejora |
|-------|-----------|---------|---------------|--------|
| 1 | 93.16% | **96.59%** | 2,277 | - |
| 2 | 96.23% | 97.72% | 636 | +1.13% |
| 3 | 96.96% | 98.12% | 534 | +0.40% |
| 4 | 97.41% | 98.44% | 441 | +0.32% |
| 5 | 97.75% | 98.76% | 374 | +0.32% |
| 6 | 97.97% | 98.93% | 315 | +0.17% |
| 7 | 98.18% | 98.84% | 262 | -0.09% |
| 8 | 98.42% | **99.22%** | 205 | +0.38% |
| 9 | 98.54% | 99.30% | 244 | +0.08% |
| 10 | 98.71% | **99.43%** | 183 | +0.13% |

**Mejor resultado hasta ahora:** 99.43% accuracy (época 10)

### Comparación con Otros Endgames:

| Endgame | Época 1 | Época 10 | Mejor | Complejidad |
|---------|---------|----------|-------|-------------|
| KQvK | 98.07% | 99.77% | 99.92% | Simple |
| KRvK | 99.68% | 99.98% | 99.99% | Simple |
| KPvK | 96.59% | 99.43% | 99.43%+ | Media |

**KPvK es más difícil porque:**
1. Más empates (32.83% vs 6%)
2. Promoción del peón añade complejidad
3. Bloqueo del rey es crítico

## El "Truco" del Peón

### Características Especiales:

**1. Promoción:**
```
Peón en rank 6 → 1 movimiento para promocionar
Peón en rank 2 → 4 movimientos para promocionar
Coordenada Y es crítica
```

**2. Movimiento Asimétrico:**
```
Solo avanza hacia adelante
No puede retroceder
Dirección importa
```

**3. Bloqueo:**
```
Rey puede bloquear el peón
Distancia rey-peón es crítica
Chebyshev distance captura esto
```

### Cómo el Encoding Relativo lo Maneja:

```python
# Features del peón:
row = 0.857  # Rank 6 (cerca de promoción)
col = 0.500  # File e

# Distancia a promoción:
dist_to_rank_8 = 1.0 - row  # 0.143 (1 casilla)

# Distancia al rey defensor:
chebyshev_dist(pawn, black_king) = 0.286  # 2 casillas

# El modelo aprende automáticamente:
# Si dist_to_rank_8 < 0.2 AND dist_to_king > 0.14 → Win
# Si dist_to_king <= 0.14 → Draw/Loss (rey bloquea)
```

## Estadísticas DTZ

**DTZ (Distance to Zeroing move):** Movimientos hasta el próximo movimiento irreversible.

```
Loss positions: DTZ range [-20, -2], mean=-2.62
Draw positions: DTZ = 0 (siempre)
Win positions:  DTZ range [1, 19], mean=1.56
```

**Interpretación:**
- Posiciones ganadoras: ~1.5 movimientos hasta avanzar el peón
- Posiciones perdedoras: ~2.6 movimientos hasta que el peón avance (y gane)
- Empates: DTZ = 0 (no hay progreso posible)

## Conclusiones

### ✅ Éxitos:

1. **Bug corregido:** Encoding relativo ahora usa orden correcto [K,Q,R,B,N,P]
2. **Dataset correcto:** KPvK real (no KPvKP)
3. **Peones válidos:** Solo en ranks 1-6
4. **Alta accuracy:** 99.43% en época 10
5. **Maneja complejidad:** Promoción, bloqueo, asimetría

### 📊 Comparación:

| Métrica | KQvK | KRvK | KPvK |
|---------|------|------|------|
| Época 1 | 98.07% | 99.68% | 96.59% |
| Época 10 | 99.77% | 99.98% | 99.43% |
| Empates | 6.26% | 5.57% | 32.83% |
| Dificultad | Baja | Baja | Media |

### 🎯 Próximos Pasos:

1. ✅ KPvK completado (99.43%+)
2. ⏭️ Regenerar KQvK y KRvK con encoding correcto
3. ⏭️ Pasar a endgames de 4 piezas
4. ⏭️ Considerar entrenar DTZ (futuro)

---

**Fecha:** 2026-03-12
**Estado:** KPvK corregido y entrenando
**Accuracy:** 99.43% (época 10, aún mejorando)
**Bug:** Corregido (orden de piece type)
