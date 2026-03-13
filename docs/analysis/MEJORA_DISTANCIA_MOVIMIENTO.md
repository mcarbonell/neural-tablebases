# Mejora: Distancia de Movimiento Real por Pieza

## Problema Actual

El encoding relativo usa:
- **Manhattan distance:** |dx| + |dy|
- **Chebyshev distance:** max(|dx|, |dy|)

Pero **no captura cómo se mueve cada pieza específicamente**.

## Propuesta: Añadir "Move Distance"

Para cada par de piezas, calcular: **¿Cuántos movimientos necesita la pieza para llegar?**

### Fórmulas por Tipo de Pieza:

```python
def piece_move_distance(piece_type, from_sq, to_sq):
    """
    Calcula movimientos mínimos para que una pieza llegue a otra casilla.
    """
    dx = abs(file(to_sq) - file(from_sq))
    dy = abs(rank(to_sq) - rank(from_sq))
    
    if piece_type == KING:
        # Rey: Chebyshev distance
        return max(dx, dy)
    
    elif piece_type == QUEEN:
        # Dama: 1 si diagonal/línea recta, 2 si no
        if dx == 0 or dy == 0 or dx == dy:
            return 1
        else:
            return 2
    
    elif piece_type == ROOK:
        # Torre: 1 si misma fila/columna, 2 si no
        if dx == 0 or dy == 0:
            return 1
        else:
            return 2
    
    elif piece_type == BISHOP:
        # Alfil: 1 si diagonal, infinito si casilla de color diferente
        if (dx + dy) % 2 != 0:
            return float('inf')  # Imposible (color diferente)
        elif dx == dy:
            return 1
        else:
            return 2
    
    elif piece_type == KNIGHT:
        # Caballo: Más complejo, usar tabla precalculada
        return knight_distance_table[from_sq][to_sq]
    
    elif piece_type == PAWN:
        # Peón: Solo avanza hacia adelante
        if dx == 0 and dy > 0:  # Mismo file, avanza
            return dy
        else:
            return float('inf')  # No puede llegar
```

### Tabla de Distancias de Caballo

```python
# Precalcular distancias de caballo (BFS)
def compute_knight_distances():
    """
    Calcula distancia mínima de caballo entre todas las casillas.
    """
    distances = np.zeros((64, 64), dtype=int)
    
    for from_sq in range(64):
        # BFS desde from_sq
        queue = [(from_sq, 0)]
        visited = {from_sq}
        
        while queue:
            sq, dist = queue.pop(0)
            distances[from_sq][sq] = dist
            
            # Generar movimientos de caballo
            for move in knight_moves(sq):
                if move not in visited:
                    visited.add(move)
                    queue.append((move, dist + 1))
    
    return distances

# Ejemplos de distancias de caballo:
# a1 → b3: 1 movimiento
# a1 → c2: 1 movimiento
# a1 → a2: 3 movimientos (a1→c2→b4→a2)
# a1 → h8: 6 movimientos
```

## Encoding Mejorado

### Actual (4 dims por par):
```python
- Manhattan distance: 1 dim
- Chebyshev distance: 1 dim
- Direction (dx, dy): 2 dims
```

### Propuesto (5 dims por par):
```python
- Manhattan distance: 1 dim
- Chebyshev distance: 1 dim
- Move distance (piece-specific): 1 dim  ← NUEVO
- Direction (dx, dy): 2 dims
```

### Impacto en Dimensiones:

| Piezas | Actual | Mejorado | Incremento |
|--------|--------|----------|------------|
| 3 | 43 | 46 | +7% |
| 4 | 65 | 71 | +9% |
| 5 | 91 | 101 | +11% |

**Fórmula:** `n×10 + (n×(n-1)/2)×5 + 1`

## Ventajas

### 1. Captura Movimiento Real

```python
# Torre en a1, peón en h8
Manhattan: 14
Chebyshev: 7
Move distance: 2  ← Torre necesita 2 movimientos

# El modelo aprende:
# "Torre a 2 movimientos del peón → Puede capturar"
```

### 2. Diferencia Entre Piezas

```python
# Alfil vs Torre a misma distancia
Alfil en a1 → h8:
  Manhattan: 14
  Chebyshev: 7
  Move distance: 1  ← Diagonal directa

Torre en a1 → h8:
  Manhattan: 14
  Chebyshev: 7
  Move distance: 2  ← Necesita esquina

# El modelo aprende que alfil es más rápido en diagonales
```

### 3. Casos Especiales

```python
# Alfil en casilla de color diferente
Alfil blanco en a1 (negra) → h7 (negra):
  Move distance: 1

Alfil blanco en a1 (negra) → h8 (blanca):
  Move distance: inf  ← Imposible

# Peón solo avanza
Peón en e4 → e5:
  Move distance: 1

Peón en e4 → e3:
  Move distance: inf  ← No puede retroceder
```

## Ejemplo: KRRvK

### Posición Crítica:

```
♜ . . . . . . .  (Torre blanca en a8)
. . . . . . . .
. . . . . . . .
. . . ♚ . . . .  (Rey negro en d5)
. . . . . . . .
. . . . . . . .
. . . . . . . .
♖ . . ♔ . . . .  (Torre blanca en a1, Rey blanco en d1)
```

**Encoding actual:**
```python
Torre a8 ↔ Rey negro d5:
  Manhattan: 3 + 3 = 6
  Chebyshev: max(3, 3) = 3
  
Torre a1 ↔ Rey negro d5:
  Manhattan: 3 + 4 = 7
  Chebyshev: max(3, 4) = 4
```

**Encoding mejorado:**
```python
Torre a8 ↔ Rey negro d5:
  Manhattan: 6
  Chebyshev: 3
  Move distance: 2  ← a8→d8→d5 o a8→a5→d5
  
Torre a1 ↔ Rey negro d5:
  Manhattan: 7
  Chebyshev: 4
  Move distance: 2  ← a1→d1→d5 o a1→a5→d5
```

**El modelo aprende:** "Ambas torres a 2 movimientos → Coordinación perfecta"

## Implementación

```python
def encode_board_relative_v2(board: chess.Board) -> np.ndarray:
    """
    Encoding relativo mejorado con distancia de movimiento real.
    """
    pieces_on_board = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            pieces_on_board.append((piece, square))
    
    pieces_on_board.sort(key=lambda x: (
        0 if x[0].color == chess.WHITE else 1,
        x[0].piece_type
    ))
    
    encoding = []
    
    # 1. Per-piece features (sin cambios)
    for piece, square in pieces_on_board:
        row = chess.square_rank(square) / 7.0
        col = chess.square_file(square) / 7.0
        encoding.extend([row, col])
        
        piece_type_vec = [0.0] * 6
        type_to_idx = {
            chess.KING: 0, chess.QUEEN: 1, chess.ROOK: 2,
            chess.BISHOP: 3, chess.KNIGHT: 4, chess.PAWN: 5
        }
        piece_type_vec[type_to_idx[piece.piece_type]] = 1.0
        encoding.extend(piece_type_vec)
        
        color = [1.0, 0.0] if piece.color == chess.WHITE else [0.0, 1.0]
        encoding.extend(color)
    
    # 2. Pairwise features (MEJORADO)
    num_pieces = len(pieces_on_board)
    for i in range(num_pieces):
        for j in range(i + 1, num_pieces):
            piece1, sq1 = pieces_on_board[i]
            piece2, sq2 = pieces_on_board[j]
            
            r1 = chess.square_rank(sq1)
            c1 = chess.square_file(sq1)
            r2 = chess.square_rank(sq2)
            c2 = chess.square_file(sq2)
            
            # Manhattan distance
            manhattan = (abs(r1 - r2) + abs(c1 - c2)) / 14.0
            
            # Chebyshev distance
            chebyshev = max(abs(r1 - r2), abs(c1 - c2)) / 7.0
            
            # Move distance (piece-specific) ← NUEVO
            move_dist = piece_move_distance(piece1.piece_type, sq1, sq2)
            move_dist_normalized = min(move_dist, 7) / 7.0  # Cap at 7
            
            # Direction vector
            dx = (c2 - c1) / 7.0
            dy = (r2 - r1) / 7.0
            
            encoding.extend([manhattan, chebyshev, move_dist_normalized, dx, dy])
    
    # 3. Global features (sin cambios)
    encoding.append(1.0 if board.turn == chess.WHITE else 0.0)
    
    return np.array(encoding, dtype=np.float32)
```

## Accuracy Esperado

Con esta mejora:

| Endgame | Actual | Mejorado | Mejora |
|---------|--------|----------|--------|
| KQvK | 99.92% | 99.95%+ | +0.03% |
| KRvK | 99.99% | 99.99%+ | +0.00% |
| KPvK | 99.89% | 99.92%+ | +0.03% |
| KRRvK | 99.9%? | 99.95%+ | +0.05% |
| KRvKP | 99.5%? | 99.7%+ | +0.2% |

**Mayor impacto en endgames tácticos** donde el timing de movimientos es crítico.

## Desventajas

1. **Más dimensiones:** +7-11% (pero sigue siendo compacto)
2. **Más cómputo:** Calcular distancias de movimiento
3. **Tabla de caballo:** Necesita precalcular 64×64 = 4KB

## Conclusión

**Recomendación:** Implementar como "encoding v2" y comparar:

1. Entrenar KRRvK con encoding actual (65 dims)
2. Entrenar KRRvK con encoding mejorado (71 dims)
3. Comparar accuracy y convergencia

**Predicción:** Mejora de 0.1-0.3% en accuracy, especialmente en endgames tácticos como KRvKP.

---

**¿Implementamos esto?** Podría ser una mejora significativa para el paper. 🎯
