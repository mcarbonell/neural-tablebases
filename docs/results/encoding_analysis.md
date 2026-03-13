# Análisis del Encoding Actual y Soluciones Escalables

## Encoding Actual

### 1. Compact Encoding (Actual)
```python
# Para KQvK (3 piezas):
# Input: 192 dimensiones = 3 piezas × 64 casillas

Pieza 0 (Rey Blanco):  [0,1,0,0,...,0]  # 64 dims, one-hot
Pieza 1 (Dama Blanca): [1,0,0,0,...,0]  # 64 dims, one-hot  
Pieza 2 (Rey Negro):   [0,0,1,0,...,0]  # 64 dims, one-hot

Total: 192 dims
```

**Orden de piezas:** Blancas primero, luego negras. Dentro de cada color: K, Q, R, B, N, P

### 2. Full Encoding (Disponible pero no usado)
```python
# 768 dimensiones = 12 tipos de pieza × 64 casillas
# [WK, WQ, WR, WB, WN, WP, BK, BQ, BR, BB, BN, BP] × 64
```

### 3. ¿Qué falta?

❌ **NO hay encoding del turno** (side to move)
❌ **NO hay información geométrica**
❌ **NO hay información de tipo de pieza** (en compact)

## Problemas del Encoding Actual

### Problema 1: Sin información de turno
```python
# Estas dos posiciones son DIFERENTES pero tienen el mismo encoding:
Posición A: Rey blanco en e4, turno blanco
Posición B: Rey blanco en e4, turno negro
```

### Problema 2: Sin geometría
```python
# El modelo no puede calcular:
- Distancia entre piezas
- Adyacencia
- Líneas de ataque
- Control de casillas
```

### Problema 3: Pérdida de información en compact
```python
# En compact encoding, el modelo debe INFERIR qué pieza es cada una
# Solo sabe el orden: [Blancas, Negras] × [K, Q, R, B, N, P]
```

## Soluciones Escalables (Sin reglas específicas)

### Opción 1: Encoding Relativo (RECOMENDADO)

**Idea:** En lugar de posiciones absolutas, usar coordenadas relativas.

```python
# Para cada pieza, codificar:
1. Coordenadas absolutas normalizadas: (x/7, y/7)  # 2 dims
2. Tipo de pieza (one-hot): [K,Q,R,B,N,P]         # 6 dims
3. Color: [White, Black]                           # 2 dims

# Para cada PAR de piezas, añadir:
4. Distancia Manhattan normalizada                 # 1 dim
5. Distancia Chebyshev normalizada                 # 1 dim
6. Vector dirección normalizado (dx, dy)           # 2 dims

Total por pieza: 10 dims
Total pares (3 piezas): 3 pares × 4 dims = 12 dims
Total: 3×10 + 12 + 1 (turno) = 43 dims
```

**Ventajas:**
- ✅ Escala a cualquier endgame
- ✅ El modelo ve geometría directamente
- ✅ Mucho más compacto (43 vs 192 dims)
- ✅ Incluye turno

### Opción 2: Encoding Híbrido

```python
# Mantener one-hot + añadir features generales
One-hot: 192 dims (actual)
+ Coordenadas normalizadas por pieza: 6 dims
+ Distancias entre piezas: 3 dims
+ Turno: 1 dim
= Total: 202 dims
```

**Ventajas:**
- ✅ Fácil de implementar (solo añadir)
- ✅ Mantiene información completa
- ✅ Escala bien

### Opción 3: Encoding Tipo AlphaZero

```python
# Planos 2D (8×8) por cada tipo de información:
Plano 0: Rey blanco (8×8)
Plano 1: Dama blanca (8×8)
Plano 2: Rey negro (8×8)
Plano 3: Turno (8×8, todo 1s o 0s)
Plano 4: Distancia al centro (8×8, gradiente)
Plano 5: Distancia al borde (8×8, gradiente)

Total: 6 × 64 = 384 dims
```

**Ventajas:**
- ✅ Usado en AlphaZero con éxito
- ✅ Preserva estructura espacial
- ✅ Fácil añadir features geométricas

## Comparación de Opciones

| Opción | Dims | Geometría | Turno | Escalable | Complejidad |
|--------|------|-----------|-------|-----------|-------------|
| Actual | 192 | ❌ | ❌ | ⚠️ | Baja |
| Relativo | 43 | ✅ | ✅ | ✅ | Media |
| Híbrido | 202 | ✅ | ✅ | ✅ | Baja |
| AlphaZero | 384 | ✅ | ✅ | ✅ | Alta |

## Recomendación: Encoding Relativo

### Implementación:

```python
def encode_board_relative(board):
    """
    Encoding relativo escalable para cualquier endgame.
    """
    pieces = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            pieces.append((piece, square))
    
    # Sort: White first, then Black; K,Q,R,B,N,P
    pieces.sort(key=lambda x: (
        0 if x[0].color == chess.WHITE else 1,
        x[0].piece_type
    ))
    
    encoding = []
    
    # 1. Codificar cada pieza
    for piece, square in pieces:
        # Coordenadas normalizadas
        row, col = square // 8, square % 8
        encoding.extend([row/7.0, col/7.0])
        
        # Tipo de pieza (one-hot)
        piece_type = [0] * 6
        piece_type[piece.piece_type - 1] = 1
        encoding.extend(piece_type)
        
        # Color
        encoding.extend([1, 0] if piece.color == chess.WHITE else [0, 1])
    
    # 2. Codificar relaciones entre piezas
    for i in range(len(pieces)):
        for j in range(i+1, len(pieces)):
            sq1, sq2 = pieces[i][1], pieces[j][1]
            r1, c1 = sq1 // 8, sq1 % 8
            r2, c2 = sq2 // 8, sq2 % 8
            
            # Distancias normalizadas
            manhattan = (abs(r1-r2) + abs(c1-c2)) / 14.0
            chebyshev = max(abs(r1-r2), abs(c1-c2)) / 7.0
            
            # Vector dirección normalizado
            dx = (c2 - c1) / 7.0
            dy = (r2 - r1) / 7.0
            
            encoding.extend([manhattan, chebyshev, dx, dy])
    
    # 3. Turno
    encoding.append(1.0 if board.turn == chess.WHITE else 0.0)
    
    return np.array(encoding, dtype=np.float32)
```

### Por qué esto escala:

1. **No requiere reglas específicas:** Las distancias y relaciones son universales
2. **Funciona para cualquier endgame:** Solo cambia el número de piezas
3. **El modelo aprende las reglas:** A partir de las features geométricas
4. **Compacto:** 43 dims para 3 piezas, ~100 dims para 6 piezas

## Sobre el Turno

**CRÍTICO:** El turno debe estar en el encoding porque:

```python
# Misma posición, diferente resultado:
Turno blanco: Rey negro en jaque → Win
Turno negro: Rey negro puede mover → Draw/Loss
```

El encoding actual **NO incluye turno**, lo cual es un bug importante.

## Conclusión

Para escalar sin añadir reglas específicas:

1. **Implementar encoding relativo** (Opción 1)
2. **Añadir turno al encoding**
3. **Dejar que el modelo aprenda geometría** de las features

Esto debería:
- Mejorar accuracy de 68% a >90%
- Funcionar para cualquier endgame
- No requerir reglas específicas
