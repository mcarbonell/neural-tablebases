# Vanguard (ChessGnnV8) Architecture & Technical Specification

Este documento define la arquitectura exacta, la estructura de tensores, y las rutinas de evaluación de la red **Vanguard v8**. Su propósito es servir como la *Single Source of Truth* (SSOT) para poder reconstruir el modelo desde cero o interactuar con el checkpoint `v8_universal_35M_latest.pth` sin errores de incompatibilidad.

## 1. Topología del Modelo
El modelo utiliza una arquitectura de GNN Relacional (*Relational Graph Neural Network*), orientada a procesar el tablero de ajedrez como un grafo conexionista de 64 nodos donde las aristas son las simetrías y métricas tácticas (ataques, defensas, etc.).

* **Parámetros Totales**: ~1.19 Millones.
* **Checkpoint de Referencia**: `v8_universal_35M_latest.pth`

### Capas y Dimensiones
1. **Inputs y Embeddings**:
   - `Piece Embedding`: Matriz de 14 Tokens $\to$ 32 dimensiones.
   - `Coord Projection`: Lineal de 2 dimensiones (Rango, Columna) $\to$ 16 dimensiones.
2. **Proyección Inicial del Nodo** (`node_proj`):
   - El vector de entrada base por cada nodo concatena: `[Piece (32), Tac (4), Coord (16)] = 52 dimensiones`.
   - Proyección lineal: Entrada (52) $\to$ `Node Dim` (128) $\to$ Activación GELU.
3. **Bloques GNN (`gnn_blocks`)**:
   - Bucle de 4 instancias con la iteración `VanguardGNNLayer`.
   - **Dimensiones por capa**: 128 (Entrada) $\to$ 128 (Salida).
   - **Mecanismo Relacional**: Las activaciones son la suma indexada multivariada para las 16 aristas tácticas relacionales diferentes.
   - **Normalización**: *Post-Norm* usando `nn.LayerNorm` que se suma en la conexión residual (`h = h + LayerNorm(h_relasional) == GELU`).
4. **Global Attention Pooling**:
   - Para compilar el grafo entero (64 nodos de 128 dims) en una red flat 1D:
   - Se procesa un bloque agnóstico en PyTorch: `Cat[ Mean(64 nodos), Max(64 nodos) ]` sobre las 128 features $\to$ **Vector Global de 256 dimensiones reales**.
5. **Output Heads (Dual Output / Multi-Task)**:
   - `WDL Head`: Multi-Layer Perceptron (256 $\to$ 128 $\to$ GELU $\to$ 3 labels categóricos WDL).
   - `DTZ Head`: Multi-Layer Perceptron (256 $\to$ 128 $\to$ GELU $\to$ 1 output discreto DTZ/Pawn-Progress).

---

## 2. Definición Exacta de las Entradas (Input Pipeline)

**Alerta Histórica:** Cualquier mínima variación en el escalado, *shifting* o el orden vectorial en esta etapa destrozará el rendimiento (hundiendo la validez al 42-45%, el equivalente a estancarse en salida plana de 'Win'). Todo tensor es escupido nativamente por `src/search/rust_engine.py` a través de la API `engine.get_raw_features(fen)`.

### A. Los IDs de las Piezas (`p_ids`)
El FFI del motor de Rust escupe el tensor unidimensional de 64 índices con la siguiente rúbrica natural:
- `0`: Casilla Vacía
- `1` al `6`: Piezas Blancas (*Peón, Caballo, Alfil, Torre, Dama, Rey*)
- `7` al `12`: Piezas Negras (*Peón, Caballo, Alfil, Torre, Dama, Rey*)

> [!WARNING]
> Históricamente se intentó meter un "*Rosetta-stone Remapping*" añadiendo `+1` a estos valores o normalizando el Empty a "1". **Debe suprimirse esa operación**. El `Embedding(14, 32)` de la red de 35 Millones fue validado y entrenado **sin ningún desplazamiento**, pasando directamente las IDs raw de Rust (0 to 12).

### B. Coordenadas Espaciales (`coords`)
La topología Vanguard carece de autolización gráfica-visual así que inyecta posiciones matriciales:
- **Files y Ranks (x,y)**: Tensores del `0` al `7` apilados repetidamente en 64 casillas.
- **Normalización**: Escalan en el rango **`[-1.0, 0.0]`** (es decir, dividir por `7.0` y restarle `-1.0`). La red fue preentrenada bajo este desvío y fallará gravemente si operan en `[0,1]`.

### C. Características Tácticas (`node_tac`)
Cada matriz escupe métricas relativas calculadas por el generador pseudo-legal de Raw-Masks (*White Attacks*, *Black Attacks*, *Flags*, y *Attacker Bitmasks*):
- **Acondicionamiento**: Estos valores discretos (incluidos los flags y sumas de bitmasks) se aplanan, se convierten en tensores de *float32* y se escalan siempre estáticamente dividiendo por **`8.0`**. 

### D. Orden Lineal de Concatenación (El Secreto)
La red de 35 Millones en su primera capa ensambla el input basándose exclusivamente en un único mapa de concat. El orden indexado por los pesos (y validado con los offsets de varianza empírica) es:
1. **`h_piece`** (`32` dims)
2. **`t_in`** (`Tac` en `4` dims)
3. **`h_coord`** (`16` dims proyectados)

Vector Correcto: `torch.cat([h_piece, tac_node, h_coord], dim=-1)`

---

## 3. Perspectiva de Evaluación (Absolute Perspective)

A diferencia de las tablas Syzygy tradicionales (incluso tablebases V1, cuyas respuestas son relativas invariablemente al turno actual o "*Side-to-move*"), el modelo Vanguard V8 se entrenó en una **Perspectiva Absoluta enfocada en Blancas** (Operando como las últimas generaciones de `Stockfish NNUE` y `AlphaZero`).

Las clases probabilísticas de WDL devueltas por el clasificador categórico:
- `0` = Ganan Negras (Equivalent to Loss para Blancas)
- `1` = Tablas (Draw)
- `2` = Ganan Blancas (Equivalent to Win para Blancas)

> [!IMPORTANT]
> **El Inversor Estocástico (Flip Perspective)**: Para testear correctamente la red contra datos empíricos de Chess.Syzygy o para enchufar este Evaluador a un motor *Alpha-Beta / Minimax*, se requiere aplicar condicionalmente un *inversor de puntaje* siempre que evalúes posiciones con **turno de Negras** (`chess.BLACK`). Si la red predice que ganan las blancas (2), pero mueven negras, debes reportarle al motor una asimetría final de *Loss* (0).
> Código requerido: `eval_side_to_move = 2 - model_predict`

## 4. Archivos Clave del Repositorio

Para un mantenimiento rápido, este tridente de scripts cubre el end-to-end (y deben estar enlazados tal cual se testearon en Marzo de 2026):

- `src/search/rust_engine.py`: Encargado de parsear bits desde el FFI nativo y enviar los tensores `int8`.
- `src/model/v8_vanguard.py`: El core del Runtime de PyTorch y Single Truth of Tensor-Shapes. 
- `scripts/benchmark_neural_vs_sf.py`: Archivo oficial de pruebas. Realiza en paralelo las proyecciones `get_eval()` de Stockfish con las de Vanguard.
