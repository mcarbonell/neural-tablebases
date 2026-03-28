# Arquitectura GNN: Motor de Búsqueda Neuronal (Neural Search) - V8-Pro Phase

> Última actualización: 28 de marzo, 2026

## 📌 Concepto Revolucionario
Apartarse de las redes MLP estándar para construir una **Graph Neural Network (GNN)** cuya estructura física sea idéntica a las reglas del ajedrez. En esta arquitectura, la red no "imagina" el juego, sino que lo **simula** a través de la propagación de señales tácticas entre nodos (casillas).

---

## 🏗️ Topología Implementada (V8-Pro)

### Nodo: Embedding Dimensional (128D)
Cada casilla se representa como un vector latente que aprende su importancia táctica.

### Aristas: Relaciones Tácticas (16 tipos)
En lugar de vectores de calor simples, la V8-Pro usa 16 tipos de relaciones inyectadas por el motor de **Rust MoveGen**:
1.  **Direct Attacks**: Ataque directo a pieza enemiga.
2.  **Defenses**: Protección de pieza propia.
3.  **Pins (Clavadas)**: Inmovilización táctica de pieza enemiga.
4.  **X-Rays**: Influencia a través de otras piezas.
5.  **Checks (Jaque)**: Amenaza directa al Rey.

---

## 👁️ Global Attention Pooling (Feedback Reasoning)
La V8-Pro ha implementado el **Razonamiento Selectivo**. En lugar de promediar el tablero, usa una capa de atención global para decidir qué cuadrante o pieza es el centro de gravedad de la posición:
*   **Filtro Estratégico**: Ignora regiones inertes.
*   **Convergencia Tactics**: Concentra el 80% de la "capacidad cerebral" en el foco de la lucha.

---

## 🔱 Triple Head Evaluation (Universal Output)
La arquitectura ahora entrega tres señales síncronas:
1.  **WDL Classification**: Resultado binario/terciario (G/T/P).
2.  **DTZ Regression**: Profundidad técnica hasta la conversión (Syzygy).
3.  **Eval Stockfish-Distillation**: Intuición táctica en centipeones tanh([-1, 1]).

---

## 🚀 Estado de Implementación: GNN-Inference en Negamax

### ✅ Completado (28-03-2026)

**`src/searcher_v8.py`** — `GnnSearcher` operacional:
- Alpha-Beta Minimax sobre `ChessGnnV8_Pro` como función de evaluación de hojas.
- Caché de evaluación por FEN para evitar llamadas GNN redundantes dentro de un árbol.
- Perspectiva correcta: score en perspectiva de Blancas para el Minimax.
- CLI de benchmark: compara WDL-Acc a depth=0/1/2 sobre configs de endgame.

```bash
# Benchmark de validación del paper
python src/searcher_v8.py \
    --model data/v8_pro_triple_head_best.pth \
    --syzygy syzygy \
    --configs KRvK,KQvK,KPvK,KRvKP \
    --samples 200 --depths 0,1,2 \
    --output data/logs/gnn_search_benchmark.json
```

**`src/export_onnx_v8.py`** — Export ONNX + benchmark de latencia:
- Exporta a ONNX con batch dinámico y edge count dinámico.
- Mide latencia end-to-end: Rust extraction + GNN forward (target: < 10ms).

```bash
python src/export_onnx_v8.py \
    --model data/v8_pro_triple_head_best.pth \
    --output data/v8_pro_triple_head.onnx \
    --benchmark --n_bench 200
```

### ⏳ Pendiente

*   **Resultado experimental**: Aún falta ejecutar el benchmark con un modelo converged (target: depth=1 → >99% WDL-Acc). Requiere finalizar el Epoch 1 del entrenamiento en Lichess.
*   **Pruning Topológico**: Usar los attention weights de `GlobalAttentionPooling` por posición como "move priority score" para ordenar los hijos del árbol antes del alpha-beta. Esto crearía una sinergia GNN-búsqueda única en la literatura.
*   **ONNX Runtime Integration**: Sustituir el forward pass de PyTorch por `ort.InferenceSession` para inferencia más rápida en CPU durante la búsqueda en tiempo real.

---

## 📈 Resumen
Esta arquitectura fusiona lo mejor de dos mundos: la **precisión matemática** del generador de movimientos (movegen) con la **intuición estadística** de una red GNN de última generación. El GNN-Search es el puente entre ambos dominios y el experimento validador del paper.
