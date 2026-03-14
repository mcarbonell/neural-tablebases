# Mejoras propuestas (siguiente sesión)

**Fecha:** 14 de marzo de 2026  
**Contexto:** ideas para escalar el proyecto tras V4 + canónicas pawn-safe.

## 1) Escalado de datos (5+ piezas)

- **Entrenamiento “streaming” por shards**: evitar combinar todo en un único `.npz` gigante (RAM). Mantener los chunks/shards y entrenar leyendo por lotes.
- **Formato recomendado**:
  - Opción A: `data/ENDGAME/shards/*.npz` (cada shard: `x`, `wdl`, `dtz`, `metadata`).
  - Opción B: convertir a `.npy` + `np.memmap` para lecturas secuenciales rápidas (menos overhead que `.npz`).
- **Split reproducible**: split train/val por hash determinista (p.ej. hash de `canonical_key` o del índice de shard+offset), para que no dependa del orden de lectura.

## 2) “Exception maps” end-to-end (objetivo final del proyecto)

- **Pipeline estándar**: `train → verify/search → recolectar errores → guardar mapa de excepciones → medir tamaño final + 100%`.
- **Representación**:
  - Clave: algo estable y compacto (p.ej. `canonical_key` serializado o encoding cuantizado).
  - Valor: WDL/DTZ verdadero (y opcionalmente “best move” si se quiere jugar).
- **Métrica principal**: tamaño final = `modelo (INT8/FP16) + exception-map comprimido` frente a Syzygy, con exactitud/garantía.

## 3) Reproducibilidad fuerte (metadatos)

- Guardar en metadata:
  - `git_commit` (hash), parámetros completos, `shuffle_seed`/`item_offset`, `chunk_size`.
  - versión de Python/Torch/python-chess/numpy.
  - seed de entrenamiento y seed del split.
- Validar que `search_poc.py` aplica exactamente la misma normalización/canonización esperada por el modelo (ya hay metadata, falta estandarizarlo).

## 4) Tests + CI (mínimo viable)

- **Smoke tests**:
  - Generación parcial con peones usando `--shuffle-seed` (evita prefijos sesgados).
  - Canónicas en finales con peones: asegurar que `auto` usa `file_mirror`.
  - Verificar que se escriben `*_metadata.json` y que el nombre `_partial_...` no pisa datasets completos.
- **GitHub Actions**:
  - `python -m py_compile` + un smoke-test rápido (sin Syzygy si no está en CI, o con un “stub”).

## 5) Ergonomía de CLI / defaults

- `train.py`: `--hard_mining` hoy está activado por defecto; añadir un flag claro para desactivarlo (`--no-hard_mining`) o invertir el default.
- Añadir `--seed` para entrenamiento (y para el split) y volcarlo a metadata.
- Unificar nombres: `--data`/`--output`/`--name` para que sea difícil entrenar sin querer sobre un dataset parcial.

## 6) Benchmarks de evaluación

- Reportar **accuracy por clase** (W/D/L), matriz de confusión, y calibración (probabilidades).
- DTZ: además de MAE, percentiles (P50/P90/P99) y error condicionado por WDL.
- Medir **velocidad de inferencia** (ms/posición) y el coste incremental del search depth (0/1/2).

## 7) Export/compresión del modelo (cuando el pipeline esté estable)

- Export ONNX/TorchScript y cuantización (INT8) + comparación de accuracy.
- QAT (quantization-aware training) si la cuantización post-training degrada demasiado.

