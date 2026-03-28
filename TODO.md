# 🚀 TODO — V8-Pro Triple Head Phase

Última actualización: 28 de marzo, 2026

---

## ✅ COMPLETADO

```
[x] V8-Pro Architecture (RGNN, 1.17M params, 4 GNN layers)
[x] Global Attention Pooling — razonamiento selectivo por casilla
[x] Triple Head — WDL + DTZ + Stockfish-Eval simultáneos
[x] Lichess Sharding — 266 shards (~100M posiciones, 1.8 GB)
[x] Rust MoveGen — 16 tipos de arista táctica, 100% WAC precision
[x] DirectML AMD — Radeon 780M iGPU, ~693 pos/s en régimen
[x] LayerNorm + Dropout + Cosine LR Schedule
[x] Best checkpoint tracker (_best.pth) — Añadido 2026-03-28
[x] build_giant_graph vectorizado (sin bucle Python) — 2026-03-28
```

---

## 🎯 SPRINT INMEDIATO

```
[ ] eval_v8_tablebase.py — Validar modelo contra Syzygy después de training
    python src/eval_v8_tablebase.py --model data/v8_pro_triple_head_best.pth \
        --syzygy syzygy --configs KRvK,KQvK,KPvK,KRvKP --samples 500

[ ] searcher_v8.py — Portar GNN-Search a V8 (hipótesis central del paper)
    Target: WDL-Acc depth=1 > 99%

[ ] ONNX export V8 — Latencia < 10ms por posición en CPU
    Adaptar src/export_onnx.py a la interfaz de grafos V8
```

---

## 📦 ESTA SEMANA

```
[ ] Tests V8 en run_tests.py / tests/test_v8_pipeline.py
    - test_rust_engine_loads()
    - test_gnn_forward_pass()
    - test_gnn_shard_dataset()

[ ] Pre-computar grafos COO en shards (eliminar decode en DataLoader)

[ ] Fine-tuning: freeze GNN layers → re-entrenar cabezas WDL+DTZ sobre Syzygy
    (después de que el pre-train en Lichess converja ≥80%)
```

---

## 🏗️ PRÓXIMO MES

```
[ ] Paper ICGA — tabla V1→V8 accuracy progression + ablation study
[ ] Pruning topológico Minimax — usar attention weights como move priority
[ ] Motor UCI mínimo — para demos y benchmarks públicos
```

---

## 📊 METAS

```
V8-Pro WDL-Acc (Lichess): 79% → 85%+
V8-Pro WDL-Acc (Syzygy 4-piezas): 99.83% con search depth=1
ONNX latencia: < 10ms por posición en CPU
Paper ICGA: SUBMIT
```

---

## 🎓 HIPERPARAMS V8-PRO ESTÁNDAR

```
node_dim: 128
num_layers: 4
dropout: 0.1
batch_size: 1024
lr: 0.001 (Cosine Annealing → 0.0001)
optimizer: AdamW (weight_decay=1e-4)
```
