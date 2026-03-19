# 🚀 TODO: COMPLETAR 4/5-PIEZAS + BUGFIX

## ✅ STATUS ACTUAL
```
KPvK: 100.00% (V5 + Search D1) ✅ (NEW)
KPvKP: 81% (v4) -> Goal: 99%+ with V5
KRPvKP: 94.1% (5p, epoch 8+) ✓
Pendientes: Migrar todo el pipeline a Encoding V5 (King-first)
GPU (AMD 780M): DirectML active (1.1s/epoch) ✅
```

## 🎯 IMMEDIATE (HOY - 2h)
```
[ ] 1. GENERATE KPvKP V5 (Parallel)
    python src/generate_datasets_parallel.py --config KPvKP --relative --version 5 --canonical --data data/v5
[ ] 2. TRAIN KPvKP V5
    python src/train.py --data_path data/v5/KPvKP_canonical.npz --epochs 200 --model_name mlp_kpvkp_v5
[ ] 3. VERIFY SEARCH (Target 100%)
    python src/verify_search_correction.py --onnx data/mlp_kpvkp_v5.onnx

```

## 📦 4-PIEZAS (ESTA SEMANA - 1.25h cada uno)
```
[x] KPvKP ✓
[ ] KRvKP: generate + train canonical 200e
[ ] KBPvK: generate + train canonical 200e
[ ] KRvKN: generate + train canonical 200e
[ ] KQvKQ: generate + train canonical 200e
```

**Template comando**:
```bash
python src/generate_datasets_parallel.py --config ENDGAME --relative --version 4 --canonical --canonical-mode auto
python src/train.py --data_path data/ENDGAME_canonical.npz --epochs 200 --batch_size 512 --model_name mlp_ENDGAME_v4_canonical
```

## 🏗️ 5-PIEZAS (PRÓXIMO MES - sampled 20%)
```
[ ] KRRvKP: regenerate canonical + train
[ ] KNNvK: generate sampled + train
[ ] KBBvKN: generate sampled + train
[ ] etc... (top 10 por frecuencia self-play)
```
**Nota**: Usar `--limit-items 20%` para datasets >10M

## 📊 VALIDACIÓN FINAL
```
[ ] Benchmark suite: scripts/complete_4p5p_benchmark.py
[ ] ELO self-play vs Syzygy baseline
[ ] Update docs/results/4p5p_summary.md
[ ] Paper ICGA: incluir tabla completa
```

## 🎓 HIPERPARAMS ESTÁNDAR
```
epochs: 200
batch_size: 512
lr: 0.001
model: mlp
hard_mining: true (epochs 50+)
```

## 📈 METAS
```
4-piezas avg: 98% → 99.8% (search depth=2)
5-piezas avg: 94% → 99.5%
Compresión total: <1MB todas 4-piezas
ICGA paper: SUBMITIR
```

