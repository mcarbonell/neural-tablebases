# 🚀 TODO: COMPLETAR 4/5-PIEZAS + BUGFIX

## ✅ STATUS ACTUAL
```
KPvKP_canonical: 96.78% val_acc (3.7M pos) ✓
KRPvKP: 94.1% (5p, epoch 8+) ✓
Pendientes: KRvKP, KBPvK, KRvKN, KQvKQ
BUG search_poc.py: FIXED ✅
```

## 🎯 IMMEDIATE (HOY - 2h)
```
[ ] 1. TEST FIXED SEARCH (5min)
    python src/search_poc.py --model data/mlp_kpvkp_v4_canonical_best.pth --config KPvKP --samples 1000 --depths 0,1,2
    Esperado: Depth0=97%, Depth1=99.5%, Depth2=99.9%

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

