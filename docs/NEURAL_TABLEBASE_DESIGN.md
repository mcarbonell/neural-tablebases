# Neural Tablebase: Enfoque Técnico

## Compresión de Endgame Tablebases mediante Redes Neuronales

**Objetivo**: Entrenar una red neuronal que replique el conocimiento de endgame tablebases con alta precisión y tamaño reducido.

---

## 1. Baseline: Números Actuales

| Componente | Tamaño |
|------------|--------|
| Tablebase 3-5 piezas | 939 MB |
| Tablebase 6 piezas | 149.2 GB |
| Tablebase 7 piezas | 16.7 TB |
| NNUE Stockfish (completa) | 39-70 MB |
| NNUE Stockfish (ligera) | ~7 MB |

**Ratio objetivo**: 149GB → ~10MB = 15,000x compresión

---

## 2. Arquitectura Propuesta: NNUE-Endgame

### 2.1 Input Features

```python
# Piece-Square encoding adaptado para endgames específicos
# A diferencia de NNUE normal (768 features), usamos encoding denso

class EndgameInputEncoder:
    """Codifica posición de piezas para red enfocada en endgame"""
    
    # Para cada combinación de piezas (ej: K+R vs K)
    # encoding de 12 pieces * 64 squares = 768 inputs
    
    # Feature: piece type + square
    # Binary: 1 si pieza X está en cuadrado Y
    
    # Optimización: solo codificar piezas presentes
    # (K+R vs K) → 3 reyes + 1 torre = encoding más pequeño
```

### 2.2 Arquitectura de Red

```python
class NeuralTablebase(nn.Module):
    """
    Red neuronal para replicar tablebase.
    Objetivo: alta precisión con mínimo tamaño.
    """
    
    def __init__(self, piece_config):
        super().__init__()
        
        # Input: piece-square encoding
        self.input_size = piece_config.feature_count  # variable según endgame
        
        # Capas ocultas: más pequeñas que NNUE estándar
        # NNUE usa 256-512-32
        # Nosotros probamos: 128-64-32, 64-32-16
        
        self.feature_layer = nn.Linear(input_size, HIDDEN1)
        self.hidden1 = nn.Linear(HIDDEN1, HIDDEN2)
        self.hidden2 = nn.Linear(HIDDEN2, OUTPUT)
        
        # Output:
        # - WDL: 3 valores (win/draw/loss) - softmax
        # - DTM: valor continuo (distance to mate)
        
    def forward(self, position_encoding):
        x = F.relu(self.feature_layer(position_encoding))
        x = F.relu(self.hidden1(x))
        wdl = F.softmax(self.hidden2_wdl(x), dim=-1)
        dtm = self.hidden2_dtm(x)
        return wdl, dtm
```

### 2.3 Variantes de Arquitectura a Testear

| Variante | Arquitectura | Params estimados |
|----------|--------------|------------------|
| A | 768-128-64-3 | ~100K |
| B | 768-64-32-3 | ~50K |
| C | 768-256-128-64-3 | ~200K |
| D | 768-512-256-128-3 | ~500K |

**Hipótesis**: Una red de ~100K parámetros podría aprender los patrones críticos.

---

## 3. Dataset de Entrenamiento

### 3.1 Generación desde Tablebase

```python
class TablebaseDatasetGenerator:
    """Genera dataset de entrenamiento desde tablebase Syzygy"""
    
    def __init__(self, endgame_config, tablebase_path):
        self.config = endgame_config  # ej: "KQKR"
        self.tb = syzygy.Tablebase(tablebase_path)
    
    def generate_dataset(self, max_samples=None):
        """
        Genera dataset: (position_encoding, wdl, dtm)
        """
        positions = []
        
        for position in self.tb.probe_all(self.config):
            encoding = self.encode_position(position)
            wdl = position.wdl  # -1, 0, 1
            dtz = position.dtz  # distance to zero
            
            positions.append({
                'encoding': encoding,
                'wdl': wdl,
                'dtz': dtz,
                'dtm': position.dtm  # distance to mate
            })
            
            if max_samples and len(positions) >= max_samples:
                break
        
        return positions
    
    def encode_position(self, position):
        """
        Convierte posición FEN a encoding neuronal.
        """
        # Binary encoding: 12 pieces × 64 squares
        encoding = torch.zeros(12, 64)
        
        for square in range(64):
            piece = position.board.piece_at(square)
            if piece:
                piece_idx = piece_to_index(piece)  # 0-11
                encoding[piece_idx, square] = 1.0
        
        return encoding.flatten()  # 768 valores
```

### 3.2 Dataset Sizes Estimados

| Endgame | Posiciones | Tamaño dataset |
|---------|------------|----------------|
| K+Q vs K | ~4,000 | trivial |
| K+R vs K+P | ~500K | ~4MB |
| K+B+K vs K+N | ~10M | ~80MB |
| K+Q vs K+R | ~50M | ~400MB |
| 6-piezas (promedio) | ~100M-1B | ~8GB |

### 3.3 Estrategia de Muestreo

```python
class BalancedSampler:
    """
    Sampling que balancea clases - crucial para endgames.
    Los datos están desbalanceados: muchos más draws que wins.
    """
    
    def __init__(self, dataset, oversample_minority=True):
        self.dataset = dataset
        self.oversample = oversample_minority
    
    def get_batch(self, batch_size):
        if self.oversample:
            # Oversample de minorías (wins, losses)
            minority = self.get_minority_samples(batch_size // 2)
            majority = self.get_majority_samples(batch_size // 2)
            return shuffle(minority + majority)
        else:
            return random.sample(self.dataset, batch_size)
```

---

## 4. Función de Loss

### 4.1 Loss Compuesto

```python
class TablebaseLoss(nn.Module):
    """
    Loss que optimiza tanto WDL como DTM.
    """
    
    def __init__(self, wdl_weight=1.0, dtm_weight=0.1):
        super().__init__()
        self.wdl_weight = wdl_weight
        self.dtm_weight = dtm_weight
    
    def forward(self, predictions, targets):
        wdl_pred, dtm_pred = predictions
        wdl_target, dtm_target = targets
        
        # WDL: Cross-entropy (clasificación 3 clases)
        wdl_loss = F.cross_entropy(wdl_pred, wdl_target)
        
        # DTM: MSE para regresión (solo en posiciones no-draw)
        dtm_mask = (wdl_target != 0).float()  # solo donde no es draw
        dtm_loss = F.mse_loss(dtm_pred * dtm_mask, 
                              dtm_target.float() * dtm_mask)
        
        return self.wdl_weight * wdl_loss + self.dtm_weight * dtm_loss
```

### 4.2 Accuracy Targets

| Métrica | Target Mínimo | Target Óptimo |
|---------|---------------|---------------|
| WDL Accuracy | 95% | 99%+ |
| DTM Error (avg) | < 5 plies | < 1 ply |
| Mate Detection | 99% | 99.9% |

---

## 5. Entrenamiento

### 5.1 Hiperparámetros

```python
config = {
    # Optimizador
    'optimizer': 'AdamW',
    'lr': 1e-3,
    'weight_decay': 1e-4,
    
    # Scheduler
    'scheduler': 'CosineAnnealingLR',
    'T_max': 100,
    
    # Training
    'batch_size': 1024,
    'epochs': 50,
    
    # Early stopping
    'patience': 5,
    'min_delta': 0.001,
}
```

### 5.2 Progressive Training

```python
class ProgressiveTrainer:
    """
    Entrena primero endgames simples, luego progresivamente
    más complejos.like a curriculum learning.
    """
    
    def train_progressive(self):
        # Fase 1: Endgames triviales (K+Q vs K)
        self.train_endgame('KQK', epochs=10)
        
        # Fase 2: Endgames 3-piezas
        self.train_endgame('KRK', epochs=20)
        self.train_endgame('KBK', epochs=20)
        
        # Fase 3: Endgames 4-piezas
        self.train_endgame('KQRK', epochs=30)
        self.train_endgame('KRNK', epochs=30)
        
        # Fase 4: Endgames 5-piezas
        self.train_endgame('KQRRK', epochs=40)
        
        # Fase 5: 6-piezas
        self.train_endgame('KQRKR', epochs=50)
```

### 5.3 Data Augmentation

```python
class TablebaseAugmenter:
    """
    Aumenta datos usando simetrías del tablero.
    """
    
    def augment(self, position):
        # Simetría horizontal
        aug1 = flip_horizontal(position)
        
        # Simetría vertical
        aug2 = flip_vertical(position)
        
        # Rotación 180 (equivale a swap colors + flip)
        aug3 = rotate_180(position)
        
        # Swap colors (White ↔ Black)
        aug4 = swap_colors(position)
        
        return [position, aug1, aug2, aug3, aug4]
```

---

## 6. Evaluación

### 6.1 Métricas

```python
def evaluate_model(model, test_set):
    """
    Evalúa modelo contra tablebase original.
    """
    results = {
        'wdl_accuracy': 0.0,
        'dtm_mae': 0.0,
        'mate_correct': 0.0,
        'draw_accuracy': 0.0,
        'win_accuracy': 0.0,
        'loss_accuracy': 0.0,
    }
    
    for pos in test_set:
        pred_wdl, pred_dtm = model(pos.encoding)
        true_wdl = pos.wdl
        true_dtm = pos.dtm
        
        # WDL accuracy
        if pred_wdl.argmax() == true_wdl:
            results['wdl_accuracy'] += 1
        
        # Per-class accuracy
        if true_wdl == 1 and pred_wdl.argmax() == 1:
            results['win_accuracy'] += 1
        elif true_wdl == 0 and pred_wdl.argmax() == 0:
            results['draw_accuracy'] += 1
        elif true_wdl == -1 and pred_wdl.argmax() == -1:
            results['loss_accuracy'] += 1
        
        # DTM error (solo en wins/losses)
        if true_wdl != 0:
            results['dtm_mae'] += abs(pred_dtm - true_dtm)
        
        # Mate detection
        if true_dtm == 1 and pred_dtm < 2:
            results['mate_correct'] += 1
    
    # Normalize
    n = len(test_set)
    for key in results:
        results[key] /= n
    
    return results
```

### 6.2 Test de Generalización

```python
def test_generalization(model, known_endgame, novel_positions):
    """
    Test crítico: puede la red generalizar a posiciones
    no vistas durante entrenamiento?
    """
    # Entrenar con 80% de posiciones
    train_set, val_set = split(known_endgame, 0.8)
    model.train(train_set)
    
    # Testear en 20% no visto
    accuracy = evaluate(model, val_set)
    
    # También testear en posiciones "novel" del mismo endgame
    novel_accuracy = evaluate(model, novel_positions)
    
    print(f"Accuracy en posiciones conocidas: {accuracy}")
    print(f"Accuracy en posiciones noveles: {novel_accuracy}")
    
    # Si两者相近 → la red generaliza, no memoriza
    # Si两者差很远 → la red memorizó
```

---

## 7. Cuantización y Compresión

### 7.1 Quantization

```python
# Int8 quantization para reducir tamaño
def quantize_model(model):
    """Convierte floats a int8."""
    quantized = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )
    return quantized
```

### 7.2 Pruning

```python
def prune_weights(model, sparsity=0.5):
    """
    Elimina pesos cercanos a 0.
    """
    for param in model.parameters():
        if len(param.shape) == 2:  # Solo linear layers
            threshold = np.percentile(
                abs(param.data.numpy()), 
                sparsity * 100
            )
            mask = abs(param.data) > threshold
            param.data *= mask.float()
```

---

## 8. Resultados Esperados

### 8.1 Proyección por Endgame

| Endgame | Tablebase | Red Propuesta | Compresión | WDL Acc |
|---------|-----------|----------------|------------|---------|
| K+Q vs K | <1MB | ~10KB | 100x | 99.9% |
| K+R vs K+P | ~10MB | ~100KB | 100x | 99% |
| K+B+K vs K+N | ~100MB | ~1MB | 100x | 98% |
| K+Q vs K+R | ~1GB | ~10MB | 100x | 95% |
| 6-piezas avg | ~1GB | ~10MB | 100x | 90%+ |

### 8.2 Objetivo Final

- **Tablebase 6-piezas completa**: 149GB → ~100MB
- **Accuracy WDL**: >95%
- **Inference time**: <1ms por posición

---

## 9. Roadmap de Implementación

### Fase 1: Proof of Concept (1-2 semanas)
- [ ] Implementar encoder de posiciones
- [ ] Generar dataset para K+Q vs K
- [ ] Entrenar red simple (100K params)
- [ ] Evaluar accuracy

### Fase 2: Escalamiento (2-3 semanas)
- [ ] Implementar generación de dataset para 3-4 piezas
- [ ] Testear arquitecturas variando tamaño
- [ ] Encontrar mínimo params para 95%+ accuracy
- [ ] Implementar quantización

### Fase 3: 5-6 Piezas (3-4 semanas)
- [ ] Generar dataset 5-piezas (requiere compute)
- [ ] Entrenar red para 5-piezas
- [ ] Generar dataset 6-piezas
- [ ] Entrenar red para 6-piezas

### Fase 4: Optimización (2 semanas)
- [ ] Fine-tuning de hiperparámetros
- [ ] Pruning y quantización
- [ ] Comparación con baselines

---

## 10. Compute Requerido

| Fase | Dataset Size | GPU | Tiempo Estimado |
|------|-------------|-----|-----------------|
| 1 | ~100K samples | CPU ok | 1 hora |
| 2 | ~10M samples | 1x RTX 3080 | 10 horas |
| 3 | ~1B samples | 4x RTX 3080 | 1 semana |

---

## 11. Referencias

- Nasu, Y. (2018). "Efficiently Updatable Neural Network"
- Stockfish NNUE: https://github.com/official-stockfish/Stockfish
- Syzygy Tablebases: https://github.com/syzygy1/tb
- "NeuralBases" - Bachelor thesis comparando NN y tablebases (2024)
- "Comparing Lossless Compression Methods for Chess Endgame Data" (ECAI 2024)

---

*Documento generado: 2026-03-09*
*Proyecto: Neural Tablebase Compressor*
