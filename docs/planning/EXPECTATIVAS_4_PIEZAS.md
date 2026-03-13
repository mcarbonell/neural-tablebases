# Expectativas: Endgames de 4 Piezas

## Diferencias vs 3 Piezas

### Complejidad del Espacio

| Aspecto | 3 Piezas | 4 Piezas | Ratio |
|---------|----------|----------|-------|
| Combinaciones totales | 64³ = 262K | 64⁴ = 16.7M | **64x** |
| Posiciones válidas | ~330-400K | ~3-5M | **10x** |
| Tiempo de generación | 1-2 min | 10-15 min | **10x** |
| Encoding dims | 43 | 65 | **1.5x** |
| Parámetros del modelo | 453K | ~600K | **1.3x** |

### Encoding Relativo para 4 Piezas

```python
# Por pieza: 10 dims × 4 piezas = 40 dims
- Coordenadas (x, y): 2 dims
- Tipo [K,Q,R,B,N,P]: 6 dims
- Color [W,B]: 2 dims

# Por par: 4 dims × 6 pares = 24 dims
- Manhattan distance: 1 dim
- Chebyshev distance: 1 dim
- Direction (dx, dy): 2 dims

# Global: 1 dim
- Side to move: 1 dim

Total: 40 + 24 + 1 = 65 dims
```

**Pares de piezas (6 combinaciones):**
1. Pieza 1 ↔ Pieza 2
2. Pieza 1 ↔ Pieza 3
3. Pieza 1 ↔ Pieza 4
4. Pieza 2 ↔ Pieza 3
5. Pieza 2 ↔ Pieza 4
6. Pieza 3 ↔ Pieza 4

## KRRvK: Rey+2 Torres vs Rey

### Características:

**Dificultad:** ⭐ Fácil (similar a KQvK, KRvK)

**Resultado:** Siempre gana (excepto ahogado)

**Estrategia ganadora:**
1. Las 2 torres controlan filas/columnas
2. Acorralan al rey enemigo
3. Mate en la banda

**Distribución WDL esperada:**
```
Loss: ~55% (rey atacante puede ser capturado)
Draw: ~5% (ahogado)
Win: ~40% (mate inevitable)
```

### Accuracy Esperado:

| Época | Accuracy Esperado | Razón |
|-------|-------------------|-------|
| 1 | **98-99%** | Similar a KRvK (99.68%) |
| 5 | **99.5%+** | Convergencia rápida |
| 10 | **99.9%+** | Cerca de perfecto |

**Predicción:** Será tan fácil como KRvK, quizás más fácil porque 2 torres son más fuertes.

### Casos Difíciles:

1. **Ahogado:** Rey acorralado pero no en jaque
2. **Torres descoordinadas:** Torres se bloquean entre sí
3. **Rey atacante expuesto:** Puede ser capturado

## Arquitectura del Modelo

### Ajustes para 4 Piezas:

```python
# Modelo actual (3 piezas):
Input: 43 dims
Hidden: [512, 512, 256, 128]
Params: 452,740

# Modelo para 4 piezas:
Input: 65 dims
Hidden: [512, 512, 256, 128]  # Mismo
Params: ~470,000  # +4% más parámetros

# Alternativa (si es muy difícil):
Input: 65 dims
Hidden: [768, 768, 384, 192]  # Más grande
Params: ~900,000
```

**Recomendación:** Empezar con el modelo actual (512-512-256-128). Si no funciona bien, aumentar a 768-768-384-192.

## Comparación con Otros Endgames de 4 Piezas

### Por Dificultad:

| Endgame | Dificultad | Accuracy Esperado | Razón |
|---------|-----------|-------------------|-------|
| **KQQvK** | ⭐ | 99.9%+ | 2 damas, abrumador |
| **KRRvK** | ⭐ | 99.9%+ | 2 torres, fuerte |
| **KQRvK** | ⭐ | 99.9%+ | Dama+torre, muy fuerte |
| **KQPvK** | ⭐⭐ | 99.5%+ | Promoción del peón |
| **KRPvK** | ⭐⭐ | 99.5%+ | Torre+peón |
| **KPPvK** | ⭐⭐⭐ | 99%+ | 2 peones, complejo |
| **KQvKQ** | ⭐⭐⭐⭐ | 98%+ | Material igual, difícil |
| **KRvKR** | ⭐⭐⭐⭐ | 98%+ | Material igual, difícil |

### Por Tipo:

**Fáciles (siempre ganan):**
- KQQvK, KRRvK, KQRvK
- Accuracy esperado: >99.9%
- Épocas: 5-10

**Interesantes (promoción):**
- KQPvK, KRPvK, KPPvK
- Accuracy esperado: >99%
- Épocas: 10-20

**Complejos (material igual):**
- KQvKQ, KRvKR, KQvKR
- Accuracy esperado: >98%
- Épocas: 20-30

## Compresión Esperada

### Syzygy vs Neural:

```
KRRvK Syzygy: ~52 KB (archivo .rtbz)
KRRvK Neural: ~470 KB (INT8)

Compresión: 0.11x (más grande)
```

**¿Por qué más grande?**
- El modelo neural es general (funciona para múltiples endgames)
- Syzygy está optimizado específicamente para cada endgame
- Pero un solo modelo neural puede reemplazar múltiples archivos Syzygy

### Compresión Real:

```
3 endgames de 3 piezas:
  Syzygy: 10.4 + 16.2 + 8.2 = 34.8 MB
  Neural: 442 KB (un solo modelo)
  Compresión: 79x

Si añadimos 4 piezas:
  Syzygy: 34.8 + 52.3 + 25.3 + ... = ~200 MB
  Neural: 470 KB (un solo modelo)
  Compresión: 425x
```

**La ventaja crece con más endgames.**

## Tiempo de Entrenamiento

### Estimación:

```
Posiciones: ~3-5M (vs 330-400K en 3 piezas)
Batch size: 2,048
Batches por época: ~1,500-2,500 (vs 160-175)
Tiempo por época: ~2-3 min (vs 15 seg)

Total para 30 épocas: ~60-90 min (vs 7-8 min)
```

**Será ~10x más lento** que los de 3 piezas.

## Métricas de Éxito

### Mínimo Aceptable:
- Accuracy: >99%
- Épocas: <30
- Hard examples: <500

### Objetivo:
- Accuracy: >99.5%
- Épocas: <20
- Hard examples: <200

### Excelente:
- Accuracy: >99.9%
- Épocas: <10
- Hard examples: <50

## Posibles Problemas

### 1. Overfitting
**Síntoma:** Train acc >> Val acc  
**Solución:** Más dropout, más regularización

### 2. Underfitting
**Síntoma:** Train acc y Val acc bajos  
**Solución:** Modelo más grande (768-768-384-192)

### 3. Convergencia Lenta
**Síntoma:** Mejora <0.1% por época  
**Solución:** Ajustar learning rate, más épocas

### 4. Memoria Insuficiente
**Síntoma:** Out of memory  
**Solución:** Batch size más pequeño (1024 en lugar de 2048)

## Plan de Acción

### Fase 1: KRRvK (Fácil)
1. ✅ Generar dataset (~10-15 min)
2. ⏭️ Entrenar modelo (30 épocas, ~60 min)
3. ⏭️ Analizar resultados
4. ⏭️ Ajustar si es necesario

### Fase 2: KQvKQ (Difícil)
1. Generar dataset
2. Entrenar modelo
3. Comparar con KRRvK
4. Validar que el encoding escala

### Fase 3: Conclusiones
1. Documentar resultados
2. Comparar 3 vs 4 piezas
3. Estimar escalabilidad a 5+ piezas

## Conclusión

**KRRvK debería ser fácil** (similar a KRvK):
- Accuracy esperado: >99.9%
- Convergencia rápida: 5-10 épocas
- Hard examples: <100

Si funciona bien, confirma que el encoding relativo escala perfectamente a 4 piezas.

Si no funciona bien, necesitamos:
- Modelo más grande
- Más épocas
- Ajustar hiperparámetros

**Predicción:** Funcionará bien en el primer intento. 🎯

---

**Generando dataset ahora...**  
**Tiempo estimado:** 10-15 minutos  
**Próximo paso:** Entrenar y validar
