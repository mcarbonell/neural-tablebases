# Análisis de Compresión Total: 956 MB → ¿?

## Objetivo: Comprimir Todo Syzygy

**Directorio Syzygy completo:** 956 MB  
**Objetivo:** Comprimir con un modelo neural universal

## Inventario de Endgames en Syzygy

### Por Número de Piezas:

```bash
# Contar archivos por número de piezas
3 piezas: 5 endgames
4 piezas: 30 endgames
5 piezas: 145 endgames
6 piezas: ~500 endgames (estimado)
```

### Distribución de Tamaño:

| Piezas | Endgames | Tamaño Promedio | Tamaño Total |
|--------|----------|-----------------|--------------|
| 3 | 5 | ~7 MB | ~35 MB |
| 4 | 30 | ~15 MB | ~450 MB |
| 5 | 145 | ~3 MB | ~435 MB |
| 6 | ~20 | ~2 MB | ~36 MB |

**Total:** ~956 MB

## Estrategia de Compresión Neural

### Opción 1: Un Modelo por Número de Piezas

```
Modelo 3-piezas: 43 dims → 442 KB (INT8)
  Cubre: KQvK, KRvK, KPvK, KBvK, KNvK
  Syzygy: 35 MB
  Compresión: 79x

Modelo 4-piezas: 65 dims → ~500 KB (INT8)
  Cubre: 30 endgames
  Syzygy: 450 MB
  Compresión: 900x

Modelo 5-piezas: 91 dims → ~600 KB (INT8)
  Cubre: 145 endgames
  Syzygy: 435 MB
  Compresión: 725x

Modelo 6-piezas: 121 dims → ~700 KB (INT8)
  Cubre: ~20 endgames
  Syzygy: 36 MB
  Compresión: 51x

Total Neural: 442 + 500 + 600 + 700 = 2.2 MB
Total Syzygy: 956 MB
Compresión: 435x
```

### Opción 2: Un Modelo Universal

```
Modelo universal: Max 121 dims (6 piezas)
  Padding para endgames con menos piezas
  Tamaño: ~800 KB (INT8)
  
Compresión: 956 MB / 800 KB = 1,195x
```

**Ventaja:** Un solo archivo  
**Desventaja:** Menos eficiente para endgames simples

### Opción 3: Modelos Especializados

```
Modelo "Siempre Ganan" (KQQvK, KRRvK, etc.): 400 KB
  Cubre: ~50 endgames fáciles
  Syzygy: ~300 MB
  
Modelo "Con Peones" (KPvK, KQPvK, etc.): 500 KB
  Cubre: ~80 endgames con peones
  Syzygy: ~400 MB
  
Modelo "Material Igual" (KQvKQ, KRvKR, etc.): 600 KB
  Cubre: ~35 endgames complejos
  Syzygy: ~256 MB

Total Neural: 1.5 MB
Total Syzygy: 956 MB
Compresión: 637x
```

## Compresión Realista

### Con Accuracy 99.9%:

```
Modelo neural: 2.2 MB (opción 1)
Exception map: ~50 KB (0.1% de posiciones)
Total: 2.25 MB

Compresión: 956 MB / 2.25 MB = 425x
```

### Con Accuracy 99%:

```
Modelo neural: 2.2 MB
Exception map: ~500 KB (1% de posiciones)
Total: 2.7 MB

Compresión: 956 MB / 2.7 MB = 354x
```

## Encoding Relativo por Número de Piezas

| Piezas | Per Piece | Pairs | Global | Total |
|--------|-----------|-------|--------|-------|
| 3 | 3×10=30 | 3×4=12 | 1 | **43** |
| 4 | 4×10=40 | 6×4=24 | 1 | **65** |
| 5 | 5×10=50 | 10×4=40 | 1 | **91** |
| 6 | 6×10=60 | 15×4=60 | 1 | **121** |
| 7 | 7×10=70 | 21×4=84 | 1 | **155** |

**Fórmula:** `n×10 + (n×(n-1)/2)×4 + 1`

## Tamaño del Modelo por Complejidad

### Arquitectura Escalable:

```python
# 3 piezas (43 dims):
[43 → 512 → 512 → 256 → 128 → 3]
Params: 452,740 = 442 KB (INT8)

# 4 piezas (65 dims):
[65 → 512 → 512 → 256 → 128 → 3]
Params: ~470,000 = 460 KB (INT8)

# 5 piezas (91 dims):
[91 → 768 → 768 → 384 → 192 → 3]
Params: ~900,000 = 880 KB (INT8)

# 6 piezas (121 dims):
[121 → 1024 → 1024 → 512 → 256 → 3]
Params: ~1,800,000 = 1.76 MB (INT8)
```

**Total:** 442 + 460 + 880 + 1,760 = 3.5 MB

## Compresión Final Estimada

### Escenario Conservador:

```
Modelos neurales: 3.5 MB
Exception maps (1%): 1 MB
Total: 4.5 MB

Compresión: 956 MB / 4.5 MB = 212x
```

### Escenario Optimista:

```
Modelos neurales: 2.2 MB
Exception maps (0.1%): 100 KB
Total: 2.3 MB

Compresión: 956 MB / 2.3 MB = 416x
```

### Escenario Realista:

```
Modelos neurales: 3.0 MB
Exception maps (0.5%): 500 KB
Total: 3.5 MB

Compresión: 956 MB / 3.5 MB = 273x
```

## KRvKP: El Próximo Reto

### Características:

**Configuración:** Rey+Torre (blanco) vs Rey+Peón (negro)

**Dificultad:** ⭐⭐⭐ Media-Alta

**Asimetría:**
- Blanco tiene material superior (Torre > Peón)
- Negro tiene amenaza de promoción
- Muy táctico: Torre debe parar el peón

**Distribución WDL esperada:**
```
Win (blanco):  ~60% (Torre captura peón)
Draw:          ~30% (Peón promociona o ahogado)
Loss (blanco): ~10% (Peón promociona y gana)
```

**Complejidad:**
1. **Promoción del peón:** Crítica
2. **Distancia torre-peón:** Puede parar o no
3. **Posición del rey negro:** Protege el peón
4. **Posición del rey blanco:** Ayuda a la torre

### Accuracy Esperado:

| Época | Accuracy | Razón |
|-------|----------|-------|
| 1 | 95-97% | Más complejo que KPvK |
| 10 | 99%+ | Aprende tácticas |
| 20 | 99.5%+ | Domina el endgame |

**Será más difícil que KRRvK** pero el encoding relativo debería manejarlo.

## Plan de Pruebas

### Fase 1: Validar Escalabilidad (Fácil)
1. ✅ KQvK: 99.92%
2. ✅ KRvK: 99.99%
3. ✅ KPvK: 99.89%
4. ⏳ KRRvK: En progreso

### Fase 2: Probar Complejidad (Media)
5. ⏭️ KRvKP: Asimétrico, peón tricky
6. ⏭️ KQPvK: Promoción con dama
7. ⏭️ KPPvK: Doble peón

### Fase 3: Reto Final (Difícil)
8. ⏭️ KQvKQ: Material igual
9. ⏭️ KRvKR: Material igual
10. ⏭️ 5 piezas: Validar escalabilidad

## Conclusión

**Compresión objetivo:** 956 MB → 3-4 MB  
**Ratio:** ~250-300x  
**Accuracy:** >99%

**Próximos pasos:**
1. Terminar KRRvK (validar 4 piezas fácil)
2. Probar KRvKP (validar asimetría + peones)
3. Estimar si podemos alcanzar el objetivo

**Predicción:** Con 99% accuracy y exception maps, podemos comprimir 956 MB a ~3.5 MB, logrando **273x compresión**. 🎯

---

**Estado actual:**  
- KRRvK generando: 483K posiciones (en progreso)
- Próximo: KRvKP (asimétrico, peón tricky)
- Objetivo: 956 MB → 3.5 MB (273x)
