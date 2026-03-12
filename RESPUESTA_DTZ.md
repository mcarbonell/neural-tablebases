# Respuesta: ¿Qué pasa con el DTZ?

## Estado Actual

**SÍ estamos extrayendo DTZ** de Syzygy, pero **NO lo estamos usando** en el entrenamiento.

### Datos que tenemos:

```python
# En generate_datasets.py (línea 234-235):
wdl = tablebase.probe_wdl(board)  # Win/Draw/Loss
dtz = tablebase.probe_dtz(board)  # Distance to Zeroing move

# Se guardan ambos:
np.savez_compressed(output_path,
                    x=np.array(positions),
                    wdl=np.array(labels_wdl),    # ✓ Usado
                    dtz=np.array(labels_dtz))    # ✗ NO usado
```

### Datos que estamos usando:

```python
# En train.py:
self.wdl = torch.from_numpy(wdl_mapped).long()  # ✓ Entrenamos WDL
self.dtz = torch.from_numpy(data['dtz']).float()  # ✗ Solo cargamos, no entrenamos
```

## ¿Qué es DTZ?

**DTZ (Distance to Zeroing move):** Número de movimientos hasta el próximo movimiento irreversible:
- Captura de pieza
- Movimiento de peón
- Después de estos movimientos, la regla de 50 movimientos se resetea

### Ejemplo KPvK:

```
Posición: Rey+Peón en rank 6 vs Rey
WDL: Win (2)
DTZ: 2 movimientos hasta que el peón avanza (zeroing move)
```

### Distribución DTZ en KPvK:

```
Loss positions: DTZ range [-20, -2], mean=-2.62
Draw positions: DTZ = 0 (siempre)
Win positions:  DTZ range [1, 19], mean=1.56
```

## ¿Deberíamos entrenar DTZ?

### Ventajas de entrenar DTZ:

1. **Información más rica:** No solo "gana/pierde", sino "en cuántos movimientos"
2. **Mejor juego:** Puede elegir el camino más rápido a la victoria
3. **Regla de 50 movimientos:** Evita empates por regla de 50 movimientos
4. **Compresión adicional:** Un solo modelo predice WDL + DTZ

### Desventajas:

1. **Más complejo:** Necesita predecir un valor continuo (regresión)
2. **Más difícil de aprender:** DTZ es más sutil que WDL
3. **No crítico para compresión:** WDL es suficiente para juego perfecto

## Arquitectura Actual

El modelo YA tiene una cabeza para DTZ:

```python
class MLP(nn.Module):
    def __init__(self, ...):
        self.backbone = Sequential(...)
        self.wdl_head = Linear(128, 3)      # ✓ Usado
        self.dtz_head = Linear(128, 1)      # ✗ NO usado
```

## Cómo Entrenar DTZ

Para entrenar DTZ, necesitaríamos:

```python
# En train.py, añadir loss de DTZ:
wdl_loss = criterion(wdl_pred, wdl_target)
dtz_loss = F.mse_loss(dtz_pred, dtz_target)  # Regresión

# Loss combinado:
total_loss = wdl_loss + 0.1 * dtz_loss  # Weight menor para DTZ
```

## Recomendación

### Para el experimento actual:

**NO entrenar DTZ todavía** porque:
1. WDL es suficiente para demostrar compresión
2. Añade complejidad innecesaria
3. El experimento ya es un éxito con solo WDL

### Para el futuro:

**SÍ entrenar DTZ** si queremos:
1. Juego más fuerte (camino óptimo)
2. Evitar regla de 50 movimientos
3. Publicar un motor de ajedrez completo

## Comparación con Syzygy

**Syzygy tiene 2 tipos de tablebases:**

1. **WDL tablebases (.rtbw):** Solo Win/Draw/Loss
   - Más pequeñas
   - Suficiente para juego perfecto
   - KQvK: 5.3 KB

2. **DTZ tablebases (.rtbz):** Win/Draw/Loss + DTZ
   - Más grandes
   - Mejor para juego óptimo
   - KQvK: 5.0 KB (similar porque es simple)

**Nuestro modelo:**
- Puede predecir ambos con un solo modelo
- 442 KB (INT8) para WDL + DTZ
- Más grande que Syzygy individual, pero:
  - Un solo archivo para múltiples endgames
  - Escalable a cualquier endgame
  - No necesita reglas específicas

## Conclusión

**Estado actual:** Extraemos DTZ pero no lo entrenamos.

**Recomendación:** Terminar el experimento con WDL, luego añadir DTZ como mejora futura.

**Prioridad:**
1. ✅ Completar 3-piece endgames (KPvK)
2. ✅ Probar 4-piece endgames
3. ✅ Demostrar compresión exitosa
4. ⏭️ Añadir entrenamiento de DTZ (futuro)

---

**Respuesta corta:** Sí, tenemos DTZ en los datos, pero no lo estamos entrenando. Es una mejora futura, no crítica para el experimento actual.
