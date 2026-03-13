# Análisis Preliminar: DTZ por Tipo de Final

**Fecha:** 13 de marzo de 2026  
**Estado:** En progreso

## Observaciones Iniciales

### DTZ MAE por Endgame (Época ~10)

| Endgame | WDL Acc | DTZ MAE | Complejidad DTZ |
|---------|---------|---------|-----------------|
| KQvK | 99.94% | **0.64** | Alta (mate complejo) |
| KRvK | 99.99% | **1.31** | Muy Alta (mate más largo) |
| KPvK | 99.48% | **0.12** | Baja (avance lineal) |

## Hipótesis: ¿Por qué KPvK tiene mejor DTZ MAE?

### 1. Movimiento Lineal del Peón
- El peón avanza en línea recta hacia promoción
- Menos variabilidad en el número de movimientos
- Patrón más predecible para la red

### 2. Rango de DTZ Más Estrecho
- KPvK: Máximo ~7 movimientos hasta promoción
- KQvK: Hasta 20 movimientos para mate
- KRvK: Hasta 20+ movimientos para mate

### 3. Menos Opciones Tácticas
- Peón: Solo puede avanzar o capturar
- Dama/Torre: Múltiples caminos al mate
- Menos ambigüedad = mejor predicción

## Hipótesis: ¿Por qué KRvK tiene peor DTZ MAE?

### 1. Mate Más Complejo
- Torre necesita coordinar con rey
- Múltiples secuencias de mate posibles
- Más movimientos = más error acumulado

### 2. Zugzwang
- Posiciones donde cualquier movimiento empeora
- Difícil para la red capturar estas sutilezas
- Puede confundir la distancia real

### 3. Rango de DTZ Más Amplio
- Mayor variabilidad en número de movimientos
- Más difícil para la red generalizar

## Implicaciones

### Para el Proyecto

1. **DTZ es más fácil en finales simples** (peones)
2. **DTZ es más difícil en finales complejos** (piezas mayores)
3. **MAE < 1.5 es excelente** para finales de 3 piezas

### Para Futuros Endgames

**Predicciones de DTZ MAE:**
- KPvKP: ~0.2-0.3 (peones, lineal)
- KRRvK: ~1.5-2.0 (complejo, múltiples piezas)
- KQvKR: ~2.0-3.0 (muy complejo, táctico)
- KBNvK: ~3.0-4.0 (mate muy complejo)

### Para Optimización

1. **Peso de DTZ loss** podría ser dinámico:
   - Finales simples: 0.3-0.4
   - Finales complejos: 0.5-0.7

2. **Arquitectura SIREN** podría ayudar:
   - Mejor para capturar transiciones sharp
   - Útil en finales con zugzwang

3. **Canonical forms** reducirán error:
   - Menos variabilidad en posiciones
   - Más ejemplos de cada "tipo" de posición

## Comparación con Syzygy

### Exactitud

| Endgame | Syzygy DTZ | Neural DTZ MAE | % Exacto (±0) | % Cercano (±1) |
|---------|------------|----------------|---------------|----------------|
| KQvK | 100% | 0.64 | ~60%? | ~90%? |
| KRvK | 100% | 1.31 | ~40%? | ~80%? |
| KPvK | 100% | 0.12 | ~90%? | ~99%? |

**Nota:** Porcentajes estimados, pendiente de validación con Syzygy.

### Utilidad Práctica

Para juego real:
- **±1 movimiento:** Perfectamente aceptable
- **±2 movimientos:** Aceptable en la mayoría de casos
- **±3+ movimientos:** Puede causar problemas con regla de 50

Nuestros resultados:
- KQvK: 0.64 → Excelente
- KRvK: 1.31 → Muy bueno
- KPvK: 0.12 → Perfecto

## Próximos Análisis

### Cuando terminen los entrenamientos:

1. **Distribución de errores**
   - Histograma de errores DTZ
   - Identificar outliers
   - Analizar posiciones con mayor error

2. **Correlación WDL-DTZ**
   - ¿Errores de WDL correlacionan con errores de DTZ?
   - ¿Posiciones difíciles para WDL también difíciles para DTZ?

3. **Validación con Syzygy**
   - Comparar predicciones con ground truth
   - Calcular % exacto (±0, ±1, ±2)
   - Identificar patrones de error

4. **Análisis por fase**
   - DTZ en posiciones cercanas al mate
   - DTZ en posiciones lejanas al mate
   - ¿Dónde falla más la red?

## Conclusión Preliminar

Los resultados preliminares son muy prometedores:
- **KPvK: 0.12 MAE** - Casi perfecto
- **KQvK: 0.64 MAE** - Excelente
- **KRvK: 1.31 MAE** - Muy bueno

La red aprende DTZ efectivamente, con mejor performance en finales simples (peones) y mayor dificultad en finales complejos (piezas mayores con mates largos).

---

**Estado:** 🔄 Esperando resultados finales de KRvK y KPvK  
**Próximo paso:** Análisis completo con distribución de errores
