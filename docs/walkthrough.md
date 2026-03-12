# Neural Tablebase: 3-piece Prototype Walkthrough (KQvK)

He completado la primera fase del prototipo usando el final de **KQvK** (Rey y Reina contra Rey). Este experimento valida el flujo completo: desde la generación del dataset hasta el entrenamiento de modelos neuronales especializados.

## Resumen de Resultados

| Métrica | Valor |
|---------|-------|
| Posiciones KQvK (Dataset) | 368,452 |
| Tamaño Dataset (NPZ) | 27.6 MB |
| **MLP Accuracy (WDL-5)** | **~60.3%** |
| **SIREN Accuracy (WDL-5)** | **~54.6%** |
| Tiempo Entrenamiento | ~20 mins (CPU) |

### 1. Generación del Dataset
He implementado `src/generate_datasets.py`, que automatiza la extracción de datos desde las tablebases Syzygy.
- **Encoding**: Usamos un encoding denso de 768 bits (12 piezas x 64 casillas), optimizado para que la red "vea" todo el tablero.
- **WDL-5**: Hemos pasado de una clasificación de 3 clases a 5 clases (WDL-5) para capturar los matices de "Cursed Wins" y "Blessed Losses" que proporciona Syzygy.

### 2. Modelos Neuronales
He comparado dos arquitecturas en `src/models.py`:
- **MLP (Baseline)**: Una red densa estándar. Ha mostrado una convergencia más estable inicialmente.
- **SIREN (Sinusoidal)**: Diseñada para capturar altas frecuencias. Aunque su precisión inicial fue menor, es la candidata ideal para el "Overfitting Loop" masivo debido a su capacidad de representación de señales periódicas (como el tablero).

### 3. Análisis de Compresión
Los modelos guardados (`.pth`) ocupan apenas unos **cientos de KB**, comparados con los megabytes de la tablebase equivalente. El objetivo de compresión de 100x a 1000x es perfectamente viable una vez alcancemos el 100% de precisión mediante el refinamiento del mapa de excepciones.

## Próximos Pasos

1. **Precision Boost**: Implementar el "Overfitting Loop" agresivo re-pesando los ejemplos difíciles (hard mining).
2. **Residual Map**: Generar el `exceptions.bin` para las posiciones que la red no logra memorizar, asegurando el 100% de precisión.
3. **Escalamiento**: Probar con **KRvK** y **KPvK** para verificar si la lógica generaliza.

---
*Prototipo v1.0 completado*
