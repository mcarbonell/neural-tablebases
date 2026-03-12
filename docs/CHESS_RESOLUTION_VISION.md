# Chess Resolution: The Neural Tablebase Vision

## 1. El Gran Desafío: El Número de Shannon
El ajedrez tiene aproximadamente $10^{40}$ posiciones legales. Almacenar esta información de forma tradicional (bit a bit) es físicamente imposible. Las tablebases actuales (Syzygy/Lomonosov) crecen exponencialmente:
- **6 piezas**: 149 GB
- **7 piezas**: 18.4 TB
- **8 piezas (est.)**: ~2 PB (Petabytes)

## 2. Fundamentos Técnicos de la Compresión Neuronal

### 2.1 Teorema de Compresión de Kolmogorov
La **Complejidad de Kolmogorov** de una tablebase es la longitud del programa (o red neuronal) más corto capaz de generar dicha tabla sin errores. Nuestra tesis es que el ajedrez no es ruido blanco; tiene una estructura lógica profunda que permite una descripción mucho más corta que su enumeración explícita.

### 2.2 Densidad de Información: ¿Cuánto cabe en 10MB?
Para lograr el objetivo de comprimir 149GB en 10MB (**15,000x**), debemos maximizar la eficiencia de cada parámetro:
- **10MB en Float32** $\approx$ 2.6 millones de parámetros.
- **10MB en Int8 (Quantized)** $\approx$ 10 millones de parámetros.
  
**La "Función de Capacidad" ($\mathbb{C}$):**
Podemos estimar la capacidad de almacenamiento de una red como:
$$\mathbb{C} \approx P \times \Gamma$$
Donde:
- $P$ es el número de parámetros.
- $\Gamma$ es el **Factor de Generalización** (cuántas posiciones "explica" un solo parámetro gracias a la lógica geométrica).
  
*Ejemplo:* Si $\Gamma \approx 100$ (un parámetro captura un patrón común a 100 posiciones), una red de 10MB (Int8) podría codificar **1,000 millones de posiciones**, cubriendo gran parte de un final de 6 piezas.

### 2.3 Arquitecturas Predictivas vs. Representativas
Para este proyecto ignoramos la capacidad de la red para "jugar" y nos enfocamos en su capacidad para **mapear**:
- **SIRENs (Sinusoidal Representation Networks):** Usan activadores `Sine` en lugar de `ReLU`. Son ideales para señales de alta frecuencia y patrones periódicos (como las casillas del tablero), eliminando el "sesgo de baja frecuencia" de las redes normales.
- **KAN (Kolmogorov-Arnold Networks):** Sustituyen pesos fijos por funciones (splines) en las aristas. Pueden representar funciones matemáticas con **100 veces menos parámetros** que un MLP tradicional.

## 3. El "Muro Fractal" y la Solución Híbrida
El valor WDL del ajedrez es discreto y "picudo" (pequeños cambios de posición causan grandes cambios de resultado). Forzar un 100% de precisión en una red pura causaría un *overfitting* ineficiente (la red se convierte en una tabla hash lenta).
- **Solución:** Una red que capture el 99.x% del patrón + una pequeñas **Tabla de Excepciones** ultra-comprimida para el residuo.

## 4. Hoja de Ruta hacia las 32 Piezas

### Fase I: El Ladrillo Base (4-6 Piezas)
Demostrar que KAN/SIREN pueden "memorizar" finales de 4-6 piezas en una fracción del tamaño actual.
- *Hito*: Comprimir 6 piezas (149GB) en <100MB.

### Fase II: La Frontera del Almacenamiento (7-8 Piezas)
Hacer que las 7 y 8 piezas sean accesibles en RAM doméstica.
- *Hito*: Comprimir 8 piezas (~2PB) en ~20-30GB.

### Fase III: El Descubrimiento Autónomo (9-32 Piezas)
Uso de **Self-Play Neural Discovery**. Las redes de niveles inferiores enseñan a las superiores mediante *Curriculum Learning* masivo.

## 5. El Objetivo Final: Resolver el Ajedrez
La visión última es que el "conocimiento perfecto" del ajedrez sea una **función matemática continua**. Un modelo que codifique la "Ley del Ajedrez", permitiendo conocer el resultado perfecto de cualquier posición desde la jugada 1.

---
*“El ajedrez es demasiado rico para ser resuelto por la fuerza bruta, pero demasiado lógico para no ser capturado por una función.”*

**Documento de Visión v1.1**
**Proyecto**: Neural-Zero-Loss Tablebases
