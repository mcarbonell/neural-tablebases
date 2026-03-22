# Evolución al Encoding V6 (Táctica Pre-calculada)

## 📌 Objetivo del Documento
Dejar constancia del plan de migración hacia la versión **V6** de la representación geométrica (Encoding) de los tableros de ajedrez.
La premisa matemática es reducir el tiempo computacional de la red neuronal inyectando "Píldoras Algorítmicas de Lógica Clásica". Si le damos a cada neurona información ya masticada de un motor de movimientos, eliminaremos el temido "Olvido Catastrófico Geométrico" e impulsaremos la métrica más allá del **96%**.

---

## 🔬 Inspiración: El Motor x88 en JavaScript 
La inspiración para V6 nace directamente del motor puro programado en JS (`x88.js`). Dicho generador escupe 7 millones de nodos/s y, más importante aún, detecta en una sola pasada:
*   Clavadas absolutas (`pinDirection`).
*   Piezas colgadas o *Hanging* (donde atacantes superan a los defensores).
*   Movimientos libres de riesgo (`mask_safe`).
*   Potenciales de Jaque.

Dado que la prioridad del pipeline de ML es tener 0 bugs, aplicaremos la metodología de **Opción A (Implementación PyPura)**: en lugar del JS nativo en bucle cerrado, extraeremos estas mismas banderas tácticas utilizando las funciones pre-validadas de `python-chess` en tiempo de *Dataset Generation*.

---

## 🛠️ Plan de Ejecución Técnico

### Fase 1: Enriquecer el Extractor de Características en Python
Tenemos que modificar el archivo `src/generate_datasets_parallel.py` (o el equivalente donde resida `encode_board_v5`).
Para cada pieza detectada (`color, piece_type, square`), vamos a extraer 4 `booleanos`/`enteros` tácticos extra:

1.  **`is_pinned`**: Si la pieza la mueve el rival, pierde al rey.
    *   *Código Python:* `board.is_pinned(color, square)`
2.  **`attackers_count`**: Densidad de ataque rival sobre la casilla.
    *   *Código Python:* `len(board.attackers(not color, square))`
3.  **`defenders_count`**: Escudo de invulnerabilidad aliado.
    *   *Código Python:* `len(board.attackers(color, square))`
4.  **`is_hanging`**: Si los atacantes superan puramente en número a los defensores (simplificación táctica).
    *   *Código Python:* `1.0 si attackers_count > defenders_count else 0.0`

### Fase 2: Redimensión de Tensores (Matemáticas V6)
Actualmente, el V5 destilaba la información de cada pieza en **11 variables de array flotante** + los pares cruzados. 
Si añadimos 4 banderas tácticas a cada una de nuestras piezas teóricas (supongamos 5 piezas máximo), esteremos sumando **20 nuevas dimensiones puras** (`5 * 4`).
*   La dimensión máxima oficial pasará de **95 dimensiones (V5 - 5 pieces)** a **115 dimensiones (V6 - 5 pieces)**.
*   En finales de 4 piezas pasará de **68 variables** a **84 variables**.

### Fase 3: Refactorización y Compatibilidad
1.  **Modificar DataLoaders**: `src/train.py` y `src/train_large_scale.py` deben admitir de forma nativa la firma del `encoding_version == 6`, o en su defecto inferirlo por el tamaño de las `input_dim`.
2.  **Scripts de Sharding**: La Trituradora Multidimensional (`build_universal_shards.py`) pasará de `max_dim = 68/95` a absorber el padding V6 perfectamente.

---

## 🏁 Criterio de Éxito
Cuando pasemos el V6 MLP al mismo test universal, experimentaremos una bajada violenta inicial de la *Loss* (dado que "Pieza clavada = Muerte segura" será un peso con gradiente exponencial en los primeros 5 batches), catapultando el *accuracy* mucho más rápido al 99%.
