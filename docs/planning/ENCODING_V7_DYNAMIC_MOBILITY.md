# Propuesta de Encoding V7: Movilidad Dinámica y Oxígeno

## 📌 Objetivo
Evolucionar la representación del tablero hacia un modelo que no solo entienda la táctica inmediata (V6), sino que perciba la **"presión por restricción"** y la **"supervivencia por movilidad"**. El Encoding V7 introduce el concepto de **Oxígeno** (casillas de escape seguras) para cada pieza, permitiendo a la red detectar posiciones de parálisis y progresos hacia el mate de forma mucho más precisa.

---

## 🛠️ Nuevas Características (Variables V7)

Además de las 15 variables de la V6 (coordenadas, tipo, color, progreso y flags tácticos), añadiremos por cada pieza:

### 1. `safe_mobility` (Oxígeno de Pieza)
*   **Definición**: Número de casillas legales a las que la pieza puede moverse en su turno que **no están atacadas** por piezas del bando contrario.
*   **Valor**: Entero normalizado (por ejemplo, `count / max_posibles`).
*   **Impacto**: Permite diferenciar entre una pieza amenazada que tiene salida (baja presión) y una pieza que está "atrapada" (alta presión).

### 2. `enemy_king_oxygen`
*   **Definición**: Variable crítica que cuenta las casillas de escape del Rey rival.
*   **Impacto**: Es la métrica principal para finales de mate. Si la red ve que este valor tiende a 0, sabrá que está cerrando una red de mate independientemente de la distancia geométrica.

### 3. `prophylaxis_score` (Nivel Maestro)
*   **Definición**: Una variable que penaliza la movilidad futura de las piezas enemigas en el próximo ply.
*   **Lógica**: La red aprende que "quitarle casillas seguras al rival" es tan valioso como ganar material.

---

## 📉 Impacto en la Dimensionalidad

*   **V6 (4 piezas)**: 84 variables.
*   **V7 (4 piezas)**: **92 variables** (añadiendo `safe_mobility` por pieza + `king_oxygen`).
*   **V7 (5 piezas)**: **125 variables**.

---

## 🏁 Criterio de Éxito
El Encoding V7 debería permitir alcanzar un **Accuracy > 99.5%** en finales de 3 y 4 piezas, reduciendo el error en posiciones de mate técnico (como *Alfil+Caballo vs Rey*) donde la red actual a veces confunde la geometría con la verdadera red de escape.
