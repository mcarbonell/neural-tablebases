# Arquitectura GNN: Motor de Búsqueda Neuronal (Neural Search)

## 📌 Concepto Revolucionario
Apartarse de las redes MLP estándar (cajas negras) para construir una **Graph Neural Network (GNN)** cuya estructura física sea idéntica a las reglas del ajedrez. En esta arquitectura, la red no "imagina" el juego, sino que lo **simula** a través de la propagación de señales tácticas entre nodos (casillas).

---

## 🏗️ Topología de la Red

### Capa 0 (Input de Mapa de Calor)
Cada una de las 64 casillas (nodos) recibe 4 señales fundamentales calculadas por un generador de movimientos rápido (ej. `x88.js`):
1.  **Min_Attacker_Value_White**: Valor de la pieza blanca más débil atacando la casilla.
2.  **Total_Attackers_White**: Número total de atacantes blancos.
3.  **Min_Attacker_Value_Black**: Lo mismo para el bando negro.
4.  **Total_Attackers_Black**: Lo mismo para el bando negro.

### Capas 1 a N (Ply Simulation)
Cada capa de la red representa un **Ply (Semijugada)** de profundidad:
*   **Conexiones (Edges)**: Solo existen "cables" entre casillas si hay un movimiento legal posible (e1 conectado a e4 por una Torre).
*   **Mensajería (Message Passing)**: La "energía" táctica fluye por los cables. Si una casilla está saturada de señales de ataque rival y no tiene defensa, el nodo "colapsa" en la siguiente capa (pieza capturada conceptualmente).
*   **Visión de Rayos X (Skip Connections)**: Conexiones residuales que saltan capas para representar que una pieza bloqueada podría liberarse en un turno futuro.

---

## 🧠 Detección de Mate Natural
A diferencia de un modelo tradicional, en esta GNN el mate se detecta por la **convergencia de señales**. Si en el Ply 3 todas las señales de amenaza convergen en las casillas adyacentes al Rey rival y sus casillas de "Oxígeno" están en cero, la red emitirá una señal de salida de "Mate Inevitable" con 100% de confianza.

---

## 🛠️ Implementación Técnica Sugerida
*   **Grado de adyacencia dinámico**: La matriz de adyacencia se enmascara según la ocupación del tablero.
*   **Pesos compartidos**: Todas las capas de movimiento usan los mismos pesos (red recurrente), lo que mantiene el tamaño del modelo diminuto pero permite una profundidad de cálculo infinita aumentando las iteraciones.

---

## ⚡ El Ciclo de Retroalimentación (Feedback Reasoning)
A diferencia de las redes tradicionales de un solo sentido, esta GNN implementa un ciclo de **"Razonamiento Retroactivo"**:
1.  **Exploración (Forward)**: Las piezas proyectan su influencia en el tablero.
2.  **Consecuencia (Backward)**: Si una casilla de destino resulta ser peligrosa o no tiene salidas (Oxígeno = 0), la señal viaja de vuelta por el mismo cable al origen.
3.  **Inhibición**: La pieza de origen "apaga" su intención de moverse a esa casilla basándose en la retroalimentación de la capa superior.
4.  **Convergencia**: El proceso se repite $N$ veces hasta que la red alcanza un "Estado Estable" que representa la valoración pura de la posición tras considerar múltiples plys de consecuencias.

---

## 📈 Resumen
Esta arquitectura fusiona lo mejor de dos mundos: la **precisión matemática** de un generador de movimientos (movegen) con la **intuición estadística** de una red neuronal.
