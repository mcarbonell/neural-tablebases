# Propuesta Conceptual: Vanguard V11 - Square-Centric Field Theory

## 1. Visión: El Tablero como Campo de Influencia
Vanguard V11 propone un cambio de paradigma en la inteligencia artificial de ajedrez. En lugar de modelar las piezas como objetos aislados que interactúan a distancia (Piece-Centric), modelamos el **tablero como un tejido de 64 casillas interconectadas** donde la información táctica fluye como un campo físico (Field Theory).

### Cambio de Nodo:
- **V10.1 (Actual)**: Nodos = Piezas (3-32).
- **V11 (Propuesta)**: Nodos = Casillas (64 fijas).

## 2. Arquitectura del Grafo: "La Red de Posibles"
El grafo de la V11 no es dinámico en su topología base, sino en sus mensajes:
- **Nodos**: 64 neuronas espaciales representativas de cada casilla (A1 a H8).
- **Aristas (Edges)**: Conexiones basadas en movimientos legales potenciales.
    - Un nodo está conectado a sus vecinos por movimientos de Rey, Alfil, Torre, Caballo y Peón.
    - Ejemplo: El nodo `e4` tiene aristas directas hacia `e5`, `d5`, `f3`, etc.

## 3. Dinámica: "1 Capa = 1 Ply"
Este es el núcleo de la visión. En lugar de una búsqueda Minimax tradicional, la propagación de mensajes en la GNN simula la profundidad táctica:
- **Inyección Local (Layer 0)**: Cada casilla con una pieza inyecta un vector de "Identidad" y "Potencial" (ej: Alfil + Rayo diagonal).
- **Mensajería Multi-Canal**: La red no pasa un solo valor; pasa vectores que representan:
    - **Canal de Ataque W/B**: Amenazas directas e indirectas.
    - **Canal de Control de Espacio**: Casillas accesibles pero no necesariamente amenazadas.
    - **Canal de Protección**: Piezas que defienden a otras.
- **Acumulación Sinuasoidal**: Al pasar por N capas, una casilla como `e1` puede sentir que a 8 "saltos" de distancia el Rey rival en `h8` está en peligro, sin necesidad de calcular cada movimiento intermedio.

## 4. Resolución de la Oclusión: El Mecanismo de Gating
Para que los mensajes no atraviesen piezas (como una torre bloqueada por un peón), implementaremos un **Gating Mechanism** basado en atención relacional:
- **Edge Weight ($W_{ij}$)**: El flujo de información entre la casilla $i$ y la casilla $j$ se multiplica por $(1 - \text{Occupancy}_k)$, donde $k$ son las casillas intermedias.
- En una GNN, esto se traduce en que si una casilla tiene una pieza, su estado interno emite una señal de "bloqueo" que apaga las aristas de largo alcance que pasan por ella.

## 5. Filosofía: Intuición Masiva vs Fuerza Bruta (Continuación)
Vanguard V11 no compite en velocidad de nodos por segundo con NNUE. Compite en **Precisión Estática (D0)**. 
- Al tener una red que "siente" las líneas de fuerza y las debilidades espaciales en múltiples capas de paso de mensajes, la necesidad de una búsqueda profunda se reduce. 

## 6. Tensores de Adyacencia y Topología
A diferencia de V10, la V11 utiliza una **Matriz de Adyacencia Fija de 64x64** multiplicada por tipos de relación (King-move, Rook-move, etc.):
- **Adjacency Tensor**: Un cubo de $[64, 64, \text{Num\_Move\_Types}]$. 
- Esto permite que la red aprenda por separado cómo fluye el "ataque de caballo" versus el "ataque de dama", convergiendo todo en el estado final de cada casilla.

## 7. Próximos Pasos (Hoja de Ruta)
1.  **Validación KBBvK (V10.1)**: Finalizar el rescate del 99.9%. (En proceso: 99.5% estable).
2.  **Dataset V11 Generator**: Adaptar `src/generate_data.py` para extraer estados de casillas (64 nodos) en lugar de piezas.
3.  **Primer Duelo V11 vs SF**: Comparar la visión de King Safety en posiciones de mediojuego.

---
**Investigador Principal**: USER
**Arquitecto de IA**: Gemini (Antigravity)
**Fecha**: 2026-04-01
