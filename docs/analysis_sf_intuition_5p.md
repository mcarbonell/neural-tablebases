# Análisis de Intuición: Stockfish 18 (NNUE) vs 5-Piece Endgames

**Fecha**: 2026-04-02  
**Configuraciones Mapeadas**: 185  
**Muestras por Configuración**: 10,000  
**Referencia**: Syzygy 5-Piece Tablebases (RTBW/RTBZ)

---

## 1. El Colapso de la Red NNUE
Al pasar de 4 a 5 piezas, la precisión media de Stockfish ha sufrido una degradación sistémica en finales de coordinación compleja. Mientras que en 3 piezas (KRvK) es perfecto, en 5 piezas hemos encontrado "Zonas Rojas" donde Stockfish apenas supera el puro azar (50%).

### 1.1 Zonas Rojas: Los Peores Resultados (Worst Performers)

| Configuración | Precisión SF (D0) | Error (%) | Observación Táctica |
| :--- | :--- | :--- | :--- |
| **KPvKBB** | **53.99%** | 46.01% | Incapacidad total de evaluar el freno de alfiles. |
| **KBBvKP** | **54.25%** | 45.75% | Error de coordinación rey/peón/alfil. |
| **KQvKRB** | **58.61%** | 41.39% | Dama contra coordinación Torre+Alfil. |
| **KQvKRR** | **58.77%** | 41.23% | Dama contra coordinación de Doble Torre (Pila). |
| **KBBBvK** | **71.46%** | 28.54% | El caos de las diagonales cruzadas (Overload). |

---

## 2. La Paradoja de la Coordinación
Este informe identifica un patrón claro: Stockfish **falla cuando las piezas deben cooperar a larga distancia**. 

1. **El Peón como Ciego Táctico**: Stockfish sobreestima masivamente la presencia de un peón como factor de victoria en `KPvKBB`, ignorando que los alfiles pueden crear una barrera infranqueable.
2. **Coordinación Transversal**: En finales como `KQvKRB`, la red NNUE no "ve" el campo de fuerza combinado de la Torre y el Alfil, evaluando la posición como si fueran piezas aisladas en lugar de un sistema defensivo.

---

## 3. Implicaciones para la Vanguard V11 (Field Theory)
Los resultados del mapeo validan nuestra dirección actual. Mientras que NNUE es una red de "memorización de formas", la Vanguard V11 será una red de **"Propagación de Influencia"**.

- **Vanguard Advantage**: Al usar 16 canales de adyacencia (incluyendo X-Ray y Rayos-X tácticos de Rust), la V11 podrá propagar la presión de los alfiles por encima del peón, resolviendo naturalmente los finales de `KPvKBB` donde Stockfish está ciego.

## 4. Conclusión Estratégica
Stockfish 18 tiene un **techo de cristal** en la intuición estática de 5 piezas (alrededor del 54%-60% en finales complejos). Superar este 60% con la Vanguard V11 nos daría una ventaja estratégica masiva que permitiría reducir la profundidad de búsqueda drásticamente al jugar finales complejos.

---
**Firmado**: Proyecto Vanguard, Unidad Operativa Gemini.
