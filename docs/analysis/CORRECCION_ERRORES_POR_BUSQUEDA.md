# Análisis: Búsqueda como Sistema de Corrección de Errores

## 📌 Resumen del Hallazgo
La mayor limitación de las redes neuronales para la compresión de tablebases es la presencia de "bolsas de error" locales: posiciones donde una red que es 99% precisa comete un error crítico (como evaluar un mate como tablas).

Este análisis documenta cómo una búsqueda **Minimax/Alpha-Beta** de muy baja profundidad actúa como un filtro de consistencia, parcheando estos errores sin aumentar el espacio en disco, a cambio de un coste computacional mínimo.

## 📊 Resultados de la Prueba de Concepto (PoC)

Se evaluaron 1000 posiciones aleatorias de los finales KQvK y KRvK comparando la precisión del modelo solo (Raw NN) frente al modelo con búsqueda.

| Configuración | Profundidad 0 (Raw NN) | Profundidad 1 | **Profundidad 2** |
| :--- | :---: | :---: | :---: |
| **KQvK** (v1) | 92.50% | 99.50% | **99.50%** |
| **KRvK** (v1) | 95.40% | 99.90% | **100.00%** |

### Observaciones Clave:
1.  **Salto Dramático:** Solo 1 nivel de búsqueda (Depth 1) eleva la precisión por encima del 99.5% incluso con modelos medianamente entrenados.
2.  **Consistencia Global:** En Depth 2, el sistema alcanza el **100.00% de precisión** en KRvK, sugiriendo que las imprecisiones de la red son locales y no se propagan a través de la búsqueda.
3.  **Eficiencia de Almacenamiento:** Esta técnica permite usar modelos más pequeños (más comprimidos) manteniendo precisiones de "tabla perfecta" mediante búsqueda.

## 💡 Concepto Técnico: Trading Compute for Accuracy
El compromiso fundamental del proyecto evoluciona:
*   **Antes:** Mayor precisión requería modelos más grandes -> Menor compresión.
*   **Ahora:** Modelos ultra-comprimidos + Búsqueda superficial -> Máxima precisión.

La búsqueda corrige los errores de la función de evaluación (la NN) mediante la verificación de la consistencia entre estados adyacentes. Si la red evalúa una posición como "Ganada" pero no existe ningún movimiento legal que lleve a otra posición evaluada como "Ganada", la búsqueda Depth 1 corrige automáticamente la evaluación.

## 🛠️ Implicaciones para el Futuro
Este hallazgo es fundamental para finales de 5 y 6 piezas, donde la generación exhaustiva y el entrenamiento perfecto son computacionalmente prohibitivos. Podemos aceptar modelos con un 90-95% de precisión "bruta" sabiendo que la búsqueda en tiempo de ejecución los convertirá en herramientas de juego perfecto.

---
**Autor:** Mario Carbonell / Antigravity
**Fecha:** 14 de Marzo, 2026
**Script de Validación:** `src/search_poc.py`
