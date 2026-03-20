# Lecciones Aprendidas: El Salto del 97% al 99.7% en KPvKP (4 Piezas)

## 📌 Contexto del Avance
Durante la sesión de entrenamiento iniciada el 20 de marzo de 2026, logramos superar la barrera del **99.7% de precisión raw** en el final de `KPvKP` (7.4 millones de posiciones canónicas). Este documento resume las innovaciones técnicas que permitieron este salto cualitativo.

---

## 1. El Encoding V5: Soberanía del Rey (The Anchor)
**Problema:** En las versiones V1-V4, las piezas podían aparecer en cualquier slot del vector de entrada. La red perdía capacidad neuronal intentando identificar "dónde está mi rey" en cada posición.
**Solución:** Fijar al Rey del bando que mueve (STM) en el **Slot 0 (Índice 0-7)** del vector.
**Resultado:** Esto actúa como un ancla geográfica constante. La red ya no "busca" las piezas, sino que mide el tablero **desde la perspectiva de su propio Rey**, estabilizando el aprendizaje desde la primera época.

## 2. El "Overfitting Loop" (Hard Mining en Caliente)
**Problema:** Un dataset de 7 millones de posiciones tiene "regiones tranquilas" (triviales) y "regiones de guerra" (tácticas). La red tiende a priorizar el promedio, fallando en los casos tácticos más raros.
**Solución:** Implementamos un bucle de re-entrenamiento automático cada 50 épocas. La red identifica las posiciones donde ha fallado, y antes de pasar a la siguiente época, se somete a un "entrenamiento de castigo" solo con esos ejemplos difíciles (Hard Examples).
**Resultado:** Vimos saltos de precisión del **99.55% al 99.64%** en solo un ciclo de Hard Mining (Época 150).

## 3. El Horizonte de Transformación
**Problema:** Un final de peones (`KPvKP`) puede transformarse en uno de Damas (`KQvKP` o `KQvKQ`) tras la coronación. La red de peones debe saber si la futura posición de Damas es ganadora, sin tener Damas en el tablero actual.
**Descubrimiento:** La red no solo aprende geometría de peones, sino que **comprime el valor futuro** de las piezas pesadas. Si la red sabe que una carrera de peones lleva a una Dama que no puede ganar (tablas teóricas), es capaz de asignar un valor de "Tablas" a la posición inicial de peones.
**Propuesta Futura:** *Curriculum Learning por Material*. Entrenar primero en finales de Damas y luego transferir ese conocimiento a los peones para que ya conozcan "el final de la película".

## 4. La Sinergia Red + Búsqueda superficial
**Problema:** Alcanzar el 100.00% de precisión raw en 4 o 5 piezas requeriría redes neuronales prohibitivamente grandes (Gigas).
**Solución:** Usar una red de tamaño medio (~2 MB) y aplicar una **búsqueda Minimax de profundidad 1 o 2 (D1/D2)**. 
**Resultado:** Con una precisión raw del **99.6%**, la búsqueda D2 nos lleva a una precisión del **99.94%**. Es el "trade-off" perfecto: preferimos una red pequeña que sea 99.7% inteligente y que use un milisegundo de cálculo para llegar a la perfección, que una red gigante de 10GB que intente ser perfecta sola.

---

## 💡 Reflexión Final
El éxito del proyecto no depende solo de la potencia de la GPU (aunque la Radeon 780M ha sido vital con su 80% de carga), sino de la **calidad de la representación semántica**. Pasar de "píxeles" a "relaciones geométricas" y de "relaciones" a "conceptos tácticos" es el camino para dominar los finales de 6 y 7 piezas con hardware doméstico.

---
**Fecha:** 20 de Marzo, 2026
**Ubicación:** docs/analysis/LECCIONES_APRENDIDAS_KPvKP.md
