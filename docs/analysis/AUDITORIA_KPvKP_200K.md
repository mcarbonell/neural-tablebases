# Auditoría de Gran Escala: Perfección en KPvKP (4 Piezas)

## 📌 Resumen de la Evaluación
Evaluación exhaustiva de la red neuronal `mlp_kpvkp_v5` tras 136 épocas de entrenamiento, auditada contra el oráculo de Syzygy sobre una muestra aleatoria de **200,000 posiciones**.

### 📊 Datos Globales
*   **Fecha:** 20 de Marzo, 2026
*   **Modelo de Red:** MLP (68 inputs v5, ~450k parámetros)
*   **Precisión Validation (D0):** 99.58% (Teórica)
*   **Muestra Auditada:** 200,000 posiciones legales únicas.

---

## 📈 Rendimiento con Corrección por Búsqueda (Search)

El uso de búsquedas de profundidad baja (Minimax superficial) compensa los errores locales de la red neuronal, especialmente en finales tácticos de peones.

| Configuración | Precisión (%) | Fallos / 200,000 | Tasa de Mejora |
| :--- | :---: | :---: | :--- |
| **NN Directa (D0)** | **99.6035%** | 793 | - |
| **Búsqueda D1** | **99.8905%** | 219 | 72% de errores corregidos |
| **Búsqueda D2** | **99.9475%** | **105** | **86% de errores corregidos** |

> [!IMPORTANT]
> A profundidad 2 (D2), la red solo comete **1 error cada 1,900 posiciones**. Estamos a un paso de la perfección absoluta para finales de 4 piezas.

---

## 🔍 Análisis de Patrones: El "Horizonte de Transformación"
Tras analizar los 105 fallos remanentes, se ha identificado un patrón táctico recurrente que explica por qué la red (e incluso la búsqueda profunda) puede fallar:

### 1. La "Dama Inútil" (KPvKP $\to$ KQvKP Tablas)
Existen posiciones donde un bando corona una Dama, pero el Peón rival está en 7ª fila y el Rey defensor está tan bien posicionado que la Dama no puede evitar el jaque continuo o el ahogado. La red de `KPvKP` debe "memorizar" que esa futura coronación no es ganadora, un cálculo que técnicamente pertenece a la tabla de `KQvKP` pero que la red de peones debe predecir.

### 2. La "Dama Perdida" (KQvKQ Victoria)
En carreras de peones simétricas donde ambos coronan, el bando que corona primero con jaque suele ganar la Dama del rival o dar mate. Esta transición a `KQvKQ` está fuera del dominio de entrada de la red de peones y requiere que la red haya "comprimido" el resultado final de la carrera sin tener acceso a la posición post-coronación.

---

## 🛠️ Próximos Pasos Técnicos para el Proyecto
Para alcanzar el 100.00% (cero errores) en las 4 piezas, la estrategia será:

1.  **Extensión del Entrenamiento:** Dejar que las pérdidas de validación bajen del 1.0% actual, acercándose a los niveles que vimos en 3 piezas (99.93% raw).
2.  **Hard Mining (Overfitting Loop):** Usar los 105 fallos detectados en esta auditoría como ejemplos de peso extra para forzar a la red a "aprender de memoria" los Horizontes de Transformación más complejos.
3.  **Cascada de Oráculo (Cascade Oracle):** Mantener el fallback a Syzygy/Otras Redes para cualquier posición que ya no sea puramente `KPvKP`.

---
**Validador Utilizado:** `src/verify_search_correction.py` con Fallback de Cascada Activo.
