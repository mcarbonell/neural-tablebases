# Guía: Entrenamiento de la "Red Universal de Ajedrez V5"

## 🎯 Objetivo
Crear un único modelo de **2 MB** capaz de evaluar correctamente cualquier posición de 2, 3 y 4 piezas, integrando todo el "conocimiento fragmentado" (Damas, Peones, Alfil+Caballo, etc.) en un solo cerebro neuronal.

---

## 🏗️ 1. Arquitectura: El concepto del "Padding"
Para que una red pueda leer finales de 3 y 4 piezas indistintamente, el vector de entrada debe ser de tamaño fijo. 

*   **Configuración:** Usar siempre el tamaño de entrada de **5 piezas** (el máximo previsto).
*   **Implementación:** Si en el tablero solo hay 3 piezas, las posiciones 4 y 5 se rellenan con:
    *   `piece_type = 0` (Vacío).
    *   `coords = [0, 0]`.
*   **Resultado:** La red aprende que cuando ve ceros en esas ranuras, debe ignorar esas entradas. Esto permite que el modelo sea **retrocompatible**.

## 📊 2. Estrategia de Datos: El "Dataset Remix"
En lugar de entrenar solo con un archivo `.npz`, debemos crear un **DataLoader Multidataset**.

1.  **Carga Masiva:** Leer todos los archivos `.npz` generados (`KPvKP`, `KQvKR`, `KBNvK`, etc.).
2.  **Muestreo Balanceado:** En cada lote (batch) de 4096 posiciones, el 10% debe ser de `KPvKP`, el 10% de `KQvKR`, etc.
3.  **Prevención del Olvido:** Al ver todas las piezas en cada iteración, la red no pierde precisión en peones mientras aprende alfiles.

## 📈 3. Protocolo de Entrenamiento (Curriculum Learning)
Para acelerar la convergencia, no empezamos de cero. Seguimos esta cadena de mando:

1.  **Fase 1 (Cimientos):** Entrenar solo con **Reyes y Peones** (`KPvK`, `KPvKP`). La red aprende el movimiento básico y la importancia de la promoción.
2.  **Fase 2 (Alcance):** Cargar pesos de Fase 1 y añadir **Damas y Torres** (`KQvK`, `KRvK`). La red aprende las líneas largas.
3.  **Fase 3 (Técnica):** Añadir finales complejos (`KBNvK`, `KRvKN`).
4.  **Fase 4 (Consolidación Universal):** Entrenar con el **Remix Total** usando un Learning Rate muy bajo (`1e-5`) para fusionar todos los conocimientos sin "romper" nada.

## 🛠️ 4. Herramientas Necesarias
Para automatizar esto, necesitaremos desarrollar:
*   `src/universal_dataloader.py`: Capaz de leer múltiples datasets y mezclarlos al vuelo.
*   `src/merge_v5_metadata.py`: Para consolidar las estadísticas de todos los finales implicados.

---

## 🔭 Siguiente Paso sugerido
Una vez terminemos de generar los 141 finales, mi propuesta es crear la **primera versión de la Red Universal** usando los 10 finales más "educativos" que ya tenemos.

---
**Documento:** GUIA_ENTRENAMIENTO_UNIVERSAL_V5.md
**Fase:** Investigación y Arquitectura de Datos
