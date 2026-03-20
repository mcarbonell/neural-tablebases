# Visión: Computación Distribuida para la Resolución de Finales

**Concepto:** Un "SETI@home" para el ajedrez.

## 1. La Barrera de los 7 y 8 Piezas

La generación y entrenamiento de redes neuronales para finales de 3, 4 y 5 piezas es computacionalmente factible para un individuo o un pequeño grupo de investigación con acceso a hardware moderno. Sin embargo, a medida que nos acercamos a los 6, 7 y 8 piezas, el espacio de estados explota exponencialmente.

*   **Syzygy de 7 piezas:** ~16.7 Terabytes.
*   **Syzygy de 8 piezas:** Estimado en Petabytes.

Generar datasets de este tamaño, y entrenar redes neuronales sobre ellos, supera la capacidad de cualquier entidad individual. El coste de almacenamiento y cómputo se convierte en la barrera fundamental para "resolver" completamente el ajedrez mediante compresión neuronal.

## 2. La Solución: La Comunidad

La comunidad de ajedrez y de entusiastas de la computación es una de las más activas y colaborativas del mundo. Proyectos como **Stockfish (Fishnet)** y **SETI@home** han demostrado que es posible agregar una cantidad masiva de recursos computacionales donados por voluntarios.

**La visión es crear un sistema similar para las Neural Tablebases.**

### Componentes Clave del Sistema

1.  **Cliente Ligero:** Un pequeño programa que cualquier persona puede descargar y ejecutar en su ordenador. Este cliente se conectaría a un servidor central para recibir "unidades de trabajo".

2.  **Unidades de Trabajo (Work Units):**
    *   **Generación de Datos:** Una unidad de trabajo podría ser un subconjunto del espacio de posiciones de un final de 7 piezas. El cliente generaría las posiciones, las evaluaría con una tablebase Syzygy (si el voluntario tiene el disco duro para ello) y subiría el lote de datos (`posición -> resultado`).
    *   **Entrenamiento de Redes:** Otra forma de contribución sería recibir un subconjunto del dataset y una arquitectura de modelo, y realizar un número determinado de épocas de entrenamiento. El cliente devolvería los pesos actualizados del modelo.

3.  **Servidor de Orquestación:** Un servidor central que gestionaría:
    *   La distribución de las unidades de trabajo.
    *   La agregación de los datos generados.
    *   La combinación de los gradientes o pesos de los diferentes voluntarios para actualizar el modelo global (usando técnicas como *Federated Averaging*).
    *   Un "leaderboard" o sistema de reconocimiento para gamificar la contribución y agradecer a los voluntarios.

## 3. Beneficios

*   **Escalabilidad Masiva:** Permite abordar la complejidad de los finales de 7 y 8 piezas que de otro modo serían inaccesibles.
*   **Coste Cero (o Casi Cero):** El proyecto podría avanzar sin necesidad de una inversión masiva en infraestructura de hardware.
*   **Participación Comunitaria:** Involucra a la comunidad directamente en el legado del proyecto, creando un sentido de propiedad compartida sobre el objetivo de "resolver" el ajedrez.
*   **Eficiencia Energética:** Aprovecha ciclos de CPU/GPU que de otro modo estarían inactivos en miles de ordenadores personales.

## 4. Hoja de Ruta (Visión a Largo Plazo)

*   **Fase 1: Prueba de Concepto (PoC):** Desarrollar un cliente y servidor básicos para un final ya conocido (ej. KPvKP) y demostrar que los datos generados por dos clientes distintos pueden ser combinados para entrenar un modelo con éxito.
*   **Fase 2: Lanzamiento para 6 Piezas:** Invitar a la comunidad a participar en la generación del dataset completo de 6 piezas. Sería el primer gran test a escala.
*   **Fase 3: Asalto a las 7 Piezas:** Con la experiencia de la fase anterior, iniciar el proyecto de generación y entrenamiento para los finales de 7 piezas, el "santo grial" actual de las tablebases.

Esta visión transforma un desafío computacional insuperable en un proyecto comunitario global, alineado con la tradición de código abierto y colaboración que define al mundo del ajedrez por ordenador.
