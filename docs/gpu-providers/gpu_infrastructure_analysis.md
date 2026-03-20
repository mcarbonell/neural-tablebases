# Análisis de Infraestructura GPU para Neural Tablebases

Este documento recoge la comparativa de opciones de alquiler de GPU realizada en marzo de 2026 para optimizar el entrenamiento de los modelos de compresión de piezas de ajedrez.

## 1. Escenario Actual del Proyecto
- **Fase:** Optimización y pulido de finales de 3 y 4 piezas (KPvKP, etc.).
- **Necesidades:** Entrenamientos rápidos de modelos MLP, datasets de tamaño manejable (Gigabytes).
- **Prioridad:** Agilidad de iteración y bajo coste por experimento.

## 2. Comparativa de Proveedores (Precios Marzo 2026)

| Proveedor | Modelo GPU | Precio/h | Características Clave |
| :--- | :--- | :--- | :--- |
| **OVH AI Notebook** | NVIDIA L4 (24GB) | 0,83 € | Cuadernos listos para usar, **Object Storage ilimitado**. |
| **OVH AI Notebook** | NVIDIA A100 (80GB) | 3,00 € | Máxima memoria para modelos grandes, almacenamiento incluido. |
| **RunPod (Community)**| NVIDIA RTX 4090 (24GB)| ~$0.35 - $0.50 | **Mejor relación potencia/precio** para modelos actuales. |
| **Lambda Labs** | NVIDIA H100 (80GB) | ~$2.30 | Referencia en gama alta "on-demand" (fuera de grandes clouds). |
| **Vast.ai** | Varios | <$0.30 | El más barato (marketplace), pero con fiabilidad variable. |
| **Google Cloud (Spot)**| NVIDIA L4 (24GB) | ~$0.15 - $0.25 | **Excelente para ML moderno**, muy barato en modo Spot. |
| **AWS (Spot)** | NVIDIA A10G (24GB) | ~$0.40 - $0.60 | Alta disponibilidad, pero costes de salida de datos (egress). |


## 3. Recomendaciones Estratégicas

### Fase A: Pulido de 3 y 4 piezas (Actual)
Para esta fase, las GPUs de consumo de gama alta son superiores en coste-efectividad.
- **Recomendación:** Usar **RunPod o Vast.ai** con instancias de **RTX 4090**.
- **Justificación:** La arquitectura de la 4090 es extremadamente eficiente para los modelos MLP que estamos usando. Un entrenamiento típico de 30-50 épocas costará menos de 0,50€.

### Fase B: Escalado a 5 y 6 piezas (Futuro)
Cuando el número de piezas aumenta, el cuello de botella se desplaza de la GPU al almacenamiento y la gestión de datos.
- **Recomendación:** Considerar **OVH** o soluciones con almacenamiento masivo y **GCP (Google Cloud)**.
- **Justificación:** Los datasets de 6 piezas pueden ocupar cientos de Gigabytes. 
    - **OVH:** El **Object Storage ilimitado** elimina la preocupación por los costes de almacenamiento y las tasas de transferencia (egress fees).
    - **Google Cloud (L4 Spot):** La GPU NVIDIA L4 en GCP es excepcionalmente eficiente en coste/rendimiento para inferencia y entrenamiento ligero. Si se usan Buckets de Google (GCS) en la misma región, el tráfico es gratuito.

### Fase C: Hiper-escaladores (GCP vs AWS)
Si decides usar Google o Amazon, la clave es el uso de **Spot Instances** (instancias que pueden ser reclamadas por el proveedor a cambio de un 60-90% de descuento).

| Característica | Google Cloud (GCP) | Amazon Web Services (AWS) |
| :--- | :--- | :--- |
| **GPU Recomendada** | **L4 (Ada Lovelace)** - 24GB | **G5 (A10G)** - 24GB |
| **Ventaja** | Mejor precio por TFLOPS en Spot. | Máxima fiabilidad y ecosistema. |
| **Punto Débil** | Disponibilidad de Spot variable. | Costes de red muy altos (Egress). |
| **Servicio Managed** | Vertex AI | SageMaker |

> [!WARNING]
> **Cuidado con el Egress:** AWS y GCP cobran significativamente por sacar datos de su red. Si entrenas allí, intenta mantener los datasets y los modelos resultantes dentro de sus respectivos buckets (S3/GCS) o usa **Cloudflare R2** para evitar estas tasas.


## 4. Resumen de Decisión
| **Si buscas...** | **Elige...** |
| :--- | :--- |
| **Mínimo coste por hora** | Vast.ai (RTX 4090/3090) |
| **Equilibrio y facilidad** | RunPod |
| **Escalado profesional (Spot)** | **Google Cloud (L4 Instances)** |
| **Privacidad EU y Datos Masivos** | OVH Cloud |
| **Máximo rendimiento (H100/A100)** | Lambda Labs |

---
**Fecha de análisis:** 16 de marzo de 2026  
**Nota:** Precios sujetos a variaciones de mercado y disponibilidad de subastas (spot instances).

---

## 5. Análisis de Opciones Gratuitas (Marzo 2026)
Para la fase de experimentación o para proyectos con presupuesto limitado, existen excelentes opciones para acceder a cómputo de GPU sin coste.

### A. Entornos de Notebooks (Completamente Gratuitos)
Son ideales para empezar, hacer prototipos y para entrenamientos de tamaño pequeño o mediano. No requieren tarjeta de crédito.

*   **Google Colab:** La opción más popular. Ofrece acceso gratuito a GPUs como las NVIDIA T4. Es una excelente manera de obtener una cantidad considerable de cómputo por semana sin coste alguno.
*   **Kaggle Notebooks:** Similar a Colab, Kaggle proporciona un entorno con GPUs gratuitas (generalmente NVIDIA P100 o T4) y una cuota semanal de aproximadamente 30 horas.
*   **Amazon SageMaker Studio Lab:** La alternativa de AWS a Colab. Ofrece cómputo y 15GB de almacenamiento persistente de forma gratuita, y no requiere una cuenta de AWS o tarjeta de crédito para empezar.

### B. Créditos Gratuitos en Plataformas Cloud
Estos proveedores ofrecen una cantidad de dinero en créditos para usar en sus servicios (incluyendo GPUs potentes) durante un tiempo limitado. Son perfectos para pruebas más intensivas.

*   **Google Cloud Platform (GCP):** Ofrece **$300 en créditos gratuitos** para cuentas nuevas, válidos por 90 días. Esto puede financiar entre 30 y 100 horas de uso en GPUs potentes.
*   **Microsoft Azure:** Proporciona **$200 en créditos gratuitos** para nuevos usuarios, válidos por 30 días.
*   **Amazon Web Services (AWS):** A través de programas como **AWS Activate** (para startups) o **AWS Educate** (para estudiantes) se pueden conseguir importantes cantidades de créditos.

### C. Recomendación para este Proyecto
1.  **Empezar con Google Colab:** Es la forma más rápida y sencilla de realizar experimentos y validaciones sin ningún compromiso.
2.  **Para pruebas intensivas, usar GCP:** Activar la prueba gratuita de **Google Cloud (GCP)** para aprovechar los $300 de crédito. Es la oferta de bienvenida más generosa y permitirá validar el rendimiento en hardware más avanzado antes de comprometerse a un proveedor de pago.
