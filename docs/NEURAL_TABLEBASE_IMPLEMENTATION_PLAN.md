# Neural Endgame Oracle: Plan de Implementacion

## Resumen

- Documento objetivo: plan de implementacion para un `oracle de finales` ultracompacto.
- Objetivo principal: mejorar la fuerza practica del motor con conocimiento de finales muy comprimido.
- Objetivo secundario: medir cuanta calidad se pierde frente a Syzygy y cuanto cuesta recuperarla con un residual.
- Alcance v1: `4-7 piezas` usando Syzygy como fuente de verdad.
- Alcance fuera de v1: `8 piezas`, integracion UCI final, y exactitud formal estilo tablebase pura.
- Modos de entrega:
  - `neural-only`
  - `hybrid residual` = red + tabla sparse de excepciones

## Principios del experimento

- El criterio de exito no es exactitud matematica, sino `fuerza practica por byte`.
- La salida minima util para el motor no es solo `WDL`; hace falta una senal de progreso.
- Se usara `WDL5` de Syzygy como cabeza principal y `signed DTZ bucket` como cabeza de progreso.
- No se intentara predecir `DTM exacto` en v1.
- Se trabajara por `familia material`, no con un modelo unico gigante para todos los finales.
- Toda familia material se evaluara por separado antes de decidir si merece integrarse al motor.

## Artefactos e interfaces

### Dataset record

Cada registro del dataset debe contener:

- `canonical_position_key`
- `material_family_id`
- `side_to_move`
- `rule50_bucket`
- `ep_square`
- encoding de piezas por casilla
- `wdl5_target`
- `dtz_raw`
- `dtz_bucket`

### Modelo exportado

Cada artefacto de modelo debe contener:

- `family_id`
- version de features
- arquitectura usada
- pesos cuantizados `int8`
- calibracion de logits
- metadatos de evaluacion

### Residual

Cada entrada del residual debe ser:

- `canonical_position_key -> (wdl5_exact, dtz_raw_exact)`

### API runtime

Interfaz objetivo:

```text
probe_endgame_oracle(position) -> {
  wdl5,
  progress_score,
  source
}
```

Donde `source` puede ser:

- `neural`
- `residual`
- `fallback`

## Estructura del proyecto experimental

Crear un workspace standalone en:

- `experiments/chess_endgame_oracle/`

Crear un harness de benchmark en:

- `benchmarks/endgame_oracle/`

Submodulos internos previstos:

- `data/`
  - enumeracion de posiciones
  - canonicalizacion
  - serializacion binaria
- `train/`
  - modelos
  - losses
  - samplers
  - hard-negative mining
- `eval/`
  - barridos exhaustivos
  - metricas offline
  - comparativas neural-only vs hybrid
- `runtime/`
  - carga de artefactos
  - inferencia int8
  - lookup de residual
- `tools/`
  - empaquetado
  - reportes
  - conversion de artefactos

## Fuente de verdad y cobertura

- Fuente de verdad v1: `python-chess` + Syzygy.
- No entrenar desde FEN textual como fuente primaria; usar registros binarios compactos por familia material.
- Cobertura v1:
  - `4-5 piezas`: todas las familias materiales
  - `6 piezas`: top `20` familias por frecuencia en un corpus de finales de self-play
  - `7 piezas`: top `10` familias por la misma metrica
- Para `7 piezas`, el barrido exhaustivo completo solo se aplicara a las `2` familias mas prometedoras despues de la validacion por muestra.

## Canonicalizacion y semantica

- Mantener explicitamente:
  - `side_to_move`
  - `rule50_bucket`
  - `ep_square`
- Finales sin peones:
  - reducir por simetrias completas del tablero
- Finales con peones:
  - usar solo simetrias que preserven la direccion legal del peon
  - no usar transformaciones que rompan la semantica de avance
- Posiciones con `castling rights` quedan fuera de cobertura v1 y van a `fallback`.
- El pipeline debe incluir tests que demuestren que la canonicalizacion no altera las etiquetas Syzygy.

## Targets del modelo

### Cabeza principal

- `WDL5` de Syzygy:
  - `-2`
  - `-1`
  - `0`
  - `1`
  - `2`

Razon:

- preserva mejor la semantica relevante que colapsar todo a `WDL3`
- evita perder informacion sobre blessed y cursed wins/losses

### Cabeza secundaria

- `signed DTZ bucket`

Buckets propuestos:

- `0`
- `1`
- `2`
- `3`
- `4-5`
- `6-9`
- `10-15`
- `16-25`
- `26-40`
- `41-60`
- `61+`

Razon:

- fuerza progreso practico
- evita pedir exactitud innecesaria en `DTM`
- es mas estable y mas util para la regla de 50 movimientos

## Features de entrada

Input base:

- pieza por casilla
- tipo y color de cada pieza presente
- `side_to_move`
- `rule50_bucket`
- `ep_square`

Representacion:

- encoding sparse o semi-sparse por familia material
- no usar una sola plantilla fija si la familia tiene menos piezas y puede codificarse de forma mas compacta

Decision v1:

- mantener una implementacion simple primero
- luego comparar contra una version mas agresivamente compacta por familia

## Arquitecturas a probar

Tres familias base por cada familia material:

- `small`: `128-64`
- `medium`: `256-128`
- `large`: `384-192`

Reglas:

- usar capas densas simples en v1
- activacion `ReLU`
- exportacion cuantizada `int8`
- un modelo separado por familia material

Criterio de seleccion:

- elegir el modelo mas pequeno que cumpla los umbrales de aceptacion de esa familia

## Entrenamiento

### Muestreo

- muestreo estratificado por clase `WDL5`
- sobremuestreo de clases raras cuando sea necesario
- reinyectar sistematicamente hard negatives

### Loss

Loss total:

- `cross-entropy(WDL5)`
- `ordinal cross-entropy(DTZ bucket)`
- termino de calibracion de confianza

No incluir:

- `DTM exacto`
- regresion cruda de DTZ en v1

### Hard-negative mining

Despues de cada sweep parcial:

- recopilar posiciones con `WDL flip`
- recopilar posiciones con confianza baja
- recopilar posiciones no tablas con error de `DTZ bucket` alto
- reinyectarlas con mayor peso en la siguiente fase de entrenamiento

## Residual hibrido

Construccion del residual al final de cada familia:

- guardar toda posicion con `WDL5` mal clasificado
- guardar toda posicion no tablas con error de `DTZ` superior a `2` buckets

Politica runtime:

1. si la familia no esta cubierta, usar `fallback`
2. si existe entrada en residual, devolver residual
3. si no existe, inferir con la red

Objetivo del residual:

- recuperar exactitud funcional donde mas importa
- medir cuanto cuesta corregir los errores de la red
- decidir si el hibrido compensa frente a Syzygy para cada familia

## Politica de uso en el motor

Activar el oracle solo cuando:

- la familia material esta soportada
- no hay `castling rights`
- la posicion entra en la cobertura semantica del modelo

Orden de decision:

1. preservar `WDL5`
2. priorizar `progress_score`
3. romper empates con la evaluacion base del motor

Regla de despliegue:

- si una familia no mejora fuerza practica o conversion, no se integra

## Metricas obligatorias

### Offline

- `WDL5 error rate`
- `catastrophic flips`
- `DTZ bucket MAE`
- calibration error
- tamano del modelo
- tamano del residual
- tamano total del paquete
- latencia por probe
- throughput

### Practicas

- porcentaje de conversion desde posiciones ganadas
- porcentaje de hold desde posiciones tablas
- plies medios a conversion
- Elo en self-play contra baseline

## Validacion

### Tests unitarios

- invariancia de canonicalizacion
- paridad entre etiquetas de dataset y probes directos a Syzygy
- paridad `fp32` vs `int8`
- round-trip exacto de serializacion del residual

### Evaluacion exhaustiva

- `4-5 piezas`: barrido exhaustivo completo por familia
- `6 piezas`: barrido exhaustivo completo para las `20` familias seleccionadas
- `7 piezas`: validacion por muestra grande para todas; barrido exhaustivo solo para las `2` mejores candidatas al hibrido

### Benchmarks practicos

- suite de `10k` posiciones de finales ganados, tablas y perdidos
- self-play baseline vs oracle con mismo control de tiempo
- comparacion:
  - baseline
  - neural-only
  - hybrid residual

## Umbrales de aceptacion

### Neural-only

- `4-5 piezas`: `WDL5 error <= 0.05%`
- `6 piezas`: `WDL5 error <= 0.30%`
- `7 piezas`: `WDL5 error <= 0.50%` en holdout grande

### Hybrid residual

- `0` errores `WDL5` en toda familia con residual construido
- paquete hibrido total `<= 10%` del tamano Syzygy equivalente para ese subconjunto

### Practico

- `+20 Elo` en suite de finales o
- `+10 Elo` global en self-play frente al baseline

Si una familia no cumple esto, no se promueve a integracion.

## Plan por fases

### Semana 1

- pipeline de enumeracion
- canonicalizacion
- esquema binario
- tests de verdad de etiquetas

### Semanas 2-3

- entrenamiento y barrido exhaustivo de `4-5 piezas`
- primer benchmark `neural-only`
- seleccion inicial de arquitectura por familia

### Semanas 4-6

- seleccion de familias `6 piezas`
- entrenamiento con hard-negative mining
- construccion y medicion del residual hibrido

### Semanas 7-9

- familias `7 piezas`
- validacion por muestra grande
- residual exhaustivo para las `2` mejores familias

### Semanas 10-12

- empaquetado runtime
- benchmark de self-play
- decision de promocion por familia

## Hardware asumido

Configuracion objetivo:

- `1x RTX 4090 24GB`
- `128GB RAM`
- `4TB NVMe`

Fallbacks:

- con menos GPU, multiplicar tiempos por `2x-3x`
- sin GPU, limitar v1 a `4-5 piezas`

## Supuestos y decisiones cerradas

- El experimento optimiza `fuerza practica por byte`.
- `4-7 piezas` es el alcance v1.
- `8 piezas` queda fuera hasta demostrar valor en `6-7 piezas`.
- La integracion con el motor se deja en formato intermedio, no ligada aun a C++ ni Rust.
- `DTM exacto` queda fuera de v1.
- `WDL5 + signed DTZ bucket` es la combinacion de targets elegida.
- El despliegue se decide por familia material, no como bloque monolitico.

