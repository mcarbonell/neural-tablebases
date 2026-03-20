# Vision: Computacion Distribuida Para La Resolucion De Finales

Tipo: vision a largo plazo, no roadmap operativo inmediato
Refresh documental: 20 de marzo de 2026

## Idea Central

Un sistema tipo SETI@home o Fishnet para Neural Tablebases: repartir generacion de datos, verificacion o incluso entrenamiento entre muchos voluntarios.

## Por Que Esta Vision Tiene Sentido

Para 3, 4 e incluso parte de 5 piezas, un flujo individual todavia es manejable. A medida que el proyecto apunte a 6, 7 y 8 piezas, el cuello de botella pasara a ser claramente la combinacion de:

- almacenamiento
- ancho de banda
- tiempo total de computo
- orquestacion reproducible

Ahí la computacion distribuida deja de ser "idea bonita" y empieza a parecer requisito serio.

## Como Debe Leerse Este Documento Hoy

Este no es el plan de trabajo inmediato del repo. Es una vision estrategica de largo plazo. El proyecto todavia esta en una fase donde:

- el pipeline local y semilocal importa mas que una red distribuida completa
- la correccion documental, la reproducibilidad y la consolidacion de KPvKP canonical V5 tienen mas prioridad
- antes de distribuir computo a gran escala, conviene estabilizar el formato de datos, metadata, checkpoints y verificacion

## Arquitectura Posible

### Cliente ligero

Un cliente que reciba unidades de trabajo acotadas y devuelva resultados verificables.

### Tipos de work units

- generacion de muestras o shards
- verificacion de datasets
- evaluacion de modelos
- eventualmente entrenamiento parcial o agregacion federada

### Servidor de orquestacion

- reparto de unidades
- validacion de resultados
- deduplicacion
- leaderboard o atribucion
- reintentos y tolerancia a nodos poco fiables

## Problema Real A Resolver Antes

Antes de pensar en una red distribuida real, el proyecto necesita una especificacion solida de:

1. formato de shard
2. metadata obligatoria
3. claves estables de posicion canonica
4. versionado de encoding
5. validacion y firma de resultados

Sin eso, una plataforma distribuida multiplicaria el desorden en lugar de resolverlo.

## Fases Razonables

1. Consolidar pipeline local:
   canonical, metadata, ONNX, search correction, evaluacion reproducible
2. Introducir shards estables y entrenamiento/lectura por shards
3. Probar orquestacion simple en una sola maquina o LAN
4. Escalar a voluntarios o nodos externos

## Valor Del Documento

Esta vision sigue siendo valiosa porque define una direccion ambiciosa y coherente con el proyecto. Lo importante es no confundirla con la prioridad de esta semana.

---

Estado actual: vision de largo plazo
