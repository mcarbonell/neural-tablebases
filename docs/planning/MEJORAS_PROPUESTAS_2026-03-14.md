# Mejoras Propuestas

Fecha original: 14 de marzo de 2026
Refresh documental: 20 de marzo de 2026
Tipo: backlog tecnico util, no plan comprometido

## Contexto

Estas ideas nacieron tras la etapa V4 + canonicas pawn-safe. Muchas siguen siendo buenas. Lo que cambio es el contexto: la linea activa del repo ya se apoya en KPvKP canonical `v5`, metadata mas rica y entrenamiento en marcha.

## Propuestas Que Siguen Fuertes

### 1. Entrenamiento streaming por shards

Sigue siendo probablemente una de las mejoras tecnicas mas importantes para escalar a datasets mayores sin depender de un `.npz` monolitico.

### 2. Exception maps end-to-end

Sigue encajando muy bien con la filosofia del proyecto:

- modelo compacto
- correccion por busqueda
- mapa pequeno para residuos dificiles

### 3. Reproducibilidad fuerte

Esto ya no es una mejora "nice to have". Deberia tratarse como requisito basico.

### 4. Tests y smoke-tests

Sigue siendo prioritario, sobre todo en:

- generacion parcial
- modos canonicos con peones
- metadata
- nombres `_partial_...`

### 5. Ergonomia de CLI

Todavia hay margen de mejora en defaults, seeds, y coherencia de nombres.

### 6. Benchmarks de evaluacion

Hace falta seguir mejorando la capa de evaluacion:

- accuracy por clase
- confusion matrix
- percentiles DTZ
- coste de busqueda por profundidad

### 7. Export y cuantizacion

Sigue siendo una pieza importante para el objetivo real de compresion utilizable.

## Lo Que Cambiaria En La Priorizacion

Si hoy hubiera que ordenar el backlog, yo lo pondria asi:

1. reproducibilidad y metadata consistentes
2. evaluacion fuerte y comparable
3. shards / training streaming
4. exception maps end-to-end
5. ergonomia de CLI
6. cuantizacion final

## Nota De Freshness

La referencia original a "tras V4 + canonicas" es historica. La linea activa ya no deberia describirse asi sin matizar que KPvKP se esta trabajando en la rama `v5`.

---

Estado actual: backlog tecnico util
