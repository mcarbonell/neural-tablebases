# Resultados: Formas Canonicas

Fecha original: 14 de marzo de 2026
Refresh documental: 20 de marzo de 2026
Tipo: resultado historico valido, no snapshot operativo del repo

## Estado Del Documento

Este documento sigue siendo util para entender por que las formas canonicas pasaron a ser parte del pipeline principal. Lo que ya no es actual es su framing de "siguiente paso", porque desde entonces el repo ya genero datasets canonicos adicionales y la linea activa se desplazo a KPvKP canonical `v5`.

## Resumen Ejecutivo

Las formas canonicas reducen el dataset aproximadamente un 50% en los casos validados y mejoran la eficiencia de entrenamiento sin romper el rendimiento practico. Ese hallazgo sigue siendo uno de los resultados estructurales mas importantes del proyecto.

## Lo Que Sigue Vigente

- `src/generate_datasets_parallel.py` es el generador principal.
- `--enumeration permutation` sigue siendo la opcion correcta para un recorrido exhaustivo.
- `--canonical --canonical-mode auto` sigue siendo el default conceptual recomendado.
- Los sidecars `*_metadata.json` son parte importante de la reproducibilidad.
- La intuicion central del documento se mantiene: reducir simetrias es una mejora real, no solo una curiosidad tecnica.

## Implementacion

### Algoritmo de formas canonicas

- `dihedral`: 8 simetrias. Seguro solo sin peones.
- `file_mirror`: 2 simetrias. Pawn-safe.
- `auto`: usa `file_mirror` si hay peones y `dihedral` en caso contrario.
- Con `--enumeration permutation`, basta filtrar representantes canonicos para obtener un conjunto exacto.
- Con `--enumeration combination`, el recorrido es legacy y no exhaustivo. Sirve para smoke-tests, no como dataset final.

### Generador principal

- `src/generate_datasets_parallel.py` es el camino activo.
- `src/generate_datasets_parallel_canonical.py` permanece por compatibilidad.
- El generador escribe metadata automatica para dataset, indexacion y configuracion.

## Resultados Historicos Validados

| Endgame | Posiciones Originales | Posiciones Canonicas | Reduccion |
|---------|----------------------|----------------------|-----------|
| KQvK | 64,631 | 32,060 | 50.4% |
| KRvK | 70,672 | 34,825 | 50.7% |
| KPvK | 74,984 | 37,046 | 50.6% |
| Promedio | 70,096 | 34,644 | 50.6% |

## Comparacion Historica

| Endgame | Original Acc | Canonico Acc | Delta | Original DTZ MAE | Canonico DTZ MAE | Delta |
|---------|--------------|--------------|-------|------------------|------------------|-------|
| KQvK | 99.94% | 100.00% | +0.06% | 0.64 | 0.47 | -0.17 |
| KRvK | 100.00% | 100.00% | 0.00% | 1.00 | 0.78 | -0.22 |
| KPvK | 99.88% | 99.57% | -0.31% | 0.06 | 0.05 | -0.01 |

## Lectura Correcta En 2026-03-20

Este documento demuestra que:

1. la reduccion canonica funciona
2. el coste en accuracy es pequeno o nulo en los casos validados
3. el pipeline canonico merece ser el camino por defecto

Este documento no debe leerse como:

1. el estado operativo mas reciente del repositorio
2. la lista actual de proximos pasos
3. la ultima palabra sobre KPvKP, donde la linea activa ya esta en `v5`

## Impacto En El Proyecto

- Hizo viable estandarizar el pipeline con formas canonicas.
- Encaja bien con el generador paralelo y la metadata reproducible.
- Prepara el terreno para combinar modelo pequeno + correccion por busqueda + exception maps.

## Nota Sobre Los Siguientes Pasos

La frase antigua "aplicar a KRRvK" ya no refleja el workspace actual. El repo ya contiene `data/KRRvK.npz` y `data/KRRvK_canonical.npz`. El paso relevante ahora no es generar por primera vez, sino decidir como reentrenar, comparar o integrar esos datasets en la linea vigente.

---

Estado actual: historico pero vigente como resultado metodologico
