# Endgames Interesantes de 4 Piezas

## Sugerencias del Jugador de Ajedrez

Endgames seleccionados por su complejidad táctica y relevancia práctica.

---

## 1. KBPvK - Rey+Alfil+Peón vs Rey

**Dificultad para el modelo:** ⭐⭐⭐ Media-Alta

### Características Tácticas:

**Fortress (Peón de Torre):**
```
♗ . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
♟ . . . . . . .
♚ . . . . . . ♔

Peón en columna a/h + rey defensor en esquina
→ TABLAS (fortress, el rey no puede salir)
```

**Peón Central:**
```
. . . ♗ . . . .
. . . . . . . .
. . . . . . . .
. . . ♟ . . . .
. . . . . . . .
. . . ♚ . . . .
. . . . . . . .
. . . ♔ . . . .

Peón en columna d/e + alfil ayuda
→ WIN (peón promociona)
```

### Complejidad para el Modelo:

- **Fortress detection:** Debe aprender que peón de torre = tablas
- **Color del alfil:** Alfil de color incorrecto no puede ayudar
- **Coordinación:** Alfil + rey deben coordinar para promocionar

**Accuracy esperado:** 99.0-99.5%

**Distribución WDL estimada:**
- Win: 60%
- Draw: 35% (fortress + alfil color incorrecto)
- Loss: 5%

---

## 2. KQvKR - Rey+Dama vs Rey+Torre

**Dificultad para el modelo:** ⭐⭐⭐⭐ Alta

### Características Tácticas:

**Dama gana generalmente:**
```
♕ . . . . . . .
. . . . . . . .
. . . ♜ . . . .
. . . ♚ . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
♔ . . . . . . .

Dama superior a Torre
→ WIN (pero requiere técnica)
```

**Posiciones de tablas (raras):**
```
. . . . . . . ♜
. . . . . . . ♚
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
♔ . . . . . ♕ .

Torre en esquina protegida por rey
→ DRAW (Philidor position)
```

### Complejidad para el Modelo:

- **Material advantage:** Dama > Torre pero no trivial
- **Defensive resources:** Torre puede defenderse en esquina
- **Zugzwang:** Muchas posiciones de zugzwang
- **Técnica requerida:** No es mate directo

**Accuracy esperado:** 98.5-99.0%

**Distribución WDL estimada:**
- Win: 85%
- Draw: 10%
- Loss: 5%

---

## 3. KBNvK - Rey+Alfil+Caballo vs Rey

**Dificultad para el modelo:** ⭐⭐ Media

### Características Tácticas:

**Mate en la esquina del color del alfil:**
```
♗ . . . . . . ♘
♚ . . . . . . .
. ♔ . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .

Rey negro en esquina del color del alfil
→ WIN (mate en ~33 movimientos)
```

**Difícil para humanos, fácil para el modelo:**
- Humanos: Requiere conocer la técnica (triangulación)
- Modelo: Solo necesita aprender geometría

### Complejidad para el Modelo:

- **Siempre gana:** No hay tablas (excepto ahogado)
- **Geometría clara:** Acorralar al rey en esquina correcta
- **Distancias críticas:** Alfil + caballo + rey coordinados

**Accuracy esperado:** 99.5-99.9%

**Distribución WDL estimada:**
- Win: 94%
- Draw: 1% (ahogado)
- Loss: 5%

---

## 4. KRvKP - Rey+Torre vs Rey+Peón ⭐ YA PLANEADO

**Dificultad para el modelo:** ⭐⭐⭐ Media-Alta

### Características Tácticas:

**Torre debe parar el peón:**
```
♖ . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . ♚ . . . .
. . . ♟ . . . .
♔ . . . . . . .

Torre lejos, peón cerca de promoción
→ DRAW/LOSS (peón promociona)
```

**Torre para el peón:**
```
. . . ♖ . . . .
. . . . . . . .
. . . . . . . .
. . . ♚ . . . .
. . . ♟ . . . .
. . . . . . . .
. . . . . . . .
♔ . . . . . . .

Torre cerca, puede parar
→ WIN (torre captura peón)
```

### Complejidad para el Modelo:

- **Timing crítico:** Torre debe llegar a tiempo
- **Promoción threat:** Peón cerca de rank 8
- **Asimetría total:** Torre vs Peón

**Accuracy esperado:** 99.5-99.7%

---

## 5. KBvKP - Rey+Alfil vs Rey+Peón

**Dificultad para el modelo:** ⭐⭐⭐ Media-Alta

### Características Tácticas:

**Alfil del color correcto:**
```
♗ . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . ♚ . . . .
. . . ♟ . . . .
♔ . . . . . . .

Alfil controla casilla de promoción
→ WIN/DRAW (depende de posición)
```

**Alfil del color incorrecto:**
```
. ♗ . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . ♚ . . . .
. . . ♟ . . . .
♔ . . . . . . .

Alfil no puede parar peón
→ LOSS (peón promociona)
```

### Complejidad para el Modelo:

- **Color del alfil:** Crítico para el resultado
- **Fortress:** Alfil puede crear fortress
- **Promoción:** Peón puede promocionar si alfil color incorrecto

**Accuracy esperado:** 99.0-99.5%

**Distribución WDL estimada:**
- Win: 40%
- Draw: 40%
- Loss: 20%

---

## 6. KNvKP - Rey+Caballo vs Rey+Peón

**Dificultad para el modelo:** ⭐⭐⭐ Media-Alta

### Características Tácticas:

**Caballo puede parar:**
```
. . . ♘ . . . .
. . . . . . . .
. . . . . . . .
. . . ♚ . . . .
. . . ♟ . . . .
. . . . . . . .
. . . . . . . .
♔ . . . . . . .

Caballo cerca, puede bloquear
→ WIN/DRAW
```

**Peón promociona:**
```
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . ♚ . . . .
. . . ♟ . . . .
♔ . . ♘ . . . .

Caballo lejos, peón promociona
→ LOSS
```

### Complejidad para el Modelo:

- **Movimiento único del caballo:** Saltos en L
- **Bloqueo:** Caballo puede bloquear peón
- **Timing:** Crítico llegar a tiempo

**Accuracy esperado:** 99.0-99.5%

---

## 7. KRvKN - Rey+Torre vs Rey+Caballo

**Dificultad para el modelo:** ⭐⭐⭐ Media-Alta

### Características Tácticas:

**Torre gana generalmente:**
```
♖ . . . . . . .
. . . . . . . .
. . . ♘ . . . .
. . . ♚ . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
♔ . . . . . . .

Torre captura caballo o da mate
→ WIN
```

**Caballo defiende (raro):**
```
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . ♚ ♘ . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
♔ . . . . . ♖ .

Caballo protege rey, torre no puede capturar
→ DRAW (temporal)
```

### Complejidad para el Modelo:

- **Material advantage:** Torre > Caballo
- **Defensive resources:** Caballo puede defender
- **Mate patterns:** Torre + rey vs rey + caballo

**Accuracy esperado:** 99.5-99.8%

**Distribución WDL estimada:**
- Win: 90%
- Draw: 5%
- Loss: 5%

---

## 8. KRvKB - Rey+Torre vs Rey+Alfil

**Dificultad para el modelo:** ⭐⭐⭐ Media-Alta

### Características Tácticas:

Similar a KRvKN pero:
- Alfil tiene más movilidad que caballo
- Alfil puede controlar diagonales largas
- Más posiciones de tablas que KRvKN

**Accuracy esperado:** 99.5-99.8%

**Distribución WDL estimada:**
- Win: 88%
- Draw: 7%
- Loss: 5%

---

## 9. KPvKP - Rey+Peón vs Rey+Peón ⭐ MUY INTERESANTE

**Dificultad para el modelo:** ⭐⭐⭐⭐ Alta

### Características Tácticas:

**Carrera de peones:**
```
♔ . . . . . . .
♟ . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . ♟
. . . . . . . ♚

Ambos peones corren a promoción
→ Depende de quién llega primero
```

**Bloqueo mutuo:**
```
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . ♔ . . . .
. . . ♟ . . . .
. . . ♟ . . . .
. . . ♚ . . . .
. . . . . . . .

Peones bloqueados
→ DRAW (ninguno puede avanzar)
```

**Oposición:**
```
. . . . . . . .
. . . . . . . .
. . . ♔ . . . .
. . . . . . . .
. . . ♚ . . . .
. . . ♟ . . . .
. . . . . . . .
. . . . . . . .

Concepto de oposición crítico
→ WIN/DRAW según oposición
```

### Complejidad para el Modelo:

- **Doble promoción:** Ambos peones pueden promocionar
- **Oposición:** Concepto abstracto de ajedrez
- **Zugzwang:** Muchas posiciones de zugzwang
- **Timing:** Quién llega primero a promoción
- **Bloqueo:** Peones pueden bloquearse

**Accuracy esperado:** 98.5-99.0%

**Distribución WDL estimada:**
- Win: 40%
- Draw: 40%
- Loss: 20%

---

## Ranking por Dificultad para el Modelo

### Fáciles (>99.5% accuracy esperado):

1. **KBNvK** - Siempre gana, geometría clara
2. **KRvKN** - Torre superior, pocos recursos defensivos
3. **KRvKB** - Similar a KRvKN
4. **KRvKP** - Timing claro, encoding v2 ayuda

### Medios (99.0-99.5% accuracy esperado):

5. **KBvKP** - Color del alfil crítico
6. **KNvKP** - Movimiento único del caballo
7. **KBPvK** - Fortress detection

### Difíciles (98.5-99.0% accuracy esperado):

8. **KQvKR** - Material advantage pero no trivial
9. **KPvKP** - Doble promoción, oposición, zugzwang

---

## Recomendación de Orden de Prueba

### Fase 1: Validación Básica
1. **KRRvK** (en progreso) - Validar 4 piezas fácil
2. **KBNvK** - Validar que siempre-gana funciona

### Fase 2: Asimetría y Táctica
3. **KRvKP** - Asimetría, timing crítico
4. **KRvKN** - Material advantage
5. **KRvKB** - Similar a KRvKN

### Fase 3: Complejidad con Peones
6. **KBvKP** - Color del alfil
7. **KNvKP** - Caballo vs peón
8. **KBPvK** - Fortress detection

### Fase 4: Máxima Complejidad
9. **KPvKP** - Doble peón, oposición
10. **KQvKR** - Material advantage sutil

---

## Valor para el Paper

### Endgames que Demuestran:

**Escalabilidad:**
- KRRvK, KBNvK - Fáciles, validan que funciona

**Táctica:**
- KRvKP, KRvKN, KRvKB - Timing y material advantage

**Conceptos Abstractos:**
- KPvKP - Oposición, zugzwang
- KQvKR - Técnica sutil

**Casos Especiales:**
- KBPvK - Fortress
- KBvKP, KNvKP - Color/movimiento específico

### Sugerencia para el Paper:

Probar **5-6 endgames representativos**:
1. KRRvK (fácil, validación)
2. KRvKP (asimétrico, táctico)
3. KBNvK (siempre gana, técnica)
4. KPvKP (complejo, conceptos abstractos)
5. KQvKR (material advantage sutil)

Esto demuestra que el encoding funciona para:
- ✅ Endgames fáciles
- ✅ Endgames tácticos
- ✅ Endgames con conceptos abstractos
- ✅ Endgames con material advantage sutil

---

## Conclusión

**Endgames más interesantes para probar:**
1. **KRvKP** - Ya planeado, excelente elección
2. **KPvKP** - Máxima complejidad, muy práctico
3. **KBPvK** - Fortress detection único
4. **KQvKR** - Material advantage sutil

**Todos son excelentes sugerencias** que demostrarían la robustez del encoding relativo. 🎯
