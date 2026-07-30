# Diseño definitivo del caso de estudio

El objetivo experimental será demostrar que:

> **La definición del dataset, el control de redundancia y el régimen de particionamiento basado en similitud alteran el rendimiento aparente, su variabilidad y la interpretación de la generalización, incluso cuando la representación y los modelos permanecen constantes.**

No agregaremos más datasets, tasks ni benchmarks externos.

## Variables que mantendremos constantes

* un único endpoint, AMP classification;
* una representación generada con Sylphy;
* cuatro clasificadores convencionales;
* un procedimiento fijo de selección de hiperparámetros;
* las mismas métricas;
* el mismo esquema general de train/validation/test.

## Variables que estudiaremos

* construcción de la clase negativa;
* control de duplicados y redundancia;
* partición estratificada aleatoria;
* particiones sequence-similarity-aware con diferentes umbrales;
* variabilidad entre seeds;
* prevalencia y baseline de AUPRC de cada test set.

---

# Roadmap completo desde el único CSV pivote

# Fase 1. Congelar y caracterizar el CSV raw

## 1.1. Preservación

El archivo raw no se modifica nunca.

Debemos registrar:

* nombre;
* tamaño;
* número de filas y columnas;
* hash SHA-256;
* fecha de incorporación;
* columnas originales;
* encoding;
* separador;
* número de secuencias únicas.

El output será:

```text
metadata/raw_file_manifest.json
```

## 1.2. Normalización determinista

El script debe:

1. convertir secuencias a mayúsculas;
2. eliminar espacios y caracteres de formato;
3. conservar el valor original en una columna separada;
4. validar el alfabeto;
5. calcular longitud;
6. identificar secuencias vacías;
7. identificar duplicados exactos;
8. detectar etiquetas contradictorias;
9. crear un identificador estable derivado de la secuencia.

Por ejemplo:

```text
sequence_id = SHA256(normalized_sequence)
```

Eso evita depender del orden de las filas.

## 1.3. Tabla de control de calidad

Debemos generar automáticamente:

| Control                 | Resultado |
| ----------------------- | --------: |
| Filas originales        |         N |
| Secuencias válidas      |         N |
| Secuencias únicas       |         N |
| Duplicados exactos      |         N |
| Secuencias no canónicas |         N |
| Secuencias vacías       |         N |
| Conflictos de etiqueta  |         N |
| Longitud mínima         |         N |
| Longitud máxima         |         N |
| Longitud mediana        |         N |

Este resumen terminará alimentando directamente Methods, Supplementary Material y la respuesta a los revisores.

---

# Fase 2. Definir correctamente el task AMP

Esta es la **primera decisión crítica**. No debemos entrenar nada hasta resolverla.

El CSV tiene columnas binarias de actividades, pero debemos establecer qué significa exactamente un `0`.

## Escenario A: cero experimental

Si `0` significa que una secuencia fue evaluada y reportada explícitamente como no antimicrobiana:

* `1` = positive AMP;
* `0` = experimentally supported negative.

Ese sería el escenario ideal.

## Escenario B: ausencia de anotación

Si `0` significa solamente que no existe una anotación AMP:

* no podemos llamarlo automáticamente “negative AMP”;
* debemos describir el task como:

> AMP-annotated versus non-AMP-annotated peptide classification

o construir un conjunto negativo más estricto utilizando la información disponible.

## Propuesta defensiva

Crear dos categorías internas:

* **strict positive**: actividad AMP explícitamente registrada;
* **candidate negative/background**: sin actividad AMP registrada bajo las reglas definidas.

Si existen evidencias explícitas de negativos, se utilizan como análisis principal. Si no existen, el manuscrito debe reconocer claramente la naturaleza del background set y evitar tratarlo como inactividad biológica demostrada.

Esto responde directamente a una de las preocupaciones más importantes de los revisores: la construcción artificial del conjunto negativo puede dominar el rendimiento aparente en AMP prediction. 

## Output de esta fase

```text
data/processed/amp_task.csv
```

Columnas mínimas:

```text
sequence_id
sequence
amp_label
negative_definition
sequence_length
canonical_sequence
raw_row_id
```

---

# Fase 3. Caracterización de los sesgos del dataset

Antes del modelado debemos estudiar diferencias entre positivos y negativos en:

* longitud;
* composición aminoacídica;
* carga neta aproximada;
* hidrofobicidad;
* proporción de residuos cargados;
* proporción de residuos hidrofóbicos.

Esto no implica agregar un nuevo caso de estudio. Es la caracterización necesaria para demostrar que entendemos las propiedades del dataset y la potencial facilidad del problema.

Outputs:

```text
results/summaries/class_distribution.csv
results/summaries/length_summary.csv
results/summaries/physicochemical_summary.csv
```

En el main text bastará una tabla compacta. Las distribuciones completas pueden quedar en Supplementary Material.

---

# Fase 4. Control de duplicación y redundancia con BioSieve

Aquí conviene separar dos problemas que antes estaban mezclados.

## 4.1. Duplicados exactos

Las secuencias idénticas deben consolidarse antes de particionar.

Reglas:

* misma secuencia y misma etiqueta: conservar una instancia;
* misma secuencia con etiquetas incompatibles: marcar conflicto;
* no resolver conflictos mediante mayoría automáticamente;
* excluir o analizar separadamente los conflictos, documentando la decisión.

## 4.2. Redundancia por similitud

BioSieve debe convertirse en la fuente oficial de:

* similitud o distancia;
* agrupaciones;
* componentes relacionados;
* validación posterior de los splits.

Para cada ejecución debemos registrar:

* algoritmo;
* versión;
* parámetros;
* tipo de alineamiento o distancia;
* identidad mínima;
* cobertura mínima;
* tratamiento de secuencias de distinta longitud;
* criterio para formar grupos.

## 4.3. No confundir redundancia con eliminación

No necesariamente tenemos que eliminar todas las secuencias similares.

La estrategia principal puede ser:

> conservar las secuencias, pero impedir que grupos altamente similares aparezcan en particiones distintas.

Esto preserva más información y evita que el procedimiento de selección de representantes introduzca un sesgo adicional.

Podemos construir una vista redundancia-reducida como análisis secundario, pero no debería convertirse automáticamente en la única versión del dataset.

---

# Fase 5. Auditoría de similitud y selección de umbrales

No escogeremos umbrales mirando cuál produce los resultados más interesantes.

Primero debemos ejecutar una auditoría con candidatos como:

```text
90%, 80%, 70%, 60%, 50%
```

y, para cada umbral, estudiar:

* número de grupos;
* tamaño de los grupos;
* grupo máximo;
* secuencias aisladas;
* grupos con ambas clases;
* posibilidad de construir train/validation/test;
* balance de clase alcanzable;
* máxima similitud entre particiones.

Para péptidos cortos, la identidad debe acompañarse de una condición de cobertura. Una coincidencia parcial corta no debería bastar para declarar dos secuencias equivalentes.

## Criterio para escoger los umbrales finales

Los umbrales se seleccionarán por:

* interpretabilidad;
* viabilidad del split;
* tamaño suficiente de cada conjunto;
* número suficiente de positivos y negativos;
* ausencia de grupos cruzando particiones;
* no colapso del dataset en unos pocos componentes gigantes.

No se seleccionarán por AUPRC.

## Resultado esperado

Probablemente terminaremos con tres regímenes:

* `SIM90`: separación de secuencias casi duplicadas;
* `SIM70`: generalización intermedia;
* `SIM50` o `SIM60`: separación más estricta.

Los valores definitivos dependerán de la estructura real del dataset.

---

# Fase 6. Construcción de las particiones

## Régimen de referencia

### Stratified random split

Será el benchmark convencional:

```text
70% train
15% validation
15% test
```

o `80/10/10` si el tamaño del dataset lo justifica.

La prevalencia debe quedar aproximadamente conservada.

## Regímenes sequence-similarity-aware

Para cada umbral:

1. construir grupos con BioSieve;
2. asignar cada grupo completo a una única partición;
3. aproximar los tamaños objetivo;
4. aproximar el balance de clases;
5. impedir cualquier división interna del grupo;
6. ejecutar validaciones posteriores.

## Validaciones obligatorias

Cada split debe pasar automáticamente:

* ninguna secuencia repetida entre particiones;
* ningún grupo cruzando particiones;
* train, validation y test no vacíos;
* presencia de ambas clases;
* prevalencia registrada;
* máxima similitud cross-split bajo el criterio declarado;
* conteos reproducibles usando el mismo seed.

Cada archivo de split debe contener:

```text
sequence_id
partition
similarity_regime
similarity_threshold
group_id
seed
```

## Número de seeds

Usaría **20 seeds** como objetivo inicial.

Eso permitirá estimar de forma más estable:

* mediana;
* IQR;
* intervalos bootstrap;
* variabilidad entre particiones.

Cinco seeds serían demasiado pocos para defender con fuerza el análisis de dispersión.

---

# Fase 7. Representación fija con Sylphy

Para aislar el efecto del dataset y del particionamiento, usaría **una única representación congelada**.

Debemos fijar antes de comenzar:

* modelo;
* checkpoint;
* versión;
* capa;
* pooling;
* tratamiento de special tokens;
* dimensión;
* precision;
* tamaño de batch;
* longitud máxima;
* truncamiento;
* hardware;
* versión de Sylphy.

Configuración:

```yaml
representation:
  framework: sylphy
  model: XXX
  checkpoint: XXX
  layer: XXX
  pooling: mean
  remove_special_tokens: true
  output_dimension: XXX
  precision: float32
  max_length: XXX
  batch_size: XXX
```

## Validaciones

* una representación por `sequence_id`;
* dimensionalidad constante;
* ausencia de NaN e Inf;
* orden independiente del CSV;
* hash del archivo final;
* correspondencia exacta entre IDs y embeddings.

Output:

```text
representations/amp_sylphy_embeddings.parquet
metadata/representation_manifest.json
```

No regeneraremos embeddings para cada seed. Los embeddings serán un artefacto fijo compartido por todos los experimentos.

---

# Fase 8. Modelos y selección de hiperparámetros

Mantendría los cuatro modelos del trabajo original:

* KNN;
* Logistic Regression;
* Random Forest;
* SVM.

## Reglas

* scaler ajustado solamente con train;
* validación utilizada para model selection;
* test evaluado una única vez después de seleccionar configuración;
* ningún hiperparámetro elegido mirando test;
* mismo search space para todos los regímenes;
* mismo procedimiento de selección para cada split;
* todos los hiperparámetros finales registrados.

## Pipelines

### KNN

```text
StandardScaler → KNN
```

### Logistic Regression

```text
StandardScaler → Logistic Regression
```

### SVM

```text
StandardScaler → SVM
```

### Random Forest

```text
Random Forest
```

Debemos dejar explícito si SVM genera probabilidades mediante calibración o si trabajamos con decision scores. Para AUPRC, los scores son suficientes, pero debe quedar documentado.

---

# Fase 9. Matriz experimental

La matriz principal sería:

| Componente        | Valores                                        |
| ----------------- | ---------------------------------------------- |
| Dataset           | AMP task procesado                             |
| Representación    | Una representación Sylphy                      |
| Modelos           | KNN, LR, RF, SVM                               |
| Partición         | Stratified random + 3 similarity-aware regimes |
| Seeds             | 20                                             |
| Métrica principal | Test AUPRC                                     |
| Secundarias       | AUROC, MCC, balanced accuracy, F1              |
| Incertidumbre     | Mediana, IQR, bootstrap 95% CI                 |

Con cuatro estrategias, cuatro modelos y veinte seeds:

[
4 \times 4 \times 20 = 320
]

ejecuciones finales, sin contar el tuning.

Es una escala completamente manejable.

---

# Fase 10. Métricas y definiciones

## Baseline de AUPRC

Debe calcularse para cada test set:

[
\mathrm{AUPRC}_{baseline}
=========================

\frac{N_{positive,test}}{N_{test}}
]

No utilizaremos una única baseline si la prevalencia cambia entre particiones.

## Validation optimism

Definiremos:

[
G =
\mathrm{AUPRC}_{validation}
---------------------------

\mathrm{AUPRC}_{test}
]

Interpretación:

* (G > 0): validación optimista;
* (G = 0): concordancia;
* (G < 0): test superior a validación.

Esto corrige la ambigüedad de la figura actual, que mezcla “validation–test gap” con un eje `Test − Val`. 

## Variabilidad entre seeds

Eliminaremos `seed fragility`.

Usaremos:

> **Across-seed variability**

medida mediante:

[
\mathrm{IQR}
============

Q_{0.75} -
Q_{0.25}
]

## Interpretación estadística

La presentaremos como:

> a descriptive demonstrative analysis

Reportaremos:

* mediana;
* IQR;
* bootstrap 95% confidence intervals;
* número exacto de runs;
* distribuciones completas en el repositorio.

No necesitamos vender causalidad universal ni validación integral del framework.

---

# Fase 11. Artefactos que debe generar cada ejecución

Cada run debe producir:

## Predictions

```text
sequence_id
true_label
prediction_score
predicted_label
model
partition_strategy
similarity_threshold
seed
split
```

## Metrics

```text
model
partition_strategy
similarity_threshold
seed
split
auprc
auroc
mcc
f1
balanced_accuracy
positive_prevalence
auprc_baseline
```

## Execution record

```text
run_id
git_commit
configuration_hash
dataset_hash
representation_hash
split_hash
python_version
dependency_versions
start_time
end_time
hardware
```

Todo resultado del paper debe poder rastrearse hasta:

```text
raw CSV
→ preprocessing configuration
→ processed dataset
→ similarity groups
→ split
→ representation
→ model configuration
→ predictions
→ metric
→ figure
```

Eso será la evidencia práctica del argumento central del artículo.

---

# Fase 12. Nueva Figura 2

La figura debería dejar de depender del número arbitrario de clusters.

Propongo:

### A. Test AUPRC across generalisation regimes

Boxplots por modelo:

* stratified random;
* SIM90;
* SIM70;
* SIM50/60.

### B. Validation optimism

[
\mathrm{Validation} - \mathrm{Test}
]

### C. Performance versus similarity threshold

Mediana de test AUPRC frente al umbral.

### D. Across-seed variability

IQR de test AUPRC frente al umbral.

### E. Split integrity and dataset composition

Podría mostrar:

* máxima identidad cross-split;
* prevalencia positiva por test set;
* o número de secuencias retenidas por régimen.

Mi preferencia sería **máxima similitud cross-split**, porque demuestra que la partición realmente está haciendo lo declarado.

---

# Fase 13. Qué debe quedar en el main text

El lector no debería tener que entrar al repositorio para conocer:

* fuente del CSV;
* definición del endpoint;
* positivos y negativos;
* semántica del conjunto negativo;
* rango de longitudes;
* residuos aceptados;
* tratamiento de duplicados;
* procedimiento de similitud;
* umbrales;
* definición de los splits;
* representación;
* modelos;
* proporciones;
* seeds;
* tuning;
* baseline de AUPRC;
* definición del gap;
* resumen de incertidumbre.

El repositorio contiene la ejecución completa; el manuscrito debe contener la lógica científica suficiente para comprenderla. Esa es una exigencia compartida por ambos revisores. 

---

# Plan operativo para hoy

## Bloque 1 — Dejar el dataset trazable

1. crear estructura limpia del repo;
2. mover el CSV a `data/raw/`;
3. calcular hash;
4. inspeccionar columnas;
5. documentar la semántica de cada columna;
6. crear el validador;
7. generar el primer QC report.

## Bloque 2 — Construir formalmente el AMP task

1. definir positivos;
2. resolver qué representa el cero;
3. definir negativos/background;
4. normalizar secuencias;
5. resolver duplicados exactos;
6. generar `amp_task.csv`;
7. producir tabla descriptiva.

## Bloque 3 — Ejecutar BioSieve

1. generar auditoría de similitud;
2. probar grid de umbrales;
3. inspeccionar tamaños de grupos;
4. identificar grupos con etiquetas mixtas;
5. verificar factibilidad de splits;
6. escoger tres regímenes por criterios predefinidos.

## Bloque 4 — Dejar los splits terminados

1. estratificado aleatorio;
2. similarity-aware en tres niveles;
3. 20 seeds;
4. validaciones automáticas;
5. archivos de asignación;
6. reportes de prevalencia y similitud cross-split.

## Bloque 5 — Congelar Sylphy

1. elegir modelo/checkpoint;
2. escribir configuración;
3. generar embeddings;
4. validarlos;
5. almacenar manifest y hash.

Al terminar hoy, el caso debe quedar preparado para que mañana solamente falte:

```text
train → evaluate → summarize → plot
```

---

# Definición de “caso de estudio operacional”

No daremos esta fase por terminada hasta que se cumpla todo lo siguiente:

* el CSV raw está intacto y hasheado;
* el endpoint está definido;
* la clase negativa está justificada;
* el procesamiento es ejecutable desde cero;
* los duplicados están documentados;
* BioSieve genera grupos reproducibles;
* los umbrales están justificados sin mirar performance;
* los splits pasan controles automáticos;
* las prevalencias están registradas;
* los embeddings de Sylphy están versionados;
* cada artefacto tiene un identificador;
* no existen rutas absolutas;
* una instalación limpia puede reconstruir los datos procesados y splits.

El **primer paso concreto ahora** es inspeccionar el CSV pivote y definir exactamente la semántica de las columnas, especialmente qué significa un `0` en la columna antimicrobial. Esa decisión determina todo el resto del caso.
