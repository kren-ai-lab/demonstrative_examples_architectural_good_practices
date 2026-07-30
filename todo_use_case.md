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

# Nueva Fase 4. Agrupamiento homology-aware específico por tarea

## Objetivo

Para cada task:

1. tomar todas sus secuencias positivas y negativas;
2. agruparlas conjuntamente mediante MMseqs2;
3. conservar todas las secuencias;
4. utilizar los grupos como unidades indivisibles durante la partición;
5. impedir que secuencias relacionadas bajo la configuración definida aparezcan en particiones diferentes.

Por tanto, **no hacemos redundancy reduction en el análisis principal**. Hacemos:

> redundancy and homology control through group-constrained partitioning.

Esto conserva el tamaño del dataset y evita introducir el efecto adicional de escoger representantes.

---

# 1. El agrupamiento debe hacerse independientemente por task

Tenemos dos universos experimentales diferentes:

## Task 1

```text
amp_vs_toxic_without_amp_evidence
```

Contiene:

* AMP positivos;
* péptidos tóxicos sin evidencia AMP.

## Task 2

```text
amp_vs_without_amp_evidence
```

Contiene:

* AMP positivos;
* todos los péptidos sin evidencia AMP.

Cada task debe generar su propio FASTA y su propio agrupamiento:

```text
amp_task.csv
├── task 1 → FASTA → MMseqs2 clusters
└── task 2 → FASTA → MMseqs2 clusters
```

Esto significa que una secuencia AMP presente en ambos tasks puede pertenecer a grupos diferentes, porque el conjunto de secuencias con las que se compara también cambia. **Eso es correcto:** el agrupamiento forma parte de la definición específica de cada benchmark.

No reutilizaría los clusters del task amplio para el task tóxico.

---

# 2. Positivos y negativos se agrupan juntos

Esto es crucial.

No debemos ejecutar:

```text
positives → clusters
negatives → clusters
```

Debemos ejecutar:

```text
positives + negatives → clusters
```

La etiqueta no participa en MMseqs2. Solo se incorpora después para caracterizar los grupos.

Si una secuencia AMP y una secuencia del background son suficientemente similares:

* deben pertenecer al mismo grupo;
* deben quedar en la misma partición;
* no deben eliminarse automáticamente;
* el grupo debe marcarse como `mixed-label`.

Separar el clustering por clase permitiría que secuencias similares con etiquetas opuestas quedaran en train y test, justamente lo que queremos evitar.

---

# 3. Definición operacional de “homology-aware”

En el código podremos usar:

```text
homology_aware
```

pero en Methods debemos definirlo cuidadosamente:

> Homology-aware groups were operationally defined as MMseqs2 sequence-similarity connected components satisfying predefined sequence-identity and bidirectional-coverage criteria.

Así dejamos claro que:

* utilizamos similitud de secuencia como proxy operacional;
* no afirmamos homología evolutiva demostrada;
* la definición es completamente reproducible.

MMseqs2 permite controlar explícitamente identidad mínima y cobertura, y `alignment-mode 3` calcula identidad como residuos idénticos divididos por las columnas alineadas, incluyendo gaps internos. Además, `cluster-mode 1` corresponde al agrupamiento por componentes conectados. ([GitHub][1])

---

# 4. Estrategias experimentales

No trabajaría con seis umbrales. La matriz debe mantenerse manejable.

## Baseline

### `random_stratified`

* no utiliza grupos de homología;
* mantiene aproximadamente la prevalencia;
* representa el escenario convencional.

## Homology-aware principal

### `H90`

```text
minimum sequence identity: 0.90
minimum coverage: 0.90
```

Interpretación:

> control de secuencias altamente similares y redundancia cercana.

## Homology-aware estricto

### `H70`

```text
minimum sequence identity: 0.70
minimum coverage: 0.90
```

Interpretación:

> control más estricto de generalización entre regiones relacionadas del espacio de secuencias.

La matriz principal quedaría así:

| Task                      | Random | H90 | H70 |
| ------------------------- | -----: | --: | --: |
| AMP vs toxic background   |      ✓ |   ✓ |   ✓ |
| AMP vs general background |      ✓ |   ✓ |   ✓ |

Son **seis condiciones conceptuales**, pero solo cuatro ejecuciones de MMseqs2:

```text
2 tasks × 2 thresholds = 4 clusterings
```

El split random no necesita clustering.

MMseqs2 y Linclust se han utilizado con umbrales de identidad de 90%, 70% y 50%, pero esos valores son criterios operacionales de agrupamiento y no deben interpretarse automáticamente como límites universales de familia o fold. ([Nature][2])

## H50

No lo incluiría inicialmente en la matriz principal.

Podemos ejecutarlo únicamente durante la auditoría para responder:

* ¿se forman componentes gigantes?
* ¿sigue siendo posible construir train/validation/test?
* ¿la interpretación continúa siendo razonable en péptidos cortos?

Solo entraría al experimento si agrega información clara. **Mi diseño principal sería Random + H90 + H70.**

---

# 5. Configuración MMseqs2 propuesta

Usaría `easy-cluster`, no `easy-linclust`, porque el tamaño del dataset es manejable y preferimos priorizar sensibilidad y una ejecución estándar.

## Parámetros fijos

```yaml
mmseqs:
  workflow: easy-cluster
  cluster_mode: 1
  alignment_mode: 3
  coverage_mode: 0
  minimum_coverage: 0.90

  configurations:
    H90:
      minimum_sequence_identity: 0.90

    H70:
      minimum_sequence_identity: 0.70
```

### Justificación

#### `cluster_mode: 1`

Componentes conectados.

Si:

```text
A ~ B
B ~ C
```

las tres secuencias quedan en el mismo componente, aunque `A` y `C` no superen directamente el umbral.

Para leakage control esto es conveniente: ninguna relación directa que supere los criterios puede cruzar entre train, validation y test. MMseqs2 identifica `cluster-mode 1` como connected-component clustering. ([GitHub][1])

#### `alignment_mode: 3`

Obliga a MMseqs2 a calcular explícitamente la identidad del alineamiento, en vez de depender de la aproximación basada en score que puede utilizar por defecto. ([GitHub][1])

#### `coverage_mode: 0`

Exige cobertura respecto de la secuencia más larga. Esto actúa como un criterio bidireccional conservador y evita agrupar secuencias solamente porque comparten un motivo local corto. La cobertura en MMseqs2 depende explícitamente de `-c` y `--cov-mode`; en modo 0 se normaliza usando la longitud máxima entre las secuencias comparadas. ([Nature][2])

#### `minimum_coverage: 0.90`

Para péptidos cortos prefiero 90% en vez de 80%. Así, la similitud debe cubrir casi toda la secuencia y no solamente un fragmento.

---

# 6. Punto que debemos validar antes de congelar el config

## E-value en péptidos cortos

MMseqs2 también aplica un criterio de E-value además de identidad y cobertura. ([Nature][2])

En péptidos cortos, un E-value restrictivo podría eliminar alineamientos que cumplen identidad y cobertura, simplemente por la longitud reducida. Por eso, antes de congelar la configuración debemos realizar una **sanity check**, no un nuevo experimento:

1. seleccionar pares idénticos o casi idénticos conocidos;
2. ejecutar la configuración;
3. confirmar que MMseqs2 los conecta;
4. comparar el E-value predeterminado con uno permisivo;
5. fijar un único valor antes del modelado.

La decisión no debe depender del AUPRC, sino de que el agrupamiento recupere correctamente relaciones que cumplen los criterios declarados.

---

# 7. Responsabilidades de BioSieve

BioSieve será el orquestador oficial. Debe:

1. leer `amp_task.csv`;
2. seleccionar un `task_id`;
3. verificar unicidad de `sequence_id`;
4. exportar el FASTA;
5. ejecutar MMseqs2;
6. capturar versión y comando;
7. convertir el TSV de MMseqs2 a una tabla estable;
8. agregar etiquetas solamente después del clustering;
9. calcular estadísticas de los grupos;
10. generar hashes y manifest.

La lógica del agrupamiento debe ser independiente de las etiquetas.

---

# 8. Outputs por task y configuración

```text
data/homology/
├── amp_vs_toxic_without_amp_evidence/
│   ├── H90/
│   │   ├── sequences.fasta
│   │   ├── mmseqs_clusters.tsv
│   │   ├── cluster_membership.csv
│   │   ├── cluster_summary.csv
│   │   ├── mmseqs.log
│   │   └── manifest.json
│   └── H70/
│       └── ...
└── amp_vs_without_amp_evidence/
    ├── H90/
    │   └── ...
    └── H70/
        └── ...
```

---

# 9. Tabla de membresía

## `cluster_membership.csv`

Una fila por secuencia:

```text
task_id
homology_config_id
sequence_id
cluster_id
representative_sequence_id
amp_label
cluster_size
cluster_positive_count
cluster_negative_count
mixed_label_cluster
```

Ejemplo:

```csv
task_id,homology_config_id,sequence_id,cluster_id,representative_sequence_id,amp_label,cluster_size,cluster_positive_count,cluster_negative_count,mixed_label_cluster
amp_vs_toxic_without_amp_evidence,H90,seq_001,H90_C000001,seq_001,1,4,3,1,1
amp_vs_toxic_without_amp_evidence,H90,seq_002,H90_C000001,seq_001,0,4,3,1,1
```

El `cluster_id` debe generarse en nuestro pipeline, no depender directamente del orden arbitrario entregado por MMseqs2. Por ejemplo:

```text
cluster_id = SHA256(
    task_id
    + homology_config_id
    + sorted(member_sequence_ids)
)
```

Podemos usar un prefijo corto para los CSV y conservar el hash completo en el manifest.

---

# 10. Resumen de agrupamiento

## `cluster_summary.csv`

Una fila por grupo:

```text
task_id
homology_config_id
cluster_id
representative_sequence_id
cluster_size
positive_count
negative_count
positive_fraction
mixed_label_cluster
minimum_length
maximum_length
```

También debemos generar un resumen global:

| Métrica                       | Descripción                 |
| ----------------------------- | --------------------------- |
| Número de secuencias          | Total del task              |
| Número de clusters            | Componentes generados       |
| Singletons                    | Grupos de una secuencia     |
| Cluster máximo                | Mayor componente            |
| Mediana de tamaño             | Distribución de redundancia |
| Clusters mixtos               | Contienen ambas etiquetas   |
| Secuencias en clusters mixtos | Potencial ambigüedad        |
| Compression ratio             | clusters / sequences        |

Los mixed-label clusters no representan necesariamente errores. Pueden reflejar:

* anotaciones incompletas;
* backgrounds sin evidencia AMP;
* secuencias similares con actividad diferente;
* dependencia contextual de la actividad;
* ruido o contradicciones entre fuentes.

Los conservamos y los mantenemos íntegros durante la partición.

---

# 11. Reglas de integridad

Cada ejecución debe comprobar:

* todas las secuencias del task aparecen en el FASTA;
* cada `sequence_id` aparece una sola vez;
* todas las secuencias reciben exactamente un cluster;
* ningún cluster contiene IDs inexistentes;
* ningún ID pertenece a más de un cluster;
* el número de miembros coincide con el task;
* las etiquetas no fueron utilizadas para construir clusters;
* los resultados son idénticos al repetir la ejecución con la misma versión y configuración.

---

# 12. Relación con la partición

MMseqs2 no construirá directamente train, validation y test. Solo generará los grupos.

Posteriormente:

```text
H90 clusters → atomic groups → train / validation / test
H70 clusters → atomic groups → train / validation / test
```

El seed afectará la asignación de grupos, pero **no el agrupamiento MMseqs2**.

Por tanto:

```text
Cada task/configuración se clusteriza una vez.
Cada agrupamiento puede generar 20 splits mediante diferentes seeds.
```

Los grupos nunca se dividen.

---

# 13. Criterios de factibilidad antes del modelado

Antes de aceptar H90 o H70 debemos comprobar:

* que no exista un componente que concentre una fracción excesiva del dataset;
* que sea posible generar train, validation y test;
* que ambas clases aparezcan en las tres particiones;
* que el tamaño del test sea suficiente;
* que la prevalencia pueda mantenerse razonablemente;
* que el agrupamiento no elimine indirectamente una clase de alguna partición.

Si H70 produce un componente gigante, no cambiaremos parámetros mirando performance. Documentaremos el problema y decidiremos entre:

* mantener H70 con proporciones de split adaptadas;
* usar un umbral intermedio, como H80;
* excluir H70 del experimento principal.

La decisión se toma mirando **estructura y factibilidad**, nunca resultados predictivos.

---

# Diseño final recomendado

## Experimento principal

```text
Task 1:
    random_stratified
    homology_aware_H90
    homology_aware_H70

Task 2:
    random_stratified
    homology_aware_H90
    homology_aware_H70
```

## Lo que no agregamos

* distance-aware con Sylphy;
* clustering separado por clase;
* representative-only datasets;
* combinaciones de identity × coverage;
* muchos algoritmos de clustering;
* umbrales interpretados como family-level o fold-level;
* eliminación automática de grupos mixtos.

Esto responde directamente a la solicitud de los revisores de declarar algoritmo, métrica, secuencias utilizadas, balance, significado de los parámetros y régimen de generalización, sin convertir el caso de estudio en una matriz inmanejable. 

**Mi recomendación final:** avanzar con `Random + H90 + H70`, cobertura bidireccional fija de 90%, componentes conectados y clustering independiente para cada task. Antes de implementar el script, solo debemos validar el comportamiento del E-value con los péptidos cortos.

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
