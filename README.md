# TDHA-fMRI
TDHA-fMRI using CNN



## Datasets
- Download datasets: https://www.nitrc.org/frs/?group_id=383


## N Samples

n samples:

-   https://fcon_1000.projects.nitrc.org/indi/adhd200/

## DATOS PROCESADOS

**FUENTE:** [NITRC](https://www.nitrc.org/frs/?group_id=383)  
**DESCRIPCIÓN:** Señal BOLD extraída de ADHD200 preprocesados con Pipeline ATHENEA.

| SITIO       | REGISTROS | TDAH | CONTROL |
|-------------|-----------|------|---------|
| NYU         | 177       | 90   | 87      |
| NeuroIMAGE  | 39        | 17   | 22      |
| KKI         | 78        | 20   | 58      |
| OHSU        | 66        | 28   | 38      |
| PEKING      | 183       | 74   | 109     |
| **TOTAL**   | **543**   | **229** | **314** |



# __Hold-Out Validation__

## NYU - Hold-Out Validation

> **Configuración de las pruebas:** 
> - Train 80% – Validation 20%  
> - 20 repeticiones con semillas aleatorias distintas  
> - Accuracy promedio y desviación estándar calculados sobre las repeticiones

| # | ROIs          | Iteraciones | Accuracy promedio | Desv. Estándar | Rango resultados |
|---|---------------|-------------|-------------------|----------------|------------------|
| 1 | **18**        | 20          | **0.739**        | 0.077          | 0.583 - 0.833    |
| 2 | 20-nyu-pca    | 20          | 0.618            | 0.072          | 0.528 - 0.750    |
| 3 | 116-all-pca   | 20          | 0.601            | 0.070          | 0.361 - 0.667    |
| 4 | 39            | 20          | 0.544            | 0.056          | 0.500 - 0.694    |
| 5 | 116           | 20          | 0.542            | 0.073          | 0.472 - 0.722    |



#  Convenciones Experimentos de Ablación - MLFLOW


---

##  Estructura general del ID
```
<TRACK>-WL<win>-ST<st>-FC<fc_spec>-CH<chs>
```

---

## TRACK
- `B` = baseline (SVM / regresión logística / estático)  
- `C` = cnn (FC dinámica con CNN)

---


## WL / ST
- `WL<win_len>` → tamaño de ventana  
- `ST<stride>` → stride  

**Ejemplo:**  
```
WL60-ST10
```

---

## FC (fc_spec)
Tipos de conectividad funcional:

- `P`  = Pearson  
- `S`  = Spearman  

**Sufijos opcionales:**  
- `Z` = Fisher-z  
- `A` = |z| (valor absoluto)  

**Ejemplos:**  
```
FCPZ
FCSPZ
FCPZA
```

---

## CH (chs)
Canales sobre la FC dinámica, separados por `+`.

- `M` = mean(z)  
- `S` = std(z)  
- `A` = mean(|z|)  
- `p_10` = percentil 10  
- `p_90` = percentil 90 

**Avanzados:**  
- `E` = espectral  
- `D` = difusión  
- `H` = entropía  
- `C` = clustering  

**Ejemplos:**  
```
CHM+S
CHM
CHM+S+E
```
