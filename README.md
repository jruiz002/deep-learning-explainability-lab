# Laboratorio de Explicabilidad

**Curso:** Deep Learning - Universidad del Valle de Guatemala

**Integrantes:**

- José Ruiz 
- José Auyón 
- Gerardo Fernández 

En este task se implementó Grad-CAM desde cero sobre un ResNet-18 preentrenado en ImageNet, se calculó CAM directamente
desde los pesos de la capa final y se verificó numéricamente que ambos métodos coinciden. Después se usaron los mapas para
analizar tres imágenes. Todo el código usa PyTorch, NumPy, SciPy y Matplotlib; no se usó `pytorch-grad-cam`, `captum`
ni librerías similares.

## Cómo levantar y correr el proyecto

Requisitos: Python 3.10 a 3.12 (lo desarrollamos con 3.11) y conexión a internet la primera vez, porque PyTorch descarga
los pesos de ResNet-18 (~45 MB). Las tres imágenes ya vienen en `data/images/`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                   # hace importable el paquete xai_lab

python -m pytest -q                # pruebas unitarias (32)
python scripts/run_task1.py        # corre todo el Task 1
python scripts/run_task2.py        # corre todo el Task 2 (SHAP sobre German Credit)
python scripts/run_task3.py        # corre todo el Task 3 (LIME desde cero)
```

Al correrlo, la consola imprime un reporte de ejecución con una sección por inciso (`Task 1.1`, `1.2`, `1.3`,
`1.4(a)`, `1.4(b)` y `1.4(c)`), una tabla de **verificaciones numéricas** (OK/FALLO contra una tolerancia) y la lista de
archivos generados. El programa termina con código de salida 1 si alguna verificación falla. Las figuras quedan en
`outputs/task1/figures/` y los números en `outputs/task1/results.json`. Tarda unos segundos en CPU.
El Task 2 se corre igual (`scripts/run_task2.py`; descarga German Credit de OpenML la primera vez, unos 10 s en total), imprime
su propio reporte con verificaciones numéricas, guarda las figuras en `outputs/task2/figures/` y los números en
`outputs/task2/results.json`, y reescribe su sección del informe (entre `INFORME_TASK2:INICIO` y `INFORME_TASK2:FIN`, debajo del
informe del Task 1).
El Task 3 (`scripts/run_task3.py`, unos 2 s; reutiliza el modelo y los clientes del Task 2) imprime su reporte con verificaciones numéricas,
guarda las figuras en `outputs/task3/figures/` y los números en `outputs/task3/results.json`, y reescribe su sección del informe (entre
`INFORME_TASK3:INICIO` y `INFORME_TASK3:FIN`, debajo del informe del Task 2). (`make setup`, `make test`, `make task1`, `make task2` y
`make task3` hacen lo mismo.)

Estructura del código:

```
data/images.json     fuente, autor, licencia, SHA-256 y recuadro de cada imagen
src/xai_lab/
  gradcam.py         Grad-CAM con hooks de avance y retroceso (1.2)
  cam.py             CAM desde los pesos de fc (1.3)
  maps.py            ReLU/min-max, interpolación, máscara top-20 %, IoU, masa en recuadro
  causal.py          prueba causal con gris
  task1.py           los experimentos de 1.1 a 1.4
  data.py, model.py, viz.py, report.py, console.py, config.py
  credit.py          German Credit, partición, GBC, fondo y selección de clientes (Tasks 2 y 3)
  shapley.py         Shapley exacto con NumPy (2.5)
  task2.py           los experimentos de 2.1 a 2.6
  viz_tabular.py, console_task2.py, report_task2.py   figuras, reporte de consola e informe del Task 2
  lime_tabular.py    LIME tabular con NumPy: muestreo, kernel, ridge ponderada en forma cerrada, R²_w y N_ef (Task 3)
  task3.py           los experimentos 3.1 y 3.2
  viz_task3.py, console_task3.py, report_task3.py   figuras, reporte de consola e informe del Task 3
scripts/run_task1.py, scripts/run_task2.py, scripts/run_task3.py   puntos de entrada
tests/               pruebas de las identidades matemáticas, geometría y hooks
```

> La sección **Informe** de abajo la genera automáticamente `python scripts/run_task1.py` con los datos de la
> ejecución (tablas, errores, probabilidades y figuras); por eso no se edita a mano, ya que cada corrida la reescribe.

---

<!-- INFORME:INICIO (generado por scripts/run_task1.py; no editar a mano) -->

# Informe

## 0. Configuración

- Modelo: ResNet-18 de torchvision con pesos `IMAGENET1K_V1`, en modo evaluación.
- Preprocesamiento: redimensión a 256 píxeles en el lado corto, recorte central de 224×224 y normalización por canal con
  media (0.485, 0.456, 0.406) y desviación estándar (0.229, 0.224, 0.225).
- Semilla aleatoria: **42** (Python, NumPy y PyTorch). Ejecución en CPU con float32.
- Versiones: torch 2.14.1, torchvision 0.29.1, Python 3.11.2.

## Task 1.1 - Modelo e imágenes

Se seleccionaron tres imágenes de Wikimedia Commons:

| Imagen | Característica | Fuente | Autor | Licencia |
|---|---|---|---|---|
| (a) | Objeto claro y centrado | [a_golden_retriever.jpg](https://commons.wikimedia.org/wiki/File:Callie_the_golden_retriever_puppy.jpg) | MichaelMcPhee (Wikimedia Commons) | CC BY 3.0 |
| (b) | Dos objetos de clases distintas (perro y gato) | [b_cat_and_dog.jpg](https://commons.wikimedia.org/wiki/File:Cute_cat_and_dog_1.jpg) | S. J. Pyrotechnic (Wikimedia Commons) | CC BY-SA 2.0 |
| (c) | Objeto fuera de su contexto habitual (velero en tierra, sobre un remolque) | [c_sailboat_on_trailer.jpg](https://commons.wikimedia.org/wiki/File:Catalina_22_sailboat_on_a_trailer.jpg) | Jwkane (Wikimedia Commons) | CC BY-SA 3.0 |

Se obtuvo el top-5 de cada imagen con el modelo en modo evaluación:

| # | (a) | prob. | (b) | prob. | (c) | prob. |
|---|---|---|---|---|---|---|
| 1 | golden retriever | 0.9948 | Chihuahua | 0.4449 | trimaran | 0.2054 |
| 2 | Tibetan mastiff | 0.0016 | tabby | 0.1335 | catamaran | 0.1791 |
| 3 | cocker spaniel | 0.0008 | French bulldog | 0.1230 | projectile | 0.0970 |
| 4 | Saluki | 0.0007 | Egyptian cat | 0.0525 | missile | 0.0863 |
| 5 | Labrador retriever | 0.0005 | tiger cat | 0.0395 | amphibian | 0.0654 |

Para la imagen (c) se definió el recuadro del objeto sobre la imagen original como (135, 150, 880, 450) (velero completo, sin el remolque)
y se transformó con el mismo redimensionado y recorte que recibe el modelo. Sus coordenadas en la imagen de 224×224
(x0, y0, x1, y1) son **(0.0, 37.3, 224.0, 144.0)**. Como el velero es más ancho que el recorte central, el recuadro
llega a ambos bordes.

![Figura 1. Entradas de 224×224 (con el recuadro en (c)) y top-5 de probabilidades.](outputs/task1/figures/fig_1_1_predictions.png)

*Figura 1. Entradas de 224×224 (con el recuadro en (c)) y top-5 de probabilidades.*

En (a) la clase de mayor probabilidad es `golden retriever` (0.9948). En (b) la clase de mayor probabilidad es
`Chihuahua` (0.4449) y la clase de gato mejor ubicada es `tabby` (0.1335), presente en el top-5. En (c) el
top-1 (`trimaran`) alcanza sólo 0.2054 y el top-5 mezcla embarcaciones con `projectile` y `missile`,
lo que indica que el contexto de la imagen afecta la predicción.

## Task 1.2 - Grad-CAM desde cero

Se implementó Grad-CAM sobre `layer4` con un hook de avance, que captura las activaciones A (512 canales de 7×7), y un hook de
retroceso, que captura ∂y^c/∂A. Se respetaron las reglas del enunciado:

1. Se retropropagó desde el **logit** y^c (antes del softmax), no desde la probabilidad ni desde una pérdida.
2. Se calculó α_k^c como el promedio espacial del gradiente de cada canal k.
3. Se formó L^c = ReLU(Σ_k α_k^c A^k), se amplió a 224×224 con interpolación bilineal y se normalizó con su mínimo y su máximo.

Se generó el mapa de la clase de mayor probabilidad de cada imagen y, en (b), también el de la clase del segundo objeto.
La posición del máximo de cada mapa (en píxeles de la imagen de 224×224) fue: (a) golden retriever (#207): (fila 111, columna 48); (b) Chihuahua (#151): (fila 48, columna 48); tabby (#281): (fila 175, columna 48); (c) trimaran (#871): (fila 80, columna 175).

![Figura 2. Grad-CAM superpuesto: (a) clase top-1; (b) clase del perro y clase del gato; (c) clase top-1.](outputs/task1/figures/fig_1_2_gradcam.png)

*Figura 2. Grad-CAM superpuesto: (a) clase top-1; (b) clase del perro y clase del gato; (c) clase top-1.*

Al observar la figura, en (a) el mapa se concentra en la cara y las orejas del cachorro; en (b) el mapa de cada clase cae sobre el animal
correspondiente; y en (c) cae sobre el casco y la cabina del velero, con extensión hacia la parte baja, donde está el remolque.

## Task 1.3 - CAM desde los pesos de `fc`

Se calculó M^c_ij = Σ_k w_k^c A^k_ij directamente con los pesos de `fc`, sin gradientes, para la clase de mayor probabilidad de
cada imagen, con Z = 49.

**(b)** Descomposición del puntaje: (1/Z) Σ M + b_c contra el logit y^c.

| Imagen | Clase | Logit y^c | (1/Z) Σ M + b_c | Error absoluto |
|---|---|---|---|---|
| a | golden retriever (#207) | 18.294334 | 18.294336 | 1.91e-06 |
| b | Chihuahua (#151) | 11.057706 | 11.057705 | 9.54e-07 |
| c | trimaran (#871) | 10.640809 | 10.640808 | 9.54e-07 |

**(c)** Error absoluto máximo entre Z·L^c (sin ReLU ni normalización) y M^c sobre las 49 posiciones:

| Imagen | Clase | Error abs. máximo |
|---|---|---|
| a | golden retriever (#207) | 1.14e-05 |
| b | Chihuahua (#151) | 7.63e-06 |
| c | trimaran (#871) | 5.72e-06 |

**(d)** Error absoluto máximo entre ambos mapas después de ReLU y normalización mínimo-máximo:

| Imagen | Clase | Error abs. máximo |
|---|---|---|
| a | golden retriever (#207) | 4.17e-07 |
| b | Chihuahua (#151) | 2.38e-07 |
| c | trimaran (#871) | 1.79e-07 |

![Figura 3. CAM (M^c), Z·L^c de Grad-CAM y sus diferencias, por imagen.](outputs/task1/figures/fig_1_3_cam_vs_gradcam.png)

*Figura 3. CAM (M^c), Z·L^c de Grad-CAM y sus diferencias, por imagen.*

La descomposición del logit se cumplió con un error absoluto máximo de 1.91e-06 y la igualdad Z·L^c = M^c con un error máximo de 1.14e-05, valores del orden de la
precisión de float32. Tras aplicar ReLU y normalización, el error máximo entre ambos mapas fue de 4.17e-07. La comparación del inciso (c)
se realizó antes de ReLU y normalización porque esas operaciones no son lineales y ocultarían la igualdad. En esta arquitectura, por lo tanto,
Grad-CAM y CAM producen el mismo mapa.

## Task 1.4 - Análisis

### (a) α frente a w/Z (imagen a, cinco canales de mayor peso)

| Canal | w | α (Grad-CAM) | w/Z | Error relativo |
|---|---|---|---|---|
| 28 | 0.288863 | 0.0058951629 | 0.0058951633 | 6.77e-08 |
| 242 | 0.255639 | 0.0052171177 | 0.0052171179 | 3.83e-08 |
| 156 | 0.246214 | 0.0050247773 | 0.0050247771 | 3.59e-08 |
| 473 | 0.223402 | 0.0045592198 | 0.0045592198 | 4.17e-09 |
| 497 | 0.210018 | 0.0042860918 | 0.0042860915 | 5.32e-08 |

![Figura 4. α_k^c contra w_k^c/Z en los 512 canales (en círculo, los cinco de mayor peso).](outputs/task1/figures/fig_1_4a_alpha_vs_w.png)

*Figura 4. α_k^c contra w_k^c/Z en los 512 canales (en círculo, los cinco de mayor peso).*

El error relativo máximo observado en los cinco canales fue de 6.77e-08. Como y^c = (1/Z) Σ_k w_k Σ_ij A^k_ij + b_c, la derivada
∂y^c/∂A^k_ij = w_k/Z **es la misma en todas las posiciones (i, j)**, porque el pooling global promedia dando el mismo peso 1/Z a cada posición.
Ese hecho se usa al promediar espacialmente el gradiente: el promedio de Z valores idénticos es ese mismo valor, por lo que α_k^c = w_k^c/Z.

Los errores no son exactamente cero porque autograd calcula el gradiente con sumas y productos en float32, que se redondean de forma distinta a la
división directa w/Z. Su magnitud es del orden de la precisión de la máquina (ε ≈ 1.2e-7) y no corresponde a un error del método.

### (b) ¿El mapa discrimina entre las dos clases? (imagen b)

Se tomó el 20 % de píxeles de mayor valor de cada mapa (10035 de 50176) para Chihuahua (#151) y tabby (#281):

- Intersección: 7 píxeles · Unión: 20063 píxeles
- **IoU = 0.0003**

![Figura 5. Conjuntos del 20 % de mayor valor para cada clase.](outputs/task1/figures/fig_1_4b_iou.png)

*Figura 5. Conjuntos del 20 % de mayor valor para cada clase.*

Un IoU cercano a cero indica que el mapa **sí discrimina** entre las dos clases: los conjuntos casi no se solapan y cada uno cae sobre el animal que le corresponde.

Aun así, la localización es gruesa. Cada posición del mapa original (7×7) cubre 32×32 píxeles de la entrada, de modo que un objeto menor de ~32 píxeles,
o dos objetos más cercanos entre sí que esa distancia, caen en la misma celda o en celdas mezcladas, y el mapa no puede separarlos ni delimitar su contorno.
La interpolación bilineal sólo suaviza el mapa y no agrega resolución. Además, el campo receptivo de `layer4` es mayor que 32 píxeles, lo que difumina aún más
los bordes. En esta imagen la discriminación fue posible porque ambos animales son grandes.

### (c) ¿Usa el modelo el fondo? (imagen c)

Clase analizada: **trimaran** (top-1).

| Medición | Valor |
|---|---|
| Fracción de la masa del mapa normalizado dentro del recuadro | **0.7063** |
| Fracción del área de la imagen que cubre el recuadro | 0.4777 |
| p(trimaran), imagen original | 0.2054 |
| p(trimaran), todo lo de fuera del recuadro en gris | 0.0371 (cambio **-0.1683**) |
| Top-1 con el fondo en gris | catamaran (0.7663) |
| Control adicional: objeto en gris y fondo intacto | p = 0.0002 (cambio -0.2052) |

El gris utilizado fue RGB (128, 128, 128), aplicado sobre el recorte de 224×224 antes de normalizar.

![Figura 6. De izquierda a derecha: original con recuadro, mapa Grad-CAM, fondo en gris y objeto en gris.](outputs/task1/figures/fig_1_4c_causal.png)

*Figura 6. De izquierda a derecha: original con recuadro, mapa Grad-CAM, fondo en gris y objeto en gris.*

El 70.6% de la masa del mapa cae dentro de un recuadro que ocupa el 47.8% de la imagen (1.5 veces lo esperable si la masa
estuviera repartida de forma uniforme). Al ocultar el objeto, la probabilidad de la clase pasó de 0.2054 a 0.0002. Por lo tanto, el mapa y la prueba causal
coinciden en que el modelo usa el objeto.

El fondo, sin embargo, no es irrelevante. Aproximadamente el 29.4% de la masa del mapa queda fuera del recuadro y, al reemplazar el fondo por gris, la probabilidad de
`trimaran` disminuyó 82% (de 0.2054 a 0.0371), mientras que el top-1 pasó a `catamaran` con 0.7663.

**Conclusión:** hay evidencia parcial de dependencia del fondo, aunque no la de un atajo total. Esta conclusión tiene límites: se analizó una sola imagen, el gris es una entrada fuera de la distribución de entrenamiento (parte del
cambio puede deberse a ello y no sólo a la eliminación de información) y el recuadro ocupa todo el ancho del recorte, lo que deja poco fondo lateral.

## Conclusiones

- Grad-CAM y CAM coincidieron numéricamente en ResNet-18 (errores máximos de 1.1e-05 en float32), porque α_k^c = w_k^c/Z.
- Los mapas distinguieron las clases de una misma imagen (IoU de 0.0003 en (b)), pero su resolución de 7×7 limita la localización de objetos pequeños o cercanos entre sí.
- Un mapa de calor por sí solo no demuestra que el modelo use o ignore el fondo; la prueba causal con gris lo complementó y en (c) mostró evidencia parcial de dependencia del fondo, aunque no la de un atajo total.

<!-- INFORME:FIN -->

<!-- INFORME_TASK2:INICIO (generado por scripts/run_task2.py; no editar a mano) -->

# Task 2

## Configuración

- Datos: German Credit (`fetch_openml("credit-g", version=1)`), 1000 clientes y 20 variables
  (13 categóricas y 7 numéricas). Objetivo: `good` = 1, `bad` = 0.
- Semilla aleatoria: **42** (partición, modelo, fondo, permutaciones y ruido de 2.3).
- Versiones: scikit-learn 1.9.0, shap 0.53.0, NumPy 2.5.2, pandas 3.0.5, Python 3.14.7.
- Limitación: las 13 categóricas se codificaron con `OrdinalEncoder`, cuyos códigos siguen el orden alfabético de las categorías y no un
  orden con significado, pero el modelo los trata como numéricos. Por eso un umbral de un árbol sobre una categórica agrupa categorías por su
  código, y los φ de esas variables resumen el efecto de ese código y no el de una categoría por separado.

## Task 2.1 - Datos y modelo

Partición estratificada 80/20: 800 clientes de entrenamiento y 200 de prueba, con una proporción de `good` de
0.70 y 0.70. Modelo: `GradientBoostingClassifier` con 200 árboles de profundidad
3.

| Métrica (prueba) | Valor |
|---|---|
| Exactitud | **0.775** |
| AUC | **0.765** |

La exactitud supera en 7.5 puntos porcentuales a predecir siempre `good` (0.700).

## Task 2.2 - TreeSHAP interventional

Se usó `TreeExplainer(model, data=fondo, feature_perturbation="interventional", model_output="raw")`, con un fondo de 100 filas de entrenamiento
(semilla 42), sobre las 200 filas de prueba. La salida explicada es el log-odds, es decir, `decision_function`.

### (a) Eficiencia

max<sub>n</sub> |φ0 + Σ<sub>j</sub> φ<sub>j</sub>(x<sub>n</sub>) − f(x<sub>n</sub>)| = **9.70e-08**
(dentro de la tolerancia de 1e-6; el error no es cero y es del orden de la precisión simple, lo que sugiere que la extensión en C++ de TreeSHAP no calcula todo en float64).

### (b) Valor base

| φ0 (`expected_value`) | Media de f en el fondo | Diferencia |
|---|---|---|
| 1.430742 | 1.430742 | 0.00e+00 |

Coinciden: con `interventional` el valor base es el promedio de la salida del modelo sobre el fondo.

### (c) Importancia SHAP frente a permutación

I<sub>j</sub> = mean|φ<sub>j</sub>| sobre la prueba; permutación con AUC, 10 repeticiones.

| # | mean\|φ\| (variable) | I_j | Permutación (variable) | Caída de AUC |
|---|---|---|---|---|
| 1 | `checking_status` | 0.7603 | `checking_status` | 0.1190 |
| 2 | `purpose` | 0.5117 | `credit_amount` | 0.0764 |
| 3 | `credit_amount` | 0.4907 | `duration` | 0.0579 |
| 4 | `duration` | 0.3917 | `credit_history` | 0.0396 |
| 5 | `savings_status` | 0.2963 | `savings_status` | 0.0262 |

Correlación de Spearman entre los vectores completos de las 20 variables: **ρ = 0.645** (p = 0.0021), una concordancia moderada.
Los dos top-5 comparten 4 de 5 variables; sólo aparece en el top-5 de I<sub>j</sub> `purpose` y sólo en
el de permutación `credit_history`.

![Figura 7. Importancia por variable: media de |φ| (izquierda) y caída del AUC por permutación, media ± desviación (derecha). Entre paréntesis, el rango por permutación.](outputs/task2/figures/fig_2_2_importance.png)

*Figura 7. Importancia por variable: media de |φ| (izquierda) y caída del AUC por permutación, media ± desviación (derecha). Entre paréntesis, el rango por permutación.*

### (d) Dos clientes

**Cliente aprobado** (posición 0 del conjunto de prueba; p(good) = 0.866, f(x) = 1.869):

| Variable y valor | φ (log-odds) | Efecto sobre el log-odds |
|---|---|---|
| `savings_status` = no known savings | +0.916 | aumenta |
| `checking_status` = 0<=X<200 | -0.410 | reduce |
| `credit_amount` = 2427 | +0.393 | aumenta |

**Cliente rechazado** (posición 1 del conjunto de prueba; p(good) = 0.087, f(x) = -2.349):

| Variable y valor | φ (log-odds) | Efecto sobre el log-odds |
|---|---|---|
| `duration` = 36 | -0.926 | reduce |
| `savings_status` = no known savings | +0.877 | aumenta |
| `other_payment_plans` = bank | -0.806 | reduce |

En el cliente aprobado, las tres mayores contribuciones son `savings_status` = no known savings (aumenta el log-odds en 0.916), `checking_status` = 0<=X<200 (reduce el log-odds en 0.410) y `credit_amount` = 2427 (aumenta el log-odds en 0.393); el log-odds final (1.869) queda por encima del valor base (1.431). En el cliente rechazado, las tres mayores contribuciones son `duration` = 36 (reduce el log-odds en 0.926), `savings_status` = no known savings (aumenta el log-odds en 0.877) y `other_payment_plans` = bank (reduce el log-odds en 0.806); el log-odds final (-2.349) queda por debajo del valor base (1.431). Los valores categóricos se muestran con su etiqueta original.

![Figura 8. Cascada SHAP del cliente aprobado (p > 0.8).](outputs/task2/figures/fig_2_2_waterfall_aprobado.png)

*Figura 8. Cascada SHAP del cliente aprobado (p > 0.8).*

![Figura 9. Cascada SHAP del cliente rechazado (p < 0.2).](outputs/task2/figures/fig_2_2_waterfall_rechazado.png)

*Figura 9. Cascada SHAP del cliente rechazado (p < 0.2).*

![Figura 10. Gráfico de enjambre SHAP global sobre el conjunto de prueba (el color de las categóricas corresponde a su código ordinal).](outputs/task2/figures/fig_2_2_beeswarm.png)

*Figura 10. Gráfico de enjambre SHAP global sobre el conjunto de prueba (el color de las categóricas corresponde a su código ordinal).*

En el enjambre, las variables ordenadas por I<sub>j</sub> son `checking_status`, `purpose` y `credit_amount`. Para las numéricas de mayor
importancia: en `credit_amount`, no hay una relación monótona clara entre el valor y la contribución (ρ = -0.11); en `duration`, un valor mayor se asocia con menor contribución (ρ = -0.81); en `age`, un valor mayor se asocia con mayor contribución (ρ = +0.81).

## Task 2.3 - Variable duplicada con ruido

La numérica con mayor I<sub>j</sub> es `credit_amount` (I<sub>j</sub> = 0.4907). Se creó `credit_amount_copy` = x + 0.05·σ·ε, con σ = 2822.7 (calculada en el conjunto
completo) y ε ~ N(0, 1) con semilla 42; la correlación entre la variable y su copia es 0.99878. El modelo se reentrenó con 21 variables
(exactitud 0.760, AUC 0.767) y se usaron las mismas filas de fondo.

| Configuración | mean\|φ_orig\| | mean\|φ_copia\| | mean\|φ_orig + φ_copia\| | % del original | corr(φ_orig, φ_copia) |
|---|---|---|---|---|---|
| interventional | 0.5180 | 0.4284 | 0.5457 | 54.7% | -0.30 |
| tree_path_dependent | 0.3812 | 0.3629 | 0.5303 | 51.2% | -0.08 |

I<sub>j</sub> de `credit_amount` en el modelo **sin** copia: 0.4907 (es otro modelo, por lo que no es exactamente comparable).

![Figura 11. Crédito repartido entre la variable y su copia, con ambos métodos de TreeSHAP.](outputs/task2/figures/fig_2_3_duplicate.png)

*Figura 11. Crédito repartido entre la variable y su copia, con ambos métodos de TreeSHAP.*

## Task 2.4 - KernelSHAP frente a TreeSHAP

KernelSHAP (`nsamples` = 2000) sobre las primeras 10 filas de prueba, con fondos anidados de k = 10, 50 y 100 filas; TreeSHAP interventional usa
**el mismo** fondo en cada caso. El MAE se promedia sobre 10 × 20 valores φ.

| k | MAE | Error máximo | KernelSHAP (s) | TreeSHAP (s) | KernelSHAP por instancia (s) | Evaluaciones |
|---|---|---|---|---|---|---|
| 10 | 0.0347 | 0.6239 | 0.70 | 0.0020 | 0.070 | 200,010 |
| 50 | 0.0416 | 0.8252 | 1.98 | 0.0097 | 0.198 | 1,000,010 |
| 100 | 0.0301 | 0.3683 | 3.85 | 0.0189 | 0.385 | 2,000,010 |

El conteo de evaluaciones supera en 10 (una por instancia, para f(x)) el producto nsamples·k·instancias. Con el mismo fondo, TreeSHAP es entre
204 y 347 veces más rápido que KernelSHAP (los tiempos dependen de la máquina y de la corrida).

![Figura 12. MAE de KernelSHAP contra TreeSHAP y tiempos de cómputo según el tamaño del fondo.](outputs/task2/figures/fig_2_4_kernel_vs_tree.png)

*Figura 12. MAE de KernelSHAP contra TreeSHAP y tiempos de cómputo según el tamaño del fondo.*

## Task 2.5 - Shapley exacto escrito a mano

Modelo con seis variables (`duration`, `credit_amount`, `installment_commitment`, `residence_since`, `age`, `existing_credits`), misma partición, semilla e hiperparámetros (exactitud 0.685, AUC 0.634). Con un fondo de
50 filas se enumeraron las 64 coaliciones con máscaras de bits y se evaluó v(S) = media<sub>b</sub> f(x<sub>S</sub>, b<sub>¬S</sub>) en una sola llamada al modelo; los
pesos son |S|!(n−|S|−1)!/n!. Resultados de las primeras 5 filas de prueba (log-odds):

| Cliente (test) | `duration` | `credit_amount` | `installment_commitment` | `residence_since` | `age` | `existing_credits` | Σφ | f(x) − v(∅) |
|---|---|---|---|---|---|---|---|---|
| 0 | -0.4652 | +0.3353 | -0.0531 | -0.1108 | +0.3497 | +0.1798 | +0.2356 | +0.2356 |
| 1 | -1.1706 | +0.0933 | -0.0414 | -0.1200 | -0.3243 | -0.0771 | -1.6400 | -1.6400 |
| 2 | +0.3183 | -1.4394 | +0.0311 | +0.0155 | +0.5182 | -0.0785 | -0.6348 | -0.6348 |
| 3 | +0.5520 | +0.6827 | -0.0529 | -0.1470 | +0.5080 | -0.1110 | +1.4318 | +1.4318 |
| 4 | +0.6726 | -1.5005 | +0.0285 | +0.0134 | +0.9341 | -0.0839 | +0.0643 | +0.0643 |

- Máximo error absoluto contra `TreeExplainer` interventional con el mismo fondo: **2.42e-08**.
- Eficiencia, máx. |Σφ − (v(N) − v(∅))|: **2.22e-16**; además v(N) coincide con f(x) (error 5.6e-17).
- Tiempo del cálculo propio para las 5 instancias: 0.026 s.

![Figura 13. Shapley exacto propio contra TreeSHAP interventional (30 valores).](outputs/task2/figures/fig_2_5_own_vs_shap.png)

*Figura 13. Shapley exacto propio contra TreeSHAP interventional (30 valores).*

El error contra TreeSHAP (2.4e-08) es mucho mayor que el de la eficiencia del cálculo propio
(2.2e-16), lo que es coherente con que TreeSHAP acumule en precisión simple y el cálculo propio en float64.

## Task 2.6 - Análisis

### (a) Aditividad en log-odds frente a probabilidad (cliente rechazado)

Cliente de la posición 1: p(x) = 0.0871, f(x) = -2.3490, φ0 = 1.4307, σ(φ0) = 0.8070.

| Cantidad | Valor |
|---|---|
| Σ<sub>j</sub> φ<sub>j</sub> | -3.7798 |
| f(x) − φ0 | -3.7798 |
| p(x) − σ(φ0) | -0.7199 |
| Σ<sub>j</sub> [σ(φ0 + φ<sub>j</sub>) − σ(φ0)] | -0.7699 |
| [p(x) − σ(φ0)] − Σ<sub>j</sub> [σ(φ0 + φ<sub>j</sub>) − σ(φ0)] | +0.0500 |

Mayores efectos individuales en probabilidad:

| Variable | σ(φ0 + φ_j) − σ(φ0) |
|---|---|
| `duration` | -0.1834 |
| `other_payment_plans` | -0.1558 |
| `checking_status` | -0.1071 |
| `purpose` | -0.1046 |
| `savings_status` | +0.1025 |

La suma de los φ coincide con f(x) − φ0 porque la función explicada es f (el log-odds) y los valores de Shapley son aditivos en esa escala. La probabilidad
es σ(f), y σ no es lineal: σ(a + b) − σ(a) ≠ [σ(φ0 + φ_1) − σ(φ0)] + [σ(φ0 + φ_2) − σ(φ0)] porque cada término evalúa la pendiente de σ en
un punto distinto (φ0 + φ_j) y no en el punto final (f(x) = -2.349). Con estos números, la suma de efectos individuales (-0.7699) sobreestima la magnitud del
cambio real de probabilidad (-0.7199) en 0.0500.

### (b) Mayor desacuerdo entre las dos importancias: `age`

`age` ocupa el puesto 6 por I<sub>j</sub> (0.2886) y el 20 por permutación (-0.0054 ± 0.0096; diferencia de 14 puestos).
I<sub>j</sub> mide cuánto mueve la variable el log-odds de las predicciones (contribución media absoluta, sin mirar si acierta); la permutación mide cuánto
cae el AUC en el conjunto de prueba al romper la relación entre la variable y el resultado.

Evidencia calculada: sus mayores correlaciones de Spearman (entrenamiento) son `housing` (-0.31), `employment` (+0.31), `personal_status` (+0.27); el máximo en valor absoluto es 0.31, bajo, por lo que la correlación con otras variables no explica el desacuerdo.
La desviación de la permutación (0.0096) es al menos tan grande como el valor absoluto de su media (0.0054), así que su efecto sobre el AUC no se distingue de cero y la media es negativa (mezclar la variable no empeora el AUC).
Además, hacen falta 52 de 200 clientes (26.0%) para acumular la mitad de su Σ|φ|, de modo que su efecto está repartido y no concentrado en pocos clientes.
**Hipótesis** (no se probó con un experimento adicional): el modelo usa `age` (53 valores distintos, con lo que los árboles pueden ajustar con ella detalles del entrenamiento) sin que ese uso mejore el AUC en prueba. Con un
conjunto de prueba de 200 clientes y una sola semilla, esta explicación es tentativa.

### (c) Crédito entre la variable y su copia (tabla de 2.3)

En el modelo interventional, `credit_amount` recibe 0.5180 y la copia 0.4284 (54.7% para el original); con `tree_path_dependent`,
0.3812 y 0.3629 (51.2%). El axioma de simetría de Shapley pediría un reparto igual (50 %) si el modelo tratara las dos columnas de forma
simétrica, y el reparto observado se acerca más a ese 50 % con tree_path_dependent (desviación de 1.2 puntos frente a 4.7). No es exacto porque la simetría es una propiedad de la función
explicada y no del método: los árboles eligen en cada división una de las dos columnas casi idénticas, el ruido del 5 % rompe los empates y el f ajustado no es simétrico en ellas.
La correlación entre φ_orig y φ_copia es -0.30 (interventional) y -0.08 (`tree_path_dependent`).

La suma de ambas contribuciones (0.5457 y 0.5303) se compara con I<sub>j</sub> = 0.4907 del modelo sin copia
(+11.2 % y +8.1 %), mientras que cada columna por separado queda por debajo de esa suma. Con `interventional`, el
fondo rompe la dependencia entre columnas (evalúa combinaciones del original de un cliente con la copia de otro, que no ocurren en los datos); con `tree_path_dependent` sólo se usan las
proporciones de entrenamiento (`cover`) a lo largo de cada camino del árbol. Esta última **no** equivale a la esperanza condicional verdadera: sólo captura la dependencia que los
propios cortes del árbol reflejan y la ignora donde las divisiones no la expresan. Para una importancia real, esto significa que variables casi duplicadas se reparten el crédito (aquí, ninguna de las dos alcanza por sí sola la suma) y que la importancia
individual de cada una subestima la del concepto compartido; conviene sumar los grupos correlacionados, como la columna "suma" de la tabla.

### (d) Costo computacional

| k | Evaluaciones del modelo | Tiempo KernelSHAP (s) | Tiempo / evaluación (µs) | MAE |
|---|---|---|---|---|
| 10 | 200,010 | 0.70 | 3.51 | 0.0347 |
| 50 | 1,000,010 | 1.98 | 1.98 | 0.0416 |
| 100 | 2,000,010 | 3.85 | 1.92 | 0.0301 |

Tiempo por evaluación entre k = 50 y k = 100: 1.98 µs y 1.92 µs (diferencia de 3.1%); con k = 10 es 3.51 µs. El tiempo crece de forma aproximadamente lineal con nsamples·k (para k ≥ 50); el costo por evaluación más alto
con k = 10 sugiere un costo fijo por instancia que se amortiza al subir k. El MAE no disminuye de forma monótona con k (0.0347 → 0.0416 → 0.0301); con k = 100 es 13.4% menor que con k = 10.
Con 10 instancias y una sola corrida (el muestreo de KernelSHAP es aleatorio) este experimento no basta para afirmar que más fondo reduzca el error de forma sistemática.

Con un tiempo de `decision_function` sobre 100 filas de 1.119 ms (mediana de 30 repeticiones), el Shapley exacto con 20 variables requiere
2<sup>20</sup> coaliciones × t(100) ≈ **1173 s (0.33 h) por instancia**, frente a 0.385 s de KernelSHAP con k = 100
(3048 veces menos). Con 60 variables y 1 µs por evaluación, 2<sup>60</sup> × 10<sup>−6</sup> s ≈ **3.653e+04 años** por instancia, por lo que el cálculo exacto es inviable
y se necesitan aproximaciones (KernelSHAP) o estructura del modelo (TreeSHAP).

**Prompt utilizado**

Usa German Credit con `fetch_openml("credit-g", version=1)`, codifica `good`=1 y `bad`=0, pasa las 13 categóricas por un `OrdinalEncoder` (tratando los códigos como numéricos) y divide 80/20 estratificado con semilla 42. Entrena un `GradientBoostingClassifier` con 200 árboles y profundidad 3. Para SHAP usa `TreeExplainer` con `feature_perturbation="interventional"` y `model_output="raw"`, de modo que la salida explicada sea el log-odds de `decision_function`, con un fondo de 100 filas del entrenamiento elegidas con la semilla. Verifica la eficiencia y que el valor base sea el promedio del log-odds en el fondo, compara la importancia media |φ| con `permutation_importance` (AUC, 10 repeticiones) usando Spearman, y en KernelSHAP usa exactamente el mismo fondo que en TreeSHAP para que el error compare lo mismo. La fórmula de Shapley del 2.5 la quiero escrita a mano: enumera las 64 coaliciones con máscaras de bits, evalúalas todas en una sola llamada al modelo y usa los pesos |S|!(n−|S|−1)!/n!. Guarda los números en `outputs/task2/results.json` e imprime una tabla de verificaciones OK/FALLO con tolerancias.

**Por qué funcionó este prompt**

Fijamos los detalles que cambian los números en lugar de pedir simplemente "calcula SHAP". Con `model_output="raw"` SHAP explica el log-odds; sin eso explicaría otra salida y la eficiencia en log-odds no cerraría. Nombrar `interventional` frente a `tree_path_dependent` evitó que la librería escogiera una de las dos por defecto sin que lo supiéramos. Usar el mismo fondo en KernelSHAP y en TreeSHAP hizo que el MAE midiera sólo la aproximación y no un cambio de referencia. Pedir las 64 coaliciones vectorizadas en una sola llamada permitió que el cálculo exacto corriera en segundos, y las verificaciones automáticas con tolerancias detectaron errores antes de escribir el informe.

<!-- INFORME_TASK2:FIN -->

<!-- INFORME_TASK3:INICIO (generado por scripts/run_task3.py; no editar a mano) -->

# Task 3

## Configuración

- Salida explicada: P(good) del mismo `GradientBoostingClassifier` y los mismos tres clientes del Task 2 (`credit.build_base` y `select_clients`).
- Semilla aleatoria: **42**. Muestras por explicación: N = 5000; variables conservadas: K = 5; λ = 1;
  σ_rel = 0.75 con d' = 20, es decir σ = σ_rel·√d' = 3.3541.
- Variable ausente: z'_j ~ Bernoulli(0.5) de forma independiente por muestra y variable; con z'_j = 1 se conserva el valor del cliente y
  con z'_j = 0 se toma un valor al azar de la columna j del conjunto de entrenamiento. Peso π = exp(−D²/σ²), con D² igual al número de variables ausentes.
- Ajuste: regresión ridge ponderada en forma cerrada con `np.linalg.solve`; el intercepto **no se penaliza**. Se ajustan las 20 variables, se
  conservan las K de mayor |w| y se reajusta sólo con ellas; R²_w se calcula sobre ese reajuste con las mismas muestras y pesos.
- Versiones: scikit-learn 1.9.0, NumPy 2.5.2, pandas 3.0.5, Python 3.14.7.

## Task 3.1 - LIME desde cero

**Cliente aprobado** (posición 0 del conjunto de prueba; p(good) = 0.866; intercepto = 0.669; R²_w = 0.172; N_ef = 4808.4):

| Variable | Valor del cliente | Coeficiente |
|---|---|---|
| `savings_status` | no known savings | +0.141 |
| `credit_amount` | 2427 | +0.074 |
| `checking_status` | 0<=X<200 | -0.063 |
| `residence_since` | 2 | -0.038 |
| `duration` | 18 | +0.033 |


**Cliente rechazado** (posición 1 del conjunto de prueba; p(good) = 0.087; intercepto = 0.551; R²_w = 0.227; N_ef = 4808.4):

| Variable | Valor del cliente | Coeficiente |
|---|---|---|
| `savings_status` | no known savings | +0.141 |
| `other_payment_plans` | bank | -0.132 |
| `checking_status` | 0<=X<200 | -0.112 |
| `duration` | 36 | -0.106 |
| `age` | 29 | -0.102 |


**Cliente dudoso** (posición 87 del conjunto de prueba; p(good) = 0.504; intercepto = 0.635; R²_w = 0.279; N_ef = 4808.4):

| Variable | Valor del cliente | Coeficiente |
|---|---|---|
| `checking_status` | no checking | +0.241 |
| `duration` | 36 | -0.154 |
| `credit_amount` | 909 | -0.106 |
| `purpose` | new car | -0.074 |
| `age` | 36 | +0.044 |


![Figura 14. Coeficientes LIME (top-5) de los tres clientes; R²_w del reajuste en el título de cada panel.](outputs/task3/figures/fig_3_1_coefs.png)

*Figura 14. Coeficientes LIME (top-5) de los tres clientes; R²_w del reajuste en el título de cada panel.*

En el cliente rechazado (p = 0.087), `other_payment_plans`, `checking_status`, `duration` y `age` tienen coeficiente negativo, es decir, conservar su valor en lugar de sustituirlo por uno de entrenamiento baja P(good) en la aproximación local; el de mayor magnitud es `other_payment_plans` = bank (-0.132). En sentido contrario, `savings_status` = no known savings (+0.141) tiene coeficiente positivo. R²_w va de 0.172 (cliente aprobado) a 0.279 (cliente dudoso): en los tres clientes el modelo lineal con 5 variables explica menos de la mitad de la varianza ponderada de P(good), de modo que los coeficientes describen sólo una parte del comportamiento local del modelo. Variables presentes en el top-5 de dos clientes: aprobado y rechazado: `checking_status`, `duration` y `savings_status`; aprobado y dudoso: `checking_status`, `credit_amount` y `duration`; rechazado y dudoso: `age`, `checking_status` y `duration`.

## Task 3.2 - Estabilidad

Cliente rechazado (posición 1); para cada combinación de N y σ_rel se corrieron 20 explicaciones con las semillas
0 a 19 y se promedió el Jaccard del top-5 sobre los 190 pares.

**Jaccard medio** (filas N, columnas σ_rel):

| N \ σ_rel | 0.5 | 0.75 | 1.5 |
|---|---|---|---|
| 500 | 0.6981 | 0.6992 | 0.7216 |
| 2000 | 0.7689 | 0.7872 | 0.8043 |
| 5000 | 0.9105 | 0.9368 | 0.9368 |

**N_ef medio** (filas N, columnas σ_rel):

| N \ σ_rel | 0.5 | 0.75 | 1.5 |
|---|---|---|---|
| 500 | 407.9 | 480.3 | 498.7 |
| 2000 | 1637.9 | 1922.0 | 1995.0 |
| 5000 | 4098.9 | 4805.9 | 4987.6 |

![Figura 15. Jaccard medio y tamaño efectivo de muestra N_ef según N y σ_rel (cliente rechazado).](outputs/task3/figures/fig_3_2_stability.png)

*Figura 15. Jaccard medio y tamaño efectivo de muestra N_ef según N y σ_rel (cliente rechazado).*

El Jaccard medio crece con N en las tres anchuras (por ejemplo, con σ_rel = 0.75 pasa de 0.699 con N = 500 a 0.937 con N = 5000); con N fijo no disminuye al aumentar σ_rel; el valor más alto es 0.937 (N = 5000, σ_rel = 1.5) y el más bajo 0.698 (N = 500, σ_rel = 0.5). N_ef equivale a una fracción de N que casi no cambia con N (variación máxima de 0.4% de N dentro de cada σ_rel); promedia 81.8% de N con σ_rel = 0.5, 96.1% con 0.75 y 99.8% con 1.5, de modo que un kernel más ancho reparte el peso entre más muestras.

**Prompt utilizado**

Pendiente

**Por qué funcionó este prompt**

Pendiente

<!-- INFORME_TASK3:FIN -->
