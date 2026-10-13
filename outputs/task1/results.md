# Task 1 - resultados (generado automáticamente)

Semilla `42` - torch 2.14.1 - torchvision 0.29.1 - Python 3.11.2

## Task 1.1 - Modelo, preprocesamiento, tres imágenes y top-5

**Qué hace el código:** Carga ResNet-18 con pesos de ImageNet en modo eval; preprocesa (resize 256, recorte central 224, normalización por canal); predice el top-5 de las tres imágenes (a: objeto claro, b: gato y perro, c: objeto fuera de contexto) y transforma el recuadro de (c) a coordenadas de 224x224.

**Qué poner en el Word:**

- Tabla con fuente, autor y licencia de las 3 imágenes (indicar la fuente es obligatorio).
- Una tabla top-5 (clase + probabilidad) por imagen.
- Las coordenadas del recuadro de (c) en 224x224.
- Figura fig_1_1_predictions.png.
- Mencionar la semilla y las versiones de librerías.

**Imagen (a)** - `a_golden_retriever.jpg` - MichaelMcPhee (Wikimedia Commons) - [CC BY 3.0](https://commons.wikimedia.org/wiki/File:Callie_the_golden_retriever_puppy.jpg)

| # | clase ImageNet | probabilidad |
|---|---|---|
| 1 | golden retriever (#207) | 0.9948 |
| 2 | Tibetan mastiff (#244) | 0.0016 |
| 3 | cocker spaniel (#219) | 0.0008 |
| 4 | Saluki (#176) | 0.0007 |
| 5 | Labrador retriever (#208) | 0.0005 |

**Imagen (b)** - `b_cat_and_dog.jpg` - S. J. Pyrotechnic (Wikimedia Commons) - [CC BY-SA 2.0](https://commons.wikimedia.org/wiki/File:Cute_cat_and_dog_1.jpg)

| # | clase ImageNet | probabilidad |
|---|---|---|
| 1 | Chihuahua (#151) | 0.4449 |
| 2 | tabby (#281) | 0.1335 |
| 3 | French bulldog (#245) | 0.1230 |
| 4 | Egyptian cat (#285) | 0.0525 |
| 5 | tiger cat (#282) | 0.0395 |

**Imagen (c)** - `c_sailboat_on_trailer.jpg` - Jwkane (Wikimedia Commons) - [CC BY-SA 3.0](https://commons.wikimedia.org/wiki/File:Catalina_22_sailboat_on_a_trailer.jpg)

| # | clase ImageNet | probabilidad |
|---|---|---|
| 1 | trimaran (#871) | 0.2054 |
| 2 | catamaran (#484) | 0.1791 |
| 3 | projectile (#744) | 0.0970 |
| 4 | missile (#657) | 0.0863 |
| 5 | amphibian (#408) | 0.0654 |

**Recuadro de la imagen (c) en 224x224** (x0, y0, x1, y1) = (0.0, 37.3, 224.0, 144.0)  [definido en la imagen original como (135, 150, 880, 450)]

Figuras: `figures/fig_1_1_predictions.png`

## Task 1.2 - Grad-CAM desde cero sobre layer4

**Qué hace el código:** Hook de avance (guarda A, 512x7x7) y hook de retroceso (guarda dy^c/dA); se retropropaga desde el logit (antes del softmax); alpha_k = promedio espacial del gradiente; L = ReLU(sum alpha_k A^k), interpolación bilineal a 224x224 y normalización min-max. Mapa de la clase top-1 de cada imagen y, en (b), también el de la clase del segundo objeto.

**Qué poner en el Word:**

- La figura fig_1_2_gradcam.png (mapas superpuestos a la imagen).
- Una descripción breve de la implementación (hooks, logit, alpha, ReLU, bilineal, min-max).
- Un comentario sobre dónde cae el mapa en cada imagen.

Mapas generados para: imagen (a): golden retriever (#207); imagen (b): Chihuahua (#151), tabby (#281); imagen (c): trimaran (#871)

Figuras: `figures/fig_1_2_gradcam.png`

## Task 1.3 - CAM directo desde los pesos de fc y verificación contra Grad-CAM

**Qué hace el código:** (a) Calcula CAM sin gradientes; (b) verifica que (1/Z) sum M + b = logit; (c) verifica Z·L = M; (d) compara ambos mapas tras ReLU y min-max.

**Qué poner en el Word:**

- Tabla del inciso (b): ambos valores y el error absoluto por imagen.
- Tabla del inciso (c): error absoluto máximo por imagen.
- Tabla del inciso (d): error absoluto máximo por imagen.
- Figura fig_1_3_cam_vs_gradcam.png y una conclusión (coinciden hasta precisión float32).

**(a)** M^c_ij = sum_k w^c_k A^k_ij calculado con los pesos de `fc`, sin gradientes (mapas en la figura).

**(b)** Descomposición del puntaje: (1/Z) sum M + b_c debe igualar el logit y^c (Z = 49).

| imagen | clase | logit y^c | (1/Z) sum M + b_c | error absoluto |
|---|---|---|---|---|
| a | golden retriever (#207) | 18.294334 | 18.294336 | 1.91e-06 |
| b | Chihuahua (#151) | 11.057706 | 11.057705 | 9.54e-07 |
| c | trimaran (#871) | 10.640809 | 10.640808 | 9.54e-07 |

**(c)** Error absoluto máximo entre Z·L^c (sin ReLU ni normalización) y M^c sobre las 49 posiciones.

| imagen | clase | error abs. máximo |
|---|---|---|
| a | golden retriever (#207) | 1.14e-05 |
| b | Chihuahua (#151) | 7.63e-06 |
| c | trimaran (#871) | 5.72e-06 |

**(d)** Error absoluto máximo entre ambos mapas tras ReLU + normalización min-max.

| imagen | clase | error abs. máximo |
|---|---|---|
| a | golden retriever (#207) | 4.17e-07 |
| b | Chihuahua (#151) | 2.38e-07 |
| c | trimaran (#871) | 1.79e-07 |

Figuras: `figures/fig_1_3_cam_vs_gradcam.png`

## Task 1.4(a) - alpha vs w/Z (imagen a, cinco canales de mayor peso)

**Qué hace el código:** Compara numéricamente el alpha de Grad-CAM con w^c_k / Z para los cinco canales con mayor peso en fc.

**Qué poner en el Word:**

- Tabla con canal, w, alpha, w/Z y error relativo.
- Explicación: dy^c/dA^k_ij = w_k/Z es igual en todas las posiciones (i, j) porque el pooling global promedia con el mismo peso 1/Z; al promediar Z gradientes idénticos alpha = w/Z.
- Explicación de por qué el error es ~1e-8 y no 0: precisión de punto flotante (float32).
- Figura fig_1_4a_alpha_vs_w.png.

| canal | w | alpha | w/Z | error absoluto | error relativo |
|---|---|---|---|---|---|
| 28 | 0.288863 | 0.0058951629 | 0.0058951633 | 3.99e-10 | 6.77e-08 |
| 242 | 0.255639 | 0.0052171177 | 0.0052171179 | 2.00e-10 | 3.83e-08 |
| 156 | 0.246214 | 0.0050247773 | 0.0050247771 | 1.81e-10 | 3.59e-08 |
| 473 | 0.223402 | 0.0045592198 | 0.0045592198 | 1.90e-11 | 4.17e-09 |
| 497 | 0.210018 | 0.0042860918 | 0.0042860915 | 2.28e-10 | 5.32e-08 |

Figuras: `figures/fig_1_4a_alpha_vs_w.png`

## Task 1.4(b) - IoU entre los mapas de las dos clases (imagen b)

**Qué hace el código:** Toma el 20 % de píxeles de mayor valor de cada mapa (clase perro y clase gato) y calcula la intersección sobre la unión.

**Qué poner en el Word:**

- El IoU y los conteos de píxeles (intersección, unión).
- Respuesta: ¿el mapa discrimina entre las dos clases? (con el número del IoU).
- Límite de resolución: cada celda 7x7 cubre 32x32 píxeles, así que objetos pequeños o cercanos entre sí (< ~32 px) no se pueden separar ni delimitar bien.
- Figura fig_1_4b_iou.png.

- Clases: Chihuahua (#151) vs tabby (#281)
- Probabilidades: Chihuahua (#151): 0.4449, tabby (#281): 0.1335
- IoU del top-20 % = **0.0003** (intersección 7 px, unión 20063 px, 10035 px por conjunto)

Figuras: `figures/fig_1_4b_iou.png`

## Task 1.4(c) - Masa dentro del recuadro y prueba causal (imagen c)

**Qué hace el código:** Calcula qué fracción del mapa normalizado cae dentro del recuadro y reemplaza por gris todo lo que queda fuera del recuadro para medir el cambio en la probabilidad de la clase. (Extra: control con el objeto en gris.)

**Qué poner en el Word:**

- Fracción de masa dentro del recuadro (y el área del recuadro como referencia).
- Probabilidad original, con fondo gris y su cambio.
- Respuesta: ¿coinciden el mapa y la prueba causal en si el modelo usa el fondo? y conclusión sobre si hay evidencia de un atajo (con los números).
- Figura fig_1_4c_causal.png.

- Clase analizada: trimaran (#871)
- Fracción de masa del mapa dentro del recuadro: **0.7063** (el recuadro cubre 0.4777 de la imagen)
- p(clase) original: 0.2054
- p(clase) con todo lo de fuera del recuadro en gris: 0.0371 (cambio -0.1683)
- [control extra] p(clase) con el objeto en gris y el fondo intacto: 0.0002 (cambio -0.2052)
- Top-1 tras poner el fondo en gris: catamaran (0.7663)

Figuras: `figures/fig_1_4c_causal.png`

