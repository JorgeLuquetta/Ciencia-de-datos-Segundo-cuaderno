# Cuaderno de Ciencia de Datos — Segundo corte

Resumen de los cuadernos (Jupyter / Google Colab) de la carpeta [`Apuntes ciencia de datos`](<Apuntes ciencia de datos/>), correspondientes al segundo corte del curso de Ciencia de Datos Avanzada (UNAB, sexto semestre). El material es del profesor Alfredo Díaz Claro.

Los temas avanzan de las **redes neuronales convolucionales (CNN)** a la **visión por computador** (detección, segmentación, puntos clave), luego a los **autoencoders** (imágenes y señales ECG) y terminan con los **modelos generativos** (GAN y Stable Diffusion).

---

## Índice por tema

| # | Tema | Cuadernos |
|---|------|-----------|
| 1 | Fundamentos de CNN | 7, 9 Convolución, 8 A2, 9 Taller flores |
| 2 | Aprendizaje por transferencia | 8 |
| 3 | Detección, segmentación y puntos clave | 9 YOLO, YOLOv11, 9A, 9B, 9C, 9D |
| 4 | Autoencoders | 14, 15A, 15B, 15C, 15D |
| 5 | Modelos generativos | 19 GAN, GAN Panini, Stable Diffusion |

---

## 1. Fundamentos de redes neuronales convolucionales

### [Cuaderno 7 — CNN: Redes Neuronales Convolucionales](<Apuntes ciencia de datos/Avanzada_Cuaderno_7_CNN_Redes_Neuronales_Convolucionales_ 1.ipynb>)
Introducción a las CNN y sus componentes esenciales (convolución, pooling, capas densas). Repasa las utilidades de Keras/TensorFlow para visión: carga y preprocesamiento de imágenes, aumentación, segmentación, modelos preentrenados y normalización.
- **Taller:** CNN entrenada sobre **CIFAR-10** (10 clases), con curvas de entrenamiento y evaluación.
- Predicción sobre una imagen externa (un bus de Metrolínea) y persistencia del modelo (guardar/cargar, arquitectura en JSON).
- *Librerías:* TensorFlow/Keras, NumPy, Matplotlib, PIL.

### [Cuaderno 9 — Convolución (Sobel + Max Pooling)](<Apuntes ciencia de datos/Avanzada_Cuaderno_9_Convolucion.ipynb>)
Explica "a mano" qué hace una capa convolucional: convierte la imagen de un gato a escala de grises, aplica los kernels **Sobel Gx y Gy** (detección de bordes), calcula la magnitud del gradiente y aplica **Max Pooling 2×2**, mostrando todos los mapas intermedios.
- *Librerías:* NumPy, SciPy, PIL, Matplotlib.

### [Cuaderno 8 A2 — CNN para dígitos escritos a mano (MNIST)](<Apuntes ciencia de datos/Avanzada_Cuaderno_8_A2_CNN_neuronales_convolucionales_Números_Modo_2_(Uso_de_RNC).ipynb>)
Clasificador de dígitos con una CNN sobre **MNIST** (`tensorflow_datasets`): arquitectura, compilación, agrupación y mezclado de lotes, entrenamiento, guardado de arquitectura y pesos (con conversión opcional a TensorFlow.js). Termina descargando una imagen, preprocesándola con OpenCV y prediciendo el dígito.
- *Librerías:* TensorFlow, TFDS, OpenCV, Matplotlib.

### [Cuaderno 9 — Taller aplicado: clasificación de imágenes (flores)](<Apuntes ciencia de datos/Avanzada_Cuaderno_9_CNN_Taller_aplicado_Clasificación_de_imagenes_con_Tensorflow_y_Keras.ipynb>)
Clasificación del dataset de **flores** con `image_dataset_from_directory`: división entrenamiento/validación, traducción de clases al español, optimización del pipeline (`cache`, `shuffle`, `prefetch`), normalización con `Rescaling`, entrenamiento de la CNN, diagnóstico de **overfitting** y su mitigación con **aumentación de datos** y Dropout.
- *Librerías:* TensorFlow/Keras, PIL, pathlib.

---

## 2. Aprendizaje por transferencia

### [Cuaderno 8 — Transferencia de aprendizaje y Fine-Tuning](<Apuntes ciencia de datos/Avanzada_Cuaderno_8_CNN_Transferencia_de_aprendizaje.ipynb>)
Teoría de *transfer learning*, *fine-tuning* y modelos en cascada, y la diferencia entre ellos. Incluye una tabla de modelos preentrenados populares, tamaños de entrada y recomendaciones para hacer el modelo más eficiente.
- **Taller 1:** extracción de características y predicción con **VGG16** (innovaciones, ventajas, desventajas y usos).
- **Proyecto:** clasificación de **gatos vs. perros** con un modelo preentrenado (dataset descargado con `kagglehub`).
- *Librerías:* TensorFlow/Keras, kagglehub.

---

## 3. Visión por computador: detección, segmentación y puntos clave

### [Cuaderno 9 — Detección de objetos con YOLO y Transfer Learning](<Apuntes ciencia de datos/Avanzada_Cuaderno_9_CNN_Detección_de_Objetos_con_YOLOv8_y_Transfer_LearningL_.ipynb>)
### [Detección de objetos con YOLOv11](<Apuntes ciencia de datos/Detección_de_objetos_con_YOLOv11.ipynb>)
Los dos cuadernos tienen el mismo contenido (el segundo es otra versión/ejecución del primero).
- **Teoría (9.1):** qué es la detección de objetos (clasificación + localización), modelos comunes (R-CNN, SSD, YOLO…), datasets (COCO, Pascal VOC…), herramientas de etiquetado y formatos de *bounding boxes*.
- **Taller (9.2):** *transfer learning* con **YOLOv11** (Ultralytics) sobre el dataset **PPE Detection** (elementos de protección personal) de Roboflow: descarga y validación del dataset, edición de `dataset.yaml`, entrenamiento, métricas e inferencia.
- *Librerías:* Ultralytics, PyTorch, OpenCV.

### [Cuaderno 9A — Detección de matrículas vehiculares](<Apuntes ciencia de datos/Avanzada_Cuaderno_9A_CNN_matriculas_de_vehiculos.ipynb>)
Proyecto que detecta **placas de vehículos** con un modelo YOLO (dataset de Roboflow) y lee el texto de la placa con **EasyOCR**. Incluye descarga del modelo (`.pt` o zip) y una reflexión final.
- *Librerías:* Ultralytics, EasyOCR, Roboflow, OpenCV.

### [Cuaderno 9B — PPE: entrenar un detector de objetos YOLO](<Apuntes ciencia de datos/Avanzada_Cuaderno_9B_PPE__CNN_Entrenar_Detección_de_objeto_yolov8_Transfer_L_(1).ipynb>)
Guía práctica para entrenar un **detector personalizado con YOLO11** en Colab (GPU): carga del dataset con anotaciones YOLO, configuración de `data.yaml`, entrenamiento, monitoreo con **TensorBoard**, pruebas sobre imágenes y video, y exportación a **TFLite, ONNX, TorchScript y TensorFlow**.

### [Cuaderno 9C — Segmentación de instancias con YOLO11](<Apuntes ciencia de datos/Avanzada_Cuaderno_9C_train_yolo11_instance_segmentation.ipynb>)
Basado en los notebooks de Roboflow. Explica qué es la **segmentación de instancias** y cómo se anota con polígonos en Roboflow.
- Inferencia con un modelo preentrenado en COCO (CLI, SDK y visualización con **Supervision**).
- Entrenamiento con dataset propio, **Early Stopping**, validación, inferencia en cuadrícula, despliegue en Roboflow y exportación para usar con OpenCV.

### [Cuaderno 9D — Detección de puntos clave](<Apuntes ciencia de datos/Avanzada_Cuaderno_9D_Deteccion_puntos_clave2.ipynb>)
Adaptación al español del ejemplo de Keras (Sayak Paul). Construye un detector de **puntos clave (keypoints) en perros** con el dataset **StanfordExtra**: estructura de las anotaciones JSON, generador de datos con aumento (`imgaug`), normalización de coordenadas y un modelo **MobileNetV2** completamente convolucional con *transfer learning*.
- *Librerías:* Keras, imgaug, pandas, scikit-learn.

---

## 4. Autoencoders

### [Cuaderno 14 — Introducción a los Autoencoders](<Apuntes ciencia de datos/Avanzada_Cuaderno_14_Introducción_a_los_Autoencoder.ipynb>)
Qué son los autoencoders (aprendizaje no supervisado), el **encoder**, el **decoder** y el **espacio latente**, y para qué sirven.
- **Ejemplo 1:** autoencoder básico e híbrido (Conv2D + Dense) sobre **Fashion MNIST**, con visualización de las reconstrucciones.
- **Ejemplo 2:** autoencoder convolucional para **eliminar ruido** en imágenes.

### [Cuaderno 15A — Introducción al procesamiento de señales ECG](<Apuntes ciencia de datos/Avanzada_Cuaderno_15A_ECG_Introducción_procesamiento.ipynb>)
Práctica guiada (pasos 1 a 23 y más) para analizar un **electrocardiograma** del registro 23 de **PhysioNet** con la librería `wfdb`: frecuencia de muestreo, canales, duración, gráficas completas y con zoom, anotaciones de latidos y su significado, superposición sobre la señal, y cálculo de la frecuencia cardíaca y detección de arritmias.
- Existe una copia, [`... (1).ipynb`](<Apuntes ciencia de datos/Avanzada_Cuaderno_15A_ECG_Introducción_procesamiento (1).ipynb>), con el mismo contenido pero sin las salidas ejecutadas (por eso pesa menos).

### [Cuaderno 15B — Autoencoder para ECG](<Apuntes ciencia de datos/Avanzada_Cuaderno_15B_ECG_Autoencoder.ipynb>)
Detección de anomalías en electrocardiogramas: carga un dataset de ECG (latidos normales vs. anómalos), separa etiquetas, escala los datos con scikit-learn, visualiza latidos normales y anormales y entrena un autoencoder sólo con latidos normales.
- ⚠️ El archivo `.ipynb` **está dañado** (JSON incompleto) y es posible que Jupyter/Colab no lo abra. Conviene restaurarlo desde el historial de git o descargarlo de nuevo.

### [Cuaderno 15C — Autoencoder ECG con la base LTAFDB](<Apuntes ciencia de datos/Avanzada_Cuaderno_15C_ECG_Autoencoder_LTAFDB_es.ipynb>)
Versión completa de la detección de anomalías cardíacas con datos de PhysioNet:
1. Segmenta los latidos (ventanas de 140 muestras) a partir de las anotaciones.
2. Normaliza a [0, 1] y entrena el autoencoder **sólo con latidos normales**.
3. Usa el **error de reconstrucción** y un umbral para clasificar latidos como normales o anómalos.
4. Evalúa con métricas, **matriz de confusión** y **curva ROC/AUC**.
- *Librerías:* wfdb, TensorFlow, scikit-learn, seaborn.

### [Cuaderno 15D — Limpieza de imágenes con autoencoders](<Apuntes ciencia de datos/Avanzada_Cuaderno_15D_Limpieza_imagenes_2_autoencoders.ipynb>)
Proyecto completo de **denoising**: plantea el problema, lee y explora el set de datos, le agrega ruido, implementa y entrena el autoencoder y prueba la limpieza en imágenes nuevas.

---

## 5. Modelos generativos

### [Cuaderno 19 — GAN en Keras (MNIST)](<Apuntes ciencia de datos/Avanzada_Cuaderno_19__GAN_keras.ipynb>)
Introduce las **Redes Generativas Adversarias** y la diferencia entre aprendizaje discriminativo y generativo. En el taller se construyen el **generador** y el **discriminador**, sus funciones de pérdida y optimizadores, un paso de entrenamiento personalizado (`tf.GradientTape`) y el entrenamiento completo, que genera dígitos MNIST cada vez más realistas.

### [GAN FIFA Panini — Taller práctico DCGAN](<Apuntes ciencia de datos/CCD_UNAB_GAN_Panini_FIFA_2026.ipynb>)
Taller aplicado: entrena una **DCGAN** con imágenes de futbolistas FIFA, con aumentación de datos, una arquitectura mejorada y *label smoothing*. Después el estudiante sube su propia foto, genera su "versión futbolista" y la monta en una **tarjeta coleccionable estilo Panini del Mundial 2026**. Termina con preguntas de reflexión.

### [Taller — Generación de imágenes con Stable Diffusion](<Apuntes ciencia de datos/Avanzada_Taller_Stable_Diffusion (1).ipynb>)
Taller con 🤗 **Hugging Face Diffusers** en Google Colab (GPU), organizado en módulos:
1. Conceptos: qué es Stable Diffusion, cómo funciona y los componentes del pipeline (VAE, U-Net, codificador de texto, scheduler).
2. Verificación del entorno e instalación.
3. Text-to-Image con `StableDiffusionPipeline` (también **SD 3.5 Medium**).
4. **Prompt engineering** y *negative prompts*.
5. Parámetros avanzados: `guidance_scale` (CFG), `num_inference_steps` y semillas.
6. Schedulers (samplers), **Image-to-Image** y manejo de memoria de la GPU.
7. Ejercicio libre y recursos para seguir aprendiendo.

---

## Notas sobre la carpeta

- Casi todos los cuadernos están pensados para ejecutarse en **Google Colab** (usan `google.colab`, montaje de Drive y GPU).
- Las carpetas `ecg-data/` y `-p/` están vacías. `-p/` parece haberse creado por error con un `mkdir -p` en Windows y se puede borrar.
- Archivos que se repiten: `Cuaderno 9 YOLO` / `Detección_de_objetos_con_YOLOv11` y `15A` / `15A (1)`.
- Los cuadernos de Stable Diffusion (~14 MB) y GAN Panini (~7.5 MB) pesan mucho porque guardan las imágenes generadas dentro del archivo.
