# Adrián Perogil Fernández

# Generación de Imágenes de Prendas y Accesorios usando una GAN

Este repositorio contiene la implementación de una **Red Generativa Antagónica (GAN)** para generar imágenes de prendas y accesorios utilizando el dataset **Fashion MNIST**. El objetivo es entrenar un modelo generativo que pueda crear imágenes realistas de artículos de moda a partir de un espacio latente de ruido aleatorio.

## Descripción del Problema

El problema consiste en entrenar una GAN para generar imágenes de prendas y accesorios a partir del dataset **Fashion MNIST**, que contiene 60,000 imágenes en escala de grises de 28x28 píxeles. La GAN está compuesta por dos redes neuronales:
1. **Generador**: Crea imágenes a partir de un vector de ruido aleatorio.
2. **Discriminador**: Intenta distinguir entre imágenes reales (del dataset) y falsas (generadas por el generador).

El entrenamiento de la GAN implica un proceso de competición entre estas dos redes, donde el generador aprende a crear imágenes cada vez más realistas, mientras que el discriminador mejora su capacidad para detectar imágenes falsas.

## Recursos del Repositorio

2. **Modelos Entrenados**:
   - **Generador**: [generator_model.h5](generator_model.h5).
   - **Discriminador**: [discriminator_model.h5](discriminator_model.h5).

3. **Imágenes Generadas**:
   - Una muestra de las imágenes generadas por la GAN se encuentra en la carpeta [generated_images](generated_images/).

## Requisitos

Para ejecutar el código, necesitas las siguientes dependencias:
- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib

Recomendado hacer un env:
```bash
python -m venv gan_mnist
venv/Scripts/activate
```

Puedes instalar las dependencias usando el siguiente comando:
```bash
pip install -r requeriments.txt
```

Por úlitmo ejecutamos:
```bash
python python.py
```