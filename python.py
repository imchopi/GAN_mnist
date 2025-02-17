import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os

# Verificar GPU
if tf.config.list_physical_devices('GPU'):
    print("TensorFlow usando la GPU.")
else:
    print("TensorFlow usando solo la CPU.")

# Variables
BUFFER_SIZE = 60000
BATCH_SIZE = 256
LATENT_DIM = 100
EPOCHS = 100

# Cargar el dataset Fashion MNIST
# Como solo necesitamos los datos de entrenamiento, poniendo _ como en varias ocasiones hicimos en actividades anteriores, no tenemos en cuenta lo demás.
(train_images, _), (_, _) = keras.datasets.fashion_mnist.load_data()

# Al normalizar, conseguimos luego poder usar tanh
train_images = (train_images.astype('float32') - 127.5) / 127.5
train_images = np.expand_dims(train_images, axis=-1)  # Añadir dimensión de canal, esto lo hacemos para la escala de grises

# Crear dataset de entrenamiento, importante mezclar los datos para quitar posibles sesgos 
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Definir el generador
def build_generator():
    model = keras.Sequential([
        # Este paso se puede hacer dentro de la capa primera con input shape, pero nos salía un warning que indicaba que no era viable,q ue se hiciera una capa aparte
        layers.Input(shape=(LATENT_DIM,)),
        layers.Dense(7 * 7 * 256, use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((7, 7, 256)),

        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model

generator = build_generator()

# Definir el discriminador
def build_discriminator():
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(1e-4))

# Definir la GAN
# Vi que chatgpt y deepseek, sugerían que hiciera una clase, me pareció buena idea hacerlo para tener mas flexibilidad para modularizar el código
class GAN(keras.Model):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, g_optimizer, d_optimizer, loss_fn):
        super(GAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, LATENT_DIM])
        fake_images = self.generator(noise, training=True)

        real_labels = tf.ones((batch_size, 1))
        fake_labels = tf.zeros((batch_size, 1))

        with tf.GradientTape() as d_tape:
            real_loss = self.loss_fn(real_labels, self.discriminator(real_images, training=True))
            fake_loss = self.loss_fn(fake_labels, self.discriminator(fake_images, training=True))
            d_loss = real_loss + fake_loss
        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        with tf.GradientTape() as g_tape:
            fake_images = self.generator(noise, training=True)
            fake_preds = self.discriminator(fake_images, training=True)
            g_loss = self.loss_fn(real_labels, fake_preds)
        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        return {"d_loss": d_loss, "g_loss": g_loss}

# Entrenar la GAN
gan = GAN(generator, discriminator)
gan.compile(
    g_optimizer=keras.optimizers.Adam(1e-4),
    d_optimizer=keras.optimizers.Adam(1e-4),
    loss_fn=keras.losses.BinaryCrossentropy()
)

# Guardar los modelos en archivos .h5
generator.save('generator_model.h5')
discriminator.save('discriminator_model.h5')

# Entrenar con todos los lotes disponibles
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    for real_images in train_dataset:
        gan.train_step(real_images)

# Generar y guardar imágenes
def generate_and_save_images(model, num_images=5, save_dir='generated_images'):
    os.makedirs(save_dir, exist_ok=True)
    noise = tf.random.normal([num_images, LATENT_DIM])
    generated_images = model(noise, training=False)
    
    for i in range(num_images):
        img = (generated_images[i, :, :, 0] * 127.5 + 127.5).numpy().astype(np.uint8)  # Convertir a uint8
        img = np.expand_dims(img, axis=-1)  # Añadir dimensión de canal
        img_resized = tf.image.resize(img, (28, 28)).numpy()  # Asegurar tamaño 28x28, como hicimos en otros ejemplos con MNIST
        plt.imsave(os.path.join(save_dir, f'generated_{i}.png'), img_resized[:, :, 0], cmap='gray') # Ayuda de DeepSeek

# Generar y guardar imágenes después del entrenamiento
generate_and_save_images(generator)