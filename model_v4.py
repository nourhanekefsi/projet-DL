import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

# Constants
DATASET_DIR = "/kaggle/input/transfer-font-style-texts/font_transfer_dataset/pairs"
MODEL_DIR = "/kaggle/working/pix2pix_model"
os.makedirs(MODEL_DIR, exist_ok=True)
IMG_HEIGHT = 256
IMG_WIDTH = 256
BUFFER_SIZE = 400
BATCH_SIZE = 10
EPOCHS = 400

# Function to load and preprocess a single image
def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)  # Grayscale
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = (image - 127.5) / 127.5  # Normalize to [-1, 1]
    return image

# Load dataset as tf.data.Dataset
def load_data(image_dir):
    input_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if "input" in f])
    target_paths = sorted([os.path.join(image_dir, f.replace("input", "target")) for f in os.listdir(image_dir) if "input" in f])
    inputs = [load_image(p) for p in input_paths]
    targets = [load_image(p) for p in target_paths]
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
    return dataset

# Data augmentation
def data_augmentation(input_image, target_image):
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        target_image = tf.image.flip_left_right(target_image)
    return input_image, target_image

# Build the generator (U-Net)
def build_generator():
    inputs = layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 1])

    # Encoder
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
        downsample(128, 4),  # (bs, 64, 64, 128)
        downsample(256, 4),  # (bs, 32, 32, 256)
        downsample(512, 4),  # (bs, 16, 16, 512)
        downsample(512, 4),  # (bs, 8, 8, 512)
        downsample(512, 4),  # (bs, 4, 4, 512)
        downsample(512, 4),  # (bs, 2, 2, 512)
        downsample(512, 4),  # (bs, 1, 1, 512)
    ]

    # Decoder
    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4),  # (bs, 16, 16, 1024)
        upsample(256, 4),  # (bs, 32, 32, 512)
        upsample(128, 4),  # (bs, 64, 64, 256)
        upsample(64, 4),  # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(
        1, 4, strides=2,
        padding='same', kernel_initializer=initializer,
        activation='tanh')  # (bs, 256, 256, 1)

    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

# Build the discriminator (PatchGAN)
def build_discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 1], name='input_image')
    tar = layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 1], name='target_image')

    x = layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    batchnorm1 = layers.BatchNormalization()(conv)
    leaky_relu = layers.LeakyReLU()(batchnorm1)

    zero_pad2 = layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)
    last = layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

# Helper functions for downsampling and upsampling
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)
    )
    if apply_batchnorm:
        result.add(layers.BatchNormalization())
    result.add(layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)
    )
    result.add(layers.BatchNormalization())
    if apply_dropout:
        result.add(layers.Dropout(0.5))
    result.add(layers.ReLU())
    return result

# Load pre-trained VGG19 model for style loss calculation
def vgg19_model():
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=[IMG_HEIGHT, IMG_WIDTH, 3])
    vgg.trainable = False
    return vgg

# Gram matrix for style loss
def gram_matrix(tensor):
    channels = tensor.shape[-1]
    a = tf.reshape(tensor, [-1, channels])
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(tf.shape(a)[0], tf.float32)

# Define the style and content loss
def content_loss(target, generated):
    return tf.reduce_mean(tf.abs(target - generated))

def style_loss(target, generated):
    target_gram = gram_matrix(target)
    generated_gram = gram_matrix(generated)
    return tf.reduce_mean(tf.abs(target_gram - generated_gram))

# Total loss function combining content, style, and GAN loss
def generator_loss(disc_generated_output, gen_output, target, vgg_model):
    # GAN loss
    gan_loss = cross_entropy(tf.ones_like(disc_generated_output), disc_generated_output)

    # L1 loss (to preserve text structure)
    l1 = l1_loss(target, gen_output)

    # Convert grayscale images to RGB for VGG model
    target_rgb = tf.image.grayscale_to_rgb(target)
    gen_output_rgb = tf.image.grayscale_to_rgb(gen_output)

    # Extract VGG features for content and style losses
    target_features = vgg_model(target_rgb)
    generated_features = vgg_model(gen_output_rgb)

    # Content loss
    c_loss = content_loss(target_features, generated_features)

    # Style loss (using tf.map_fn)
    def compute_style_loss(t_g):
         t, g = t_g
         return style_loss(t, g)

    style_losses = tf.map_fn(compute_style_loss, (target_features, generated_features), dtype=tf.float32)
    s_loss = tf.reduce_mean(style_losses)

    # Total generator loss
    total_loss = gan_loss + (100 * l1) + (10 * c_loss) + (10 * s_loss)  # Adjust weights as needed
    return total_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.ones_like(disc_real_output), disc_real_output
    )
    generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
        tf.zeros_like(disc_generated_output), disc_generated_output
    )
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

# Compile the models
generator = build_generator()
discriminator = build_discriminator()
generator_optimizer = Adam(1e-3, beta_1=0.5)
discriminator_optimizer = Adam(1e-5, beta_1=0.5)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
l1_loss = tf.keras.losses.MeanAbsoluteError()

# Load pre-trained VGG19 for style loss calculation
vgg_model = vgg19_model()

# Display function for images
def display_generated_images(input_image, target, generated_image):
    plt.figure(figsize=(8, 8))
    images = [input_image, target, generated_image]
    titles = ['Input', 'Target', 'Generated']
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 4, i + 1)
        plt.imshow((img[0] * 0.5 + 0.5), cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.show()
    
# Train step
@tf.function
def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_loss = generator_loss(disc_generated_output, gen_output, target, vgg_model)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss, gen_output

# Training function
def train(dataset, epochs):
    for epoch in range(epochs):
        for input_image, target in dataset:
            gen_loss, disc_loss, generated_image = train_step(input_image, target)
        print(f"Epoch {epoch + 1}/{epochs}: Gen Loss: {gen_loss.numpy()}, Disc Loss: {disc_loss.numpy()}")
        if epoch % 5 == 0:
            # Save models every 5 epochs
            generator.save(os.path.join(MODEL_DIR, f"generator_{epoch + 1}.h5"))
            discriminator.save(os.path.join(MODEL_DIR, f"discriminator_{epoch + 1}.h5"))
        if epoch % 1 == 0:
            # Display generated images every 10 epochs
            display_generated_images(input_image.numpy(), target.numpy(), generated_image.numpy())

# Prepare dataset
dataset = load_data(DATASET_DIR)
dataset = dataset.map(data_augmentation).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Train the model
train(dataset, EPOCHS)