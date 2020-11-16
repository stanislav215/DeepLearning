
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models

def conv_block(input_tensor, num_filters):
  encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
  encoder = layers.BatchNormalization()(encoder)
  encoder = layers.Activation('relu')(encoder)
  encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
  encoder = layers.BatchNormalization()(encoder)
  encoder = layers.Activation('relu')(encoder)
  return encoder

def encoder_block(input_tensor, num_filters):
  encoder = conv_block(input_tensor, num_filters)
  encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
  
  return encoder_pool, encoder

def decoder_block(input_tensor, concat_tensor, num_filters):
  decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
  decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
  decoder = layers.BatchNormalization()(decoder)
  decoder = layers.Activation('relu')(decoder)
  decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
  decoder = layers.BatchNormalization()(decoder)
  decoder = layers.Activation('relu')(decoder)
  decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
  decoder = layers.BatchNormalization()(decoder)
  decoder = layers.Activation('relu')(decoder)
  return decoder

def model(img_shape):
    inputs = layers.Input(shape=img_shape)
    # 256

    encoder0_pool, encoder0 = encoder_block(inputs, 32)
    # 128

    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64)
    # 64

    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128)
    # 32

    encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256)
    # 16

    encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512)
    # 8

    center = conv_block(encoder4_pool, 1024)
    # center

    decoder4 = decoder_block(center, encoder4, 512)
    # 16

    decoder3 = decoder_block(decoder4, encoder3, 256)
    # 32

    decoder2 = decoder_block(decoder3, encoder2, 128)
    # 64

    decoder1 = decoder_block(decoder2, encoder1, 64)
    # 128

    decoder0 = decoder_block(decoder1, encoder0, 32)
    # 256

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0)
    return models.Model(inputs=[inputs], outputs=[outputs])

def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss
def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score
def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss