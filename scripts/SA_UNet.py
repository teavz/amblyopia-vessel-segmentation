#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Input,
    Lambda,
    MaxPooling2D,
    Permute,
)
from tensorflow.keras.models import Model


def spatial_attention(input_feature: tf.Tensor) -> tf.Tensor:
    """
    CBAM spatial attention module.

    Pools along channels (avg & max), concatenates, applies a 7×7 conv + sigmoid,
    and multiplies the attention map back onto the input feature map.
    """
    kernel_size = 7

    # ensure channels_last for pooling
    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        feat = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        feat = input_feature

    # 1) Channel-wise average pooling → shape (H, W, 1)
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(feat)
    assert avg_pool.shape[-1] == 1
    # 2) Channel-wise max pooling → shape (H, W, 1)
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(feat)
    assert max_pool.shape[-1] == 1
    # Concatenate along channel axis → shape (H, W, 2)
    concat = Concatenate(axis=-1)([avg_pool, max_pool])
    assert concat.shape[-1] == 2

    # conv → sigmoid attention map
    attention = Conv2D(
        filters=1,
        kernel_size=kernel_size,
        strides=1,
        padding="same",
        activation="sigmoid",
        kernel_initializer="he_normal",
        use_bias=False
    )(concat)
    
    assert attention.shape[-1] == 1

    # If using channels_first, permute mask back
    if K.image_data_format() == "channels_first":
        attention = Permute((3, 1, 2))(attention)

    # Multiply input feature‐map by attention map, broadcasting across channels
    return tf.keras.layers.multiply([input_feature, attention])


def _bernoulli(shape: tf.Tensor, mean: tf.Tensor) -> tf.Tensor:
    """
    Generate a binary mask by comparing uniform noise to a threshold.
    """
    rnd = tf.random.uniform(
        shape, 
        minval=0.0,
        maxval=1.0,
        dtype=tf.float32
    )
    mask = tf.nn.relu(tf.sign(mean - rnd))
    return mask


class DropBlock2D(tf.keras.layers.Layer):
    """
    DropBlock regularization for 2D feature maps.

    Randomly drops contiguous square blocks during training.
    """
    def __init__(
        self,
        block_size: int,
        keep_prob: float,
        scale: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.block_size = block_size
        self.keep_prob = float(keep_prob)
        self.scale = bool(scale)

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({
            "block_size": self.block_size,
            "keep_prob": self.keep_prob,
            "scale": self.scale,
        })
        return cfg

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        # input_shape: (batch, H, W, C)
        _, self.h, self.w, self.channel = input_shape
        p1 = (self.block_size - 1) // 2
        p0 = (self.block_size - 1) - p1
        self.padding = [[0, 0], [p0, p1], [p0, p1], [0, 0]]
        self._update_gamma()
        super().build(input_shape)

    def call(self, inputs, training=None):
        # Handle Keras 3: rely solely on the `training` flag.
        # If `training` is a Python bool, branch eagerly; otherwise use tf.cond.
        def _dropped_impl():
            mask = self._create_mask(tf.shape(inputs))
            out = inputs * mask
            if self.scale:
                scale_factor = (
                    tf.cast(tf.size(mask), tf.float32) / tf.reduce_sum(mask)
                )
                out = out * scale_factor
            return out

        # Nothing to drop if keep_prob == 1
        if self.keep_prob >= 1.0:
            return inputs

        # If training is explicitly boolean
        if isinstance(training, bool):
            return _dropped_impl() if training else inputs

        # Otherwise, `training` is a tensor-like; use tf.cond
        is_training = tf.cast(training if training is not None else False, tf.bool)
        return tf.cond(is_training, _dropped_impl, lambda: inputs)

    def _update_gamma(self):
        w = tf.cast(self.w, tf.float32)
        h = tf.cast(self.h, tf.float32)
        bs = tf.cast(self.block_size, tf.float32)
        self.gamma = (
            (1.0 - self.keep_prob) * w * h
            / (bs ** 2)
            / ((w - bs + 1.0) * (h - bs + 1.0))
        )

    def _create_mask(self, input_shape: tf.Tensor) -> tf.Tensor:
        """Sample mask and expand blocks via max-pooling."""
        B = input_shape[0]
        mask_shape = tf.stack([
            B,
            self.h - self.block_size + 1,
            self.w - self.block_size + 1,
            self.channel,
        ])
        mask = _bernoulli(mask_shape, self.gamma)
        mask = tf.pad(mask, self.padding)
        mask = tf.nn.max_pool(
            mask,
            ksize=[1, self.block_size, self.block_size, 1],
            strides=[1, 1, 1, 1],
            padding="SAME",
        )
        return 1.0 - mask


def SA_UNet(
    input_size=(512, 512, 3),
    block_size=7,
    keep_prob=0.9,
    start_neurons=16,
    lr=1e-3,
) -> Model:
    """
    Build a U-Net with spatial attention & DropBlock regularization.

    Args:
        input_size: Tuple of (H, W, C).
        block_size: Size of dropped blocks.
        keep_prob: Probability of keeping activations.
        start_neurons: Base number of filters.
        learning_rate: Learning rate for optimizer.

    Returns:
        A compiled Keras Model.
    """
    inputs = Input(input_size)

    # Encoder: conv → dropblock → bn → relu → pool
    conv1 = Conv2D(start_neurons, (3, 3), padding="same")(inputs)
    conv1 = DropBlock2D(block_size, keep_prob)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation("relu")(conv1)
    conv1 = Conv2D(start_neurons, (3, 3), padding="same")(conv1)
    conv1 = DropBlock2D(block_size, keep_prob)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation("relu")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(start_neurons * 2, (3, 3), padding="same")(pool1)
    conv2 = DropBlock2D(block_size, keep_prob)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation("relu")(conv2)
    conv2 = Conv2D(start_neurons * 2, (3, 3), padding="same")(conv2)
    conv2 = DropBlock2D(block_size, keep_prob)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation("relu")(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(start_neurons * 4, (3, 3), padding="same")(pool2)
    conv3 = DropBlock2D(block_size, keep_prob)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation("relu")(conv3)
    conv3 = Conv2D(start_neurons * 4, (3, 3), padding="same")(conv3)
    conv3 = DropBlock2D(block_size, keep_prob)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation("relu")(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    # Bottleneck + spatial attention
    convm = Conv2D(start_neurons * 8, (3, 3), padding="same")(pool3)
    convm = DropBlock2D(block_size, keep_prob)(convm)
    convm = BatchNormalization()(convm)
    convm = Activation("relu")(convm)

    convm = spatial_attention(convm)

    convm = Conv2D(start_neurons * 8, (3, 3), padding="same")(convm)
    convm = DropBlock2D(block_size, keep_prob)(convm)
    convm = BatchNormalization()(convm)
    convm = Activation("relu")(convm)

    # Decoder level 3
    deconv3 = Conv2DTranspose(
        start_neurons * 4, (3, 3), strides=(2, 2), padding="same"
    )(convm)
    uconv3 = Concatenate()([deconv3, conv3])
    uconv3 = Conv2D(start_neurons * 4, (3, 3), padding="same")(uconv3)
    uconv3 = DropBlock2D(block_size, keep_prob)(uconv3)
    uconv3 = BatchNormalization()(uconv3)
    uconv3 = Activation("relu")(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), padding="same")(uconv3)
    uconv3 = DropBlock2D(block_size, keep_prob)(uconv3)
    uconv3 = BatchNormalization()(uconv3)
    uconv3 = Activation("relu")(uconv3)

    # Decoder level 2
    deconv2 = Conv2DTranspose(
        start_neurons * 2, (3, 3), strides=(2, 2), padding="same"
    )(uconv3)
    uconv2 = Concatenate()([deconv2, conv2])
    uconv2 = Conv2D(start_neurons * 2, (3, 3), padding="same")(uconv2)
    uconv2 = DropBlock2D(block_size, keep_prob)(uconv2)
    uconv2 = BatchNormalization()(uconv2)
    uconv2 = Activation("relu")(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), padding="same")(uconv2)
    uconv2 = DropBlock2D(block_size, keep_prob)(uconv2)
    uconv2 = BatchNormalization()(uconv2)
    uconv2 = Activation("relu")(uconv2)

    # Decoder level 1
    deconv1 = Conv2DTranspose(
        start_neurons, (3, 3), strides=(2, 2), padding="same"
    )(uconv2)
    uconv1 = Concatenate()([deconv1, conv1])
    uconv1 = Conv2D(start_neurons, (3, 3), padding="same")(uconv1)
    uconv1 = DropBlock2D(block_size, keep_prob)(uconv1)
    uconv1 = BatchNormalization()(uconv1)
    uconv1 = Activation("relu")(uconv1)
    uconv1 = Conv2D(start_neurons, (3, 3), padding="same")(uconv1)
    uconv1 = DropBlock2D(block_size, keep_prob)(uconv1)
    uconv1 = BatchNormalization()(uconv1)
    uconv1 = Activation("relu")(uconv1)

    # Output
    output_conv = Conv2D(1, (1, 1), padding="same")(uconv1)
    output_act = Activation("sigmoid")(output_conv)

    model = Model(inputs=inputs, outputs=output_act)
    return model

