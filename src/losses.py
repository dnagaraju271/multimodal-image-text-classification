import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="multimodal")
class WeightedBinaryCrossentropy(tf.keras.losses.Loss):
    def __init__(self, positive_weights, name="weighted_binary_crossentropy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.positive_weights = [float(v) for v in positive_weights]

    def call(self, y_true, logits):
        weights = tf.constant(self.positive_weights, dtype=logits.dtype)
        element_loss = tf.nn.weighted_cross_entropy_with_logits(
            labels=tf.cast(y_true, logits.dtype), logits=logits, pos_weight=weights
        )
        return tf.reduce_mean(element_loss, axis=-1)

    def get_config(self):
        config = super().get_config()
        config["positive_weights"] = self.positive_weights
        return config
