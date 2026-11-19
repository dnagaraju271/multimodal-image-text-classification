from __future__ import annotations

import tensorflow as tf

from src.config import CLASSES, IMAGE_SIZE, SEQUENCE_LENGTH


def _image_encoder():
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3), pooling="avg",
    )
    backbone.trainable = False
    return tf.keras.Model(backbone.input, backbone.output, name="image_encoder")


def build_model(kind: str, vocab_size: int, dropout: float = 0.4):
    keras = tf.keras
    if kind in ("image_only", "multimodal"):
        image_input = keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
        encoder = _image_encoder()
        image_features = encoder(image_input, training=False)
        image_features = keras.layers.Dense(256, activation="relu", name="image_projection")(image_features)
    if kind in ("text_only", "multimodal"):
        text_input = keras.Input(shape=(SEQUENCE_LENGTH,), dtype="int64", name="text")
        embedded = keras.layers.Embedding(
            input_dim=max(vocab_size, 2), output_dim=128, mask_zero=True, name="word_embedding"
        )(text_input)
        text_features = keras.layers.Bidirectional(
            keras.layers.LSTM(64, dropout=0.2), name="bidirectional_lstm"
        )(embedded)
        text_features = keras.layers.Dense(128, activation="relu", name="text_projection")(text_features)

    if kind == "image_only":
        features = keras.layers.Dropout(dropout)(image_features)
        logits = keras.layers.Dense(len(CLASSES), name="class_logits")(features)
        return keras.Model(image_input, logits, name="image_only_classifier")
    if kind == "text_only":
        features = keras.layers.Dropout(dropout)(text_features)
        logits = keras.layers.Dense(len(CLASSES), name="class_logits")(features)
        return keras.Model(text_input, logits, name="text_only_classifier")
    if kind == "multimodal":
        fused = keras.layers.Concatenate(name="feature_fusion")([image_features, text_features])
        fused = keras.layers.Dense(256, activation="relu")(fused)
        fused = keras.layers.BatchNormalization()(fused)
        fused = keras.layers.Dropout(dropout)(fused)
        fused = keras.layers.Dense(128, activation="relu")(fused)
        fused = keras.layers.Dropout(dropout / 2)(fused)
        logits = keras.layers.Dense(len(CLASSES), name="class_logits")(fused)
        return keras.Model({"image": image_input, "text": text_input}, logits, name="multimodal_classifier")
    raise ValueError(f"Unknown model kind: {kind}")
