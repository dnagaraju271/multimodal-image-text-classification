from __future__ import annotations

import json

import numpy as np
import tensorflow as tf

from src.config import CLASSES, MODELS, SEED
from src.data_pipeline import fit_vectorizer, label_matrix, make_dataset
from src.losses import WeightedBinaryCrossentropy
from src.models import build_model

MODEL_KINDS = ("image_only", "text_only", "multimodal")


def save_vocabulary(vectorizer):
    MODELS.mkdir(parents=True, exist_ok=True)
    # TextVectorization adds the mask and OOV tokens itself when reloaded.
    vocabulary = vectorizer.get_vocabulary()[2:]
    (MODELS / "text_vocabulary.json").write_text(
        json.dumps(vocabulary, ensure_ascii=False), encoding="utf-8"
    )


def load_vectorizer():
    path = MODELS / "text_vocabulary.json"
    if not path.exists():
        raise FileNotFoundError("Vocabulary missing. Run python -m src.train first.")
    vocabulary = json.loads(path.read_text(encoding="utf-8"))
    return tf.keras.layers.TextVectorization(
        max_tokens=None, output_mode="int", output_sequence_length=60,
        standardize="lower_and_strip_punctuation", vocabulary=vocabulary,
    )


def _positive_weights(frame):
    y = label_matrix(frame)
    positive = y.sum(axis=0)
    negative = len(y) - positive
    return np.clip(negative / np.maximum(positive, 1), 1.0, 10.0).tolist()


def _compile(model, train_frame, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=WeightedBinaryCrossentropy(_positive_weights(train_frame)),
    )


def fit_model(
    kind, train_frame, validation_frame, vectorizer, epochs=8, batch_size=8,
    learning_rate=1e-4, dropout=0.4, fine_tune=False, verbose=2,
):
    tf.keras.utils.set_random_seed(SEED)
    model = build_model(kind, len(vectorizer.get_vocabulary()), dropout)
    _compile(model, train_frame, learning_rate)
    train_ds = make_dataset(train_frame, vectorizer, kind, batch_size, shuffle=True)
    val_ds = make_dataset(validation_frame, vectorizer, kind, batch_size, shuffle=False)
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True
    )]
    use_fine_tuning = fine_tune and kind in ("image_only", "multimodal") and epochs >= 3
    frozen_epochs = max(1, epochs - 2) if use_fine_tuning else epochs
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=frozen_epochs,
        callbacks=callbacks, verbose=verbose,
    )
    if use_fine_tuning:
        encoder = model.get_layer("image_encoder")
        encoder.trainable = True
        cut = max(1, len(encoder.layers) - 30)
        for layer in encoder.layers[:cut]:
            layer.trainable = False
        for layer in encoder.layers[cut:]:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True
        _compile(model, train_frame, learning_rate / 10.0)
        history = model.fit(
            train_ds, validation_data=val_ds, initial_epoch=frozen_epochs,
            epochs=epochs, callbacks=callbacks, verbose=verbose,
        )
    return model, history


def predict_probabilities(model, frame, vectorizer, kind, batch_size=8):
    dataset = make_dataset(frame, vectorizer, kind, batch_size, shuffle=False)
    logits = model.predict(dataset, verbose=0)
    return tf.math.sigmoid(logits).numpy()
