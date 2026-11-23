# Multi-Modal Classification and Analysis of Images and Text using Deep Learning

**Course:** IG.3510 Machine Learning · Academic year 2026–2027  
**Student:** Nagaraju Dharavath  
**Repository:** [multimodal-image-text-classification](https://github.com/dnagaraju271/multimodal-image-text-classification)  
**Portfolio project page:** [Project overview](https://dnagaraju271.github.io/projects/multimodal-classification/)  
**Interactive demonstration:** [Live dashboard](https://dnagaraju271.github.io/projects/multimodal-classification/live/)

---

## 1. Project summary

This project investigates whether combining visual information from an image with its associated text caption can improve object-category classification compared with using either modality independently.

The implementation contains and evaluates three branches:

1. **Image-only model:** an ImageNet-pretrained EfficientNetB0 convolutional neural network (CNN).
2. **Text-only model:** a trainable word-embedding layer followed by a Bidirectional Long Short-Term Memory network (BiLSTM).
3. **Multimodal model:** image and text representations are concatenated and passed through fully connected fusion layers.

The experiments use a bounded subset of MS COCO 2017. The six target categories are bicycle, bird, bus, cat, dog and horse. Ground-truth labels are derived from COCO instance annotations; captions are used as text-model input.

The trained models have been evaluated on 750 held-out images. A FastAPI service also provides real inference for a newly uploaded image and caption, and a browser dashboard displays the three model outputs side by side.

### Academic decisions confirmed with the professor

The professor confirmed that either **Softmax or Sigmoid** can be used and that a **BiLSTM** is acceptable. This project keeps **Sigmoid** because the selected COCO labels are multi-label: one image may contain several of the six target categories at the same time. Sigmoid gives each class an independent score; Softmax would force the six scores to compete as mutually exclusive alternatives.

The text branch uses a trainable Keras embedding layer learned from the training captions, followed by a Bidirectional LSTM. The embedding vectors are learned during this project's training; pretrained GloVe or Word2Vec initialization is not used in the current implementation.

### How to interpret a live prediction

The **primary answer to “what is in this image?” is the Image-only CNN prediction**. That branch receives image pixels only; the caption is never passed to it. A caption is optional in the live dashboard. When a caption is supplied, the dashboard additionally runs:
- **Text-only BiLSTM:** reads the caption only. It predicts among the six supported labels; it cannot output a category such as “train” because train is not one of the six target classes.
- **Multimodal fusion:** receives both the image and caption, so its output can be influenced by a caption that does not accurately describe the image.

Differences between those three outputs are part of the experiment: they show what each modality contributes. The image-only result remains independent of the caption.

---

## 2. Research question and objectives

**Research question:** Does combining an image with its associated caption improve object-category classification compared with an image-only or caption-only model?

The project objectives are to:

- Prepare a reproducible image-caption subset with annotation-derived labels.
- Train an image-only CNN baseline.
- Train a text-only embedding + RNN baseline.
- Build a multimodal fusion model using concatenated image and text features.
- Compare the three branches using the same held-out validation partition and classification metrics.
- Provide an inference API and an interactive demonstration for new image-caption pairs.
- Explain the models, training procedure, evaluation methodology, results and limitations.

## 3. Dataset and label construction

### 3.1 Dataset source

The project uses [MS COCO 2017](https://cocodataset.org/#download), which provides images, object-instance annotations and human-written captions.

The current prepared subset contains:

| Partition | Images | COCO partition |
|---|---:|---|
| Training | 3,000 | train2017 |
| Validation | 750 | val2017 |
| **Total** | **3,750** | |

The six selected COCO categories are:

| Target class | COCO category ID | Images containing class in the prepared subset |
|---|---:|---:|
| Bicycle | 2 | 900 |
| Bird | 16 | 857 |
| Bus | 6 | 873 |
| Cat | 17 | 873 |
| Dog | 18 | 873 |
| Horse | 19 | 860 |

Class counts overlap: an image containing more than one selected category contributes to the count of each category present. Therefore, the sum of class counts is greater than the number of unique images.

### 3.2 How labels and captions are associated

Labels are created from COCO **instance annotations**, not inferred from the captions. For each image ID:

1. Read the instance annotations and retain the six target categories.
2. Collect the selected target labels present in that image.
3. Collect its associated caption annotations.
4. Join the captions for that image into one text sequence.
5. Store one manifest row per image, including its image ID, image path, caption, labels and official split.

The manifest is stored at `dataset/processed/manifest.csv`. Class coverage information is written to `dataset/processed/coverage_report.json`.

The data preparation code selects a bounded subset from images that have at least one target label and at least one caption. A greedy selection procedure prioritizes classes with lower current counts to improve class coverage.

### 3.3 Split and leakage controls

- The training set is selected from official COCO train2017.
- The held-out validation set is selected from official COCO val2017.
- Splits are made at image-ID level; a single image does not appear in both partitions.
- All captions for an image are joined before the split is used for training.
- The text vocabulary is fitted using training captions only.
- All three model branches use the same image IDs and target labels.

The latest local integrity check found 3,750 manifest rows, zero duplicate image IDs and zero missing image files.

### 3.4 Task formulation and limitation

The current labels are multi-hot vectors. For example, an image containing both a bicycle and a dog has a target vector with both corresponding entries set to 1.

This is a multi-label classification task, not object detection. The implementation predicts whether each target category is present in an image. It does not predict bounding boxes, object locations or the number of instances.

---

## 4. Model architectures

### 4.1 Image-only CNN

**Purpose:** establish how well the target categories can be classified using image pixels alone.

**Architecture:**

1. RGB image input resized to 224 × 224.
2. ImageNet-pretrained EfficientNetB0 with the classification head removed and global average pooling.
3. Dense projection layer with 256 units and ReLU.
4. Dropout.
5. Six-unit output layer producing class logits.

The EfficientNetB0 backbone is initially frozen. When fine-tuning is enabled, the final 30 backbone layers are selectively unfrozen; Batch Normalization layers remain frozen. The fine-tuning stage uses a lower learning rate.

### 4.2 Text-only embedding + BiLSTM

**Purpose:** establish how well the target categories can be predicted from the associated caption alone.

**Architecture:**

1. Caption input.
2. Keras TextVectorization: lowercase/punctuation standardization, vocabulary limit of 20,000 tokens and sequence length of 60.
3. Trainable Embedding layer with 128 dimensions.
4. Bidirectional LSTM with 64 units in each direction and recurrent dropout configured at 0.2.
5. Dense projection layer with 128 units and ReLU.
6. Dropout.
7. Six-unit output layer producing class logits.

The vocabulary is adapted only from training captions. The embedding matrix is learned during training; pretrained GloVe/Word2Vec initialization is not used in the current version.

### 4.3 Multimodal fusion model

**Purpose:** learn from visual and textual information together.

The image branch creates a 256-dimensional image representation. The text branch creates a 128-dimensional text representation. These vectors are concatenated:

`z = [v ; t]`

where `v` is the image representation and `t` is the text representation. The concatenated vector has 384 features.

The fusion head is:

1. Concatenate image and text features (384 dimensions).
2. Dense layer (256 units, ReLU).
3. Batch Normalization.
4. Dropout (0.4).
5. Dense layer (128 units, ReLU).
6. Dropout (0.2).
7. Six-unit output layer producing class logits.

The image and text encoders are trained jointly with the fusion head. The image encoder can be fine-tuned using the same staged procedure as the image-only model.

### 4.4 Architecture comparison

| Branch | Input | Feature extractor | Classifier |
|---|---|---|---|
| Image-only | Image | EfficientNetB0 | Dense head |
| Text-only | Caption | Embedding + BiLSTM | Dense head |
| Multimodal | Image + caption | EfficientNetB0 + BiLSTM | Concatenation + Dense fusion head |

### 4.5 Activation, loss function and optimisation

The same output/loss/optimisation strategy is used for the image-only, text-only and fusion classifiers.

| Component | Choice in this project | Explanation |
|---|---|---|
| Hidden-layer activation | **ReLU** | Used in the image, text and fusion projection layers to introduce non-linearity. |
| Output activation | **Sigmoid** (independent per class) | Converts each of the six output logits into an independent probability. Several classes may be predicted for one image. |
| Loss function | **Class-weighted Binary Cross-Entropy with logits** | Suitable for multi-label targets. The positive weight compensates for class-frequency imbalance. |
| Optimizer | **Adam** | Updates model parameters using adaptive estimates of first and second moments of the gradients. |
| Initial learning rate | **0.0001 (1 × 10⁻⁴)** | Used for the initial training stage. |
| Fine-tuning learning rate | **0.00001 (1 × 10⁻⁵)** | One tenth of the initial learning rate for the unfrozen EfficientNetB0 layers. |

The final Dense layer produces six raw logits (z_i). The sigmoid operation is:

`p_i = 1 / (1 + exp(-z_i))`

The implementation passes logits directly to TensorFlow's numerically stable weighted cross-entropy operation during training. At inference, sigmoid converts those logits to probabilities. The default decision threshold is 0.50.

For a target (y\in\{0,1\}), logit (z), and positive-class weight (w^+_c), the class-weighted binary loss is:

`L(z,y) = -[w⁺ᶜ · y · log(σ(z)) + (1-y) · log(1-σ(z))]`

The positive weight for each class is calculated from the training labels as negative-count / positive-count and clipped to [1, 10]. The loss is averaged across classes and examples.

**Why Sigmoid rather than Softmax?** An image can contain more than one selected COCO category, so the labels are not mutually exclusive. Sigmoid + weighted Binary Cross-Entropy represents this multi-label structure. Softmax would be the better choice only after converting the task to mutually exclusive single-label classification, which is not the current dataset formulation.

---

## 5. Training methodology

The training code uses TensorFlow/Keras.

| Setting | Current implementation |
|---|---|
| Image size | 224 × 224 RGB |
| Text vocabulary limit | 20,000 tokens |
| Text sequence length | 60 tokens |
| Embedding dimension | 128 |
| BiLSTM | 64 units in each direction |
| Image projection | 256 units |
| Text projection | 128 units |
| Fusion dense layers | 256 and 128 units |
| Optimizer | Adam |
| Default initial learning rate | 0.0001 (1 × 10⁻⁴) |
| Fine-tuning learning rate | 0.00001 (1 × 10⁻⁵), one tenth of the initial learning rate |
| Batch size in recorded training run | 8 |
| Requested maximum epochs in recorded run | 8 |
| Main dropout | 0.4 |
| Inference threshold | 0.50 |
| Loss | Class-weighted Binary Cross-Entropy |

Early stopping monitors validation loss and restores the best weights. With fine-tuning enabled, the image-based branches use a frozen-backbone training stage followed by a lower-learning-rate fine-tuning stage.

Actual training may stop before the requested maximum number of epochs because of early stopping. The report should use the completed-epoch counts in the training logs, not simply the requested maximum.

---

## 6. Evaluation methodology

The three saved models were evaluated on the same 750-image official validation partition at a 0.50 threshold.

The project records:

- Per-class precision, recall and F1.
- Macro and micro precision, recall and F1.
- Mean Average Precision (mAP).
- Exact-match accuracy.
- Per-class support and Average Precision.

**Metric interpretation:**

- **Precision:** fraction of predicted positive labels that are correct.
- **Recall:** fraction of actual positive labels that are recovered.
- **F1:** harmonic mean of precision and recall.
- **Macro F1:** arithmetic mean of the six class-level F1 scores; each class has equal weight.
- **Micro F1:** aggregates true positives, false positives and false negatives over all class decisions.
- **mAP:** mean of the per-class Average Precision values, measuring ranking quality across thresholds.
- **Exact match:** fraction of images for which all six predicted labels exactly match the ground truth.

### 6.1 Saved held-out validation results

The following values are from the saved `results/model_comparison.json` report:

| Model | Macro F1 | Micro F1 | mAP | Exact-match accuracy |
|---|---:|---:|---:|---:|
| Image-only CNN | 0.7043 | 0.6972 | 0.8536 | 0.4000 |
| Text-only BiLSTM | 0.7551 | 0.7471 | 0.8402 | 0.5067 |
| **Multimodal fusion** | **0.7593** | **0.7498** | **0.9036** | **0.5067** |

### 6.2 Results discussion

The multimodal fusion model has the highest macro F1, micro F1 and mAP among the three saved models. Its macro F1 improvement over the text-only model is modest (0.7593 versus 0.7551). Exact-match accuracy is the same for the text-only and fusion models (0.5067).

The results support a measured conclusion: fusion improves some metrics, particularly mAP, on this selected subset, but it does not improve every metric. The findings should not be generalized to all COCO images without further evaluation.

The saved results are distinct from individual live-demo predictions. A single uploaded-image probability is not a replacement for held-out dataset evaluation.

The live dashboard's primary visual prediction is the highest-scoring class from the image-only CNN. Its score details remain available for all six classes because the underlying task is multi-label. A supplied caption does not alter this image-only output; it only enables the separate BiLSTM and fusion comparisons.

---

## 7. Live inference demonstration

### 7.1 Web interface

- [Project overview](https://dnagaraju271.github.io/projects/multimodal-classification/)
- [Interactive live dashboard](https://dnagaraju271.github.io/projects/multimodal-classification/live/)

The static frontend is hosted on GitHub Pages. The TensorFlow inference service runs separately using FastAPI. The dashboard highlights the top-ranked Image-only CNN class as the primary visual prediction and places the caption-only and fused outputs in a separate comparison area.

### 7.2 API endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Verify that the FastAPI service is responding |
| `GET /api/models/status` | Check availability of all three models and the text vocabulary |
| `GET /api/metrics` | Return the saved held-out evaluation report |
| `POST /api/predict` | Accept an image and caption and return predictions from all three models |

The prediction endpoint accepts an image file and an optional caption as multipart form fields. The current maximum upload size is 12 MB. The service always runs the image-only CNN using the image pixels. If a non-empty caption is supplied, it also vectorizes the caption and runs the text-only BiLSTM and multimodal fusion models. Sigmoid is applied to each model's logits; per-class scores are returned, and labels above threshold 0.50 are marked as predicted.

### 7.3 Live demo architecture

`Browser (GitHub Pages) → HTTPS tunnel → FastAPI on local Mac → TensorFlow/Keras models → JSON predictions → dashboard`

A Cloudflare Quick Tunnel is used for temporary HTTPS access to the local API during demonstrations. It is not a permanent production deployment: its URL changes after restart, and the local Mac, FastAPI process and tunnel must remain running. The tunnel should be stopped after the demonstration. Do not submit sensitive images through this public temporary endpoint.

GitHub Pages cannot execute the TensorFlow models itself; it only hosts the frontend.

### 7.4 Local startup

Create and activate the Python 3.12 virtual environment, then install dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Start the inference API:

```bash
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

For a public HTTPS route during a demonstration, open a second Terminal and run:

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

Copy the newly generated `https://…trycloudflare.com` URL into the dashboard's API URL field and select **Connect / Check API**. Verify that the service reports all three models and the vocabulary as ready.

The trained model artifacts and vocabulary are not committed to this Git repository. They are stored separately in a private Hugging Face model repository and are also available in the local `models/` directory on the development Mac. For a clean reproduction or another machine, obtain authorized access to those artifacts or run the training workflow below.

---

## 8. Reproducing the data, training and evaluation workflow

### 8.1 Prepare the dataset

```bash
python -m src.data_pipeline --max-train-images 3000 --max-val-images 750 --workers 8
```

This prepares the manifest, downloads the selected image files and writes the coverage report.

### 8.2 Train the three branches

```bash
python -m src.train --epochs 8 --batch-size 8 --fine-tune
```

The script fits the text vocabulary on training captions, trains the image-only, text-only and multimodal branches, saves the model files under `models/`, and evaluates the saved models on official val2017.

### 8.3 Re-evaluate saved models

```bash
python -m src.train --evaluate-only
```

### 8.4 Automated tests

```bash
python -m pytest -q
```

The latest recorded local test run completed with **5 passed**.

### 8.5 Cross-validation and hyperparameter search

The repository contains experiment implementations in `src/experiments.py`:

```bash
# Example five-fold cross-validation on the training partition
python -m src.experiments --mode cv --folds 5 --epochs 3 --models multimodal

# Example bounded randomized search
python -m src.experiments --mode search --trials 4 --folds 3 --epochs 2
```

The cross-validation and search code uses image-level folds within official train2017, leaving official val2017 out of those experiments. These commands retrain models and may take time.

**Completion status:** the code for cross-validation and randomized search is present, but `results/cross_validation.json` and `results/random_search.json` were not present during the latest project audit. These experiments must be run and their results reviewed before they are described as completed.

---

## 9. Software and repository structure

### Main dependencies

- Python 3.12
- TensorFlow / Keras
- NumPy and Pandas
- Scikit-learn
- Pillow
- FastAPI and Uvicorn
- python-multipart
- pytest

See `requirements.txt` for the pinned and bounded dependency versions.

### Main source files

| File | Responsibility |
|---|---|
| `src/config.py` | Paths, target classes, input sizes, vocabulary and thresholds |
| `src/data_pipeline.py` | COCO annotations, image/caption pairing, subset selection, manifest and TensorFlow datasets |
| `src/models.py` | EfficientNetB0, BiLSTM and multimodal model definitions |
| `src/losses.py` | Weighted Binary Cross-Entropy |
| `src/training.py` | Vocabulary, model compilation, training, fine-tuning and probability generation |
| `src/metrics.py` | Multi-label evaluation metrics |
| `src/train.py` | Training and held-out evaluation orchestration |
| `src/experiments.py` | Cross-validation and randomized hyperparameter search |
| `src/api/main.py` | FastAPI endpoints and real-time inference |
| `tests/` | Automated tests |

Large dataset files, trained model weights and generated experiment outputs are excluded from Git to keep the source repository manageable. The dataset can be regenerated from COCO using the preparation command.

---

## 10. Limitations and open questions

1. **Multi-label output:** the professor confirmed that either Softmax or Sigmoid is acceptable. Sigmoid was selected because COCO images can contain multiple selected classes. A Softmax version would require mutually exclusive single-label targets, model/loss changes and retraining.
2. **Embedding choice:** the current text embedding is trainable from this project's training captions; pretrained GloVe or Word2Vec vectors are not currently used. Confirm whether this satisfies the expected embedding method.
3. **Bounded dataset:** 3,750 images and six selected classes are a subset of COCO 2017, not the full dataset distribution.
4. **Caption ambiguity:** captions may omit objects that are visibly present in an image.
5. **No localization:** this is image-level multi-label classification; it does not output bounding boxes or object counts.
6. **Experiment evidence:** cross-validation and randomized-search implementations exist, but their result files were missing at the latest audit.
7. **Temporary demo hosting:** the live inference API depends on a locally running Mac and temporary Quick Tunnel; this is suitable for a controlled demonstration, not guaranteed production uptime.
8. **Generalization:** the saved held-out results measure this selected subset and split. Broader claims require additional evaluation.

---

## 11. Current project status

| Item | Status at latest audit |
|---|---|
| COCO subset and manifest (3,750 images) | Verified |
| Three trained Keras models and vocabulary | Available locally |
| Held-out comparison on 750 images | Saved and verified |
| FastAPI service and live three-model inference | Demonstrated |
| GitHub Pages dashboard | Published |
| Automated tests | 5 passed |
| Cross-validation results | Not yet generated |
| Hyperparameter-search results | Not yet generated |
| Output activation and loss choice | Sigmoid + weighted Binary Cross-Entropy selected with professor's permission |
| BiLSTM architecture | Accepted by professor; trainable Keras embeddings used |
| Mathematical technical report | To be completed |
| Final presentation | To be completed |

This repository documents the implementation and results available at the latest audit. Items marked pending are not presented as completed.

---

## 12. References

- Course assignment: *Deep Learning Project — Multi-Modal Classification and Analysis of Images and Text using Deep Learning* (provided course document).
- Course evaluation: *IG.3510 Machine Learning 2026–2027 — Evaluation criteria*.
- [MS COCO 2017 dataset](https://cocodataset.org/#download).
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946).
- [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf).
- TensorFlow/Keras documentation: [https://www.tensorflow.org/](https://www.tensorflow.org/).
- FastAPI documentation: [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/).

---

**Repository contents:** source code, data preparation workflow, training/evaluation implementation, tests and this documentation. Dataset files, model weights and generated experiment outputs are stored separately and are not included in the public Git repository.
