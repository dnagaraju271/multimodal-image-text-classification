# Multi-Modal Classification and Analysis of Images and Text using Deep Learning

This project compares image-only, text-only and multimodal classifiers using paired MS COCO 2017 images and captions.

## Research question
Does combining visual information and its associated caption improve multi-label object classification compared with either modality alone?

## Three model branches
1. **Image-only:** ImageNet-pretrained EfficientNetB0 and a trainable classification head, with optional fine-tuning of final backbone layers.
2. **Text-only:** trainable word embeddings, Bidirectional LSTM, dense head and dropout.
3. **Fusion:** concatenated EfficientNetB0 and BiLSTM features, dense layers and dropout.

The initial candidate labels are bicycle (COCO 2), bird (16), bus (6), cat (17), dog (18), horse (19). Labels come from COCO instance annotations, not captions. An image may have several labels, so this is **multi-label classification** with independent sigmoid outputs, not softmax. Review class coverage before drawing conclusions.

## Environment
Use the existing project Python 3.12 virtual environment:

    source .venv/bin/activate
    python -m pip install -r requirements.txt

TensorFlow 2.21 currently runs on CPU in the Mac setup. The first training run may download ImageNet EfficientNetB0 weights.

## Prepare a bounded COCO subset
    python -m src.data_pipeline --max-train-images 3000 --max-val-images 750 --workers 8

The script downloads the COCO 2017 annotation archive, selects images having at least one candidate class and captions, then downloads only the selected images (not the full COCO image archives). It creates a one-row-per-image manifest at dataset/processed/manifest.csv and class counts at dataset/processed/coverage_report.json. Inspect the report: increase subset sizes or revise the candidate classes if any class has inadequate support.

## Train all three models
    python -m src.train --epochs 8 --batch-size 8 --fine-tune

The same official train/validation partitions are used for each branch. The vocabulary is fitted on training captions only. Artifacts are saved under models/ and results/. Reduce batch size or epochs for a quick smoke test.

## Evaluation, cross-validation and randomized search
    python -m src.train --evaluate-only
    python -m src.experiments --mode cv --folds 5 --epochs 3 --models multimodal
    python -m src.experiments --mode search --trials 4 --folds 3 --epochs 2

K-fold evaluation/search use only image IDs from official train2017. The official val2017 partition remains untouched for final held-out evaluation. These experiments retrain networks and can take significant time; start with fewer folds/trials, then run the final settings and document them.

## Local research demo website
    uvicorn src.api.main:app --reload

Open http://127.0.0.1:8000. Upload an image and enter its caption to see all available branch predictions. The website is a lightweight demonstration, not the core research deliverable. Train the models first; the API will not fabricate predictions when artifacts are missing. API docs are at /docs.

## Tests
    python -m pytest

## Metrics
Macro/micro precision, recall and F1, mean average precision (mAP), exact-match accuracy and per-class metrics are produced at a fixed 0.50 threshold. Report the threshold, per-class support, data subset, seeds, epochs, batch size, optimizer, dropout and fine-tuning setup.

## Leakage controls
- One manifest row per image; all its captions are joined before any split.
- Official train2017 and val2017 are disjoint.
- Cross-validation splits at image level and only within official train2017.
- Vocabulary is adapted on each training partition only.
- All models use the same examples and target labels.
- Dataset, trained weights and generated results are excluded from Git.
