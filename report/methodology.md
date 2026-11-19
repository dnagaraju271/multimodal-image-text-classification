# Methodology and final report checklist

## Research objective
Test whether fusion of image and caption representations improves multi-label object classification over single-modality baselines.

## Dataset and labels
Use MS COCO 2017 images, instance annotations and captions. The initial candidate set is bicycle (category 2), bird (16), bus (6), cat (17), dog (18) and horse (19). Confirm support in dataset/processed/coverage_report.json before finalizing the class set. Labels are presence/absence from instance annotations, not caption-derived guesses. An image can have multiple positive labels, so use independent sigmoid outputs and binary cross-entropy rather than a mutually exclusive softmax.

The data preparation process joins all captions for an image into one text sequence and creates one manifest row per image. It downloads only a bounded image subset.

## Leakage safeguards
Keep official COCO train2017 for development and official val2017 for final held-out evaluation. An image ID must occur in only one partition. Fit the text vocabulary on the current training partition only. K-fold validation and hyperparameter search are restricted to official train2017; never tune on the held-out official validation set.

## Architecture
- Image-only: ImageNet-pretrained EfficientNetB0, initially frozen, dense head and dropout; optionally fine-tune the final backbone layers at a lower learning rate.
- Text-only: vocabulary-based tokenization, trainable word embedding, Bidirectional LSTM, dense head and dropout.
- Fusion: concatenate image and text feature vectors, then dense layers and dropout.

## Training
Use Adam and class-weighted binary cross-entropy. The script applies early stopping using validation loss. Use the same samples and class labels for all three branches. Record subset sizes, class supports, random seed, image size, token length, optimizer, learning rate, batch size, epochs, dropout, fine-tuning policy and threshold.

## Evaluation
Report macro/micro precision, recall and F1, mean average precision, exact-match accuracy and per-class support/precision/recall/F1/AP. The baseline threshold is 0.50. Explain class imbalance and why exact-match accuracy is stricter than per-label scores.

## Results table (fill with actual model_comparison.json values)
| Model | Macro F1 | Micro F1 | mAP | Exact match |
|---|---:|---:|---:|---:|
| Image-only CNN | pending | pending | pending | pending |
| Text-only BiLSTM | pending | pending | pending | pending |
| Multimodal fusion | pending | pending | pending | pending |

Do not insert invented values. Discuss whether fusion improves on both baselines and inspect per-class scores, not only aggregate metrics.

## Limitations
A bounded six-class subset is not the complete COCO distribution. Captions may omit objects that are visibly present. Image-level annotations provide no object localization. Performance depends on subset selection, imbalance, pretrained weights, hardware and threshold choice. Record all settings needed to reproduce the experiment.
