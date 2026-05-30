# Histopathologic Cancer Detection Deep Learning Model

## Project Overview
This project applies deep learning techniques to detect cancer in histopathologic images of breast cancer tissue. Using convolutional neural networks (CNNs), the model analyzes microscopic images to identify the presence of cancer cells, achieving high accuracy in classification.

## Author
- **Michael Hansen**
- **Course**: DTSA5511 Deep Learning
- **Instructor**: Dr. Ying Sun

## Dataset
The dataset comes from the [Histopathologic Cancer Detection](https://www.kaggle.com/competitions/histopathologic-cancer-detection/data) Kaggle competition, originally obtained from the [TCGA](https://portal.gdc.cancer.gov/projects/TCGA-BRCA) project. It consists of:
- Approximately 220,000 labeled images for training
- About 57,000 images for testing
- Each image is labeled as either containing cancer tissue (1) or not (0)
- A positive label indicates that the center 32x32px region of a patch contains at least one pixel of tumor tissue

## Repository layout

```
Histopathologic-Cancer-Detection-Deep-Model/
├── README.md
├── requirements.txt
├── New Research.md
├── Histopathologic Cancer Detection Deep Model.ipynb          # Original (V1) notebook
├── Histopathologic Cancer Detection Deep Model - V3.ipynb
├── Histopathologic Cancer Detection Deep Model - V4.ipynb
├── Histopathologic Cancer Detection Deep Model - V5.ipynb
├── Histopathologic Cancer Detection Deep Model - V6.ipynb
├── Histopathologic Cancer Detection Deep Model - V7a.ipynb
├── Histopathologic Cancer Detection Deep Model - V7b.ipynb
├── Histopathologic Cancer Detection Deep Model - V7c.ipynb     # Best model: two-stream CNN
├── training_history.csv          # Per-epoch metrics from the latest notebook run (CSVLogger)
├── training_history.npy          # Same history saved as a NumPy array (if generated)
├── model_checkpoints/
│   └── best_model_v4.keras       # Best checkpoint from V4 training (by val_accuracy)
├── logs/
│   └── validation/               # TensorBoard event files
└── Data/
    ├── train_labels.csv          # Competition labels (id, label) for training images
    ├── train_labels_old.csv      # Earlier copy of the label file
    ├── sample_submission.csv     # Kaggle submission format template
    ├── submission.csv            # Primary Kaggle submission output
    ├── submission0.csv           # Earlier submission attempts
    ├── submission1.csv
    ├── model_training_performance.csv  # Versioned training log (see below)
    ├── Results scratchbook.docx  # Informal results notes
    ├── train/                    # Training .tif images (local only; gitignored)
    └── test/                     # Test .tif images (local only; gitignored)
```

### File purposes

| Path | Purpose |
|------|---------|
| `Histopathologic Cancer Detection Deep Model*.ipynb` | Iterative notebook versions (V1 original → V3 → V4 → V5). Each documents architecture, training, evaluation, and submission for that experiment. |
| `Data/train_labels.csv` | Maps image IDs to binary labels (0 = no tumor in center patch, 1 = tumor present). Used to build training/validation splits. |
| `Data/train/`, `Data/test/` | Kaggle competition image folders. Not committed to git because of size; download from the competition page and place here. |
| `Data/sample_submission.csv` | Empty submission template from Kaggle (`id`, `label` columns). |
| `Data/submission*.csv` | Model predictions on the test set, formatted for Kaggle upload. |
| `Data/model_training_performance.csv` | **Cross-version training registry.** One block per model version: `epoch` rows (train/val accuracy and loss each epoch) plus a `summary` row (hyperparameters, best metrics, classification report fields, notebook reference). Append V5, V6, etc. as new versions are trained. V1–V3 are not recorded yet. |
| `training_history.csv` | Raw epoch log from the **current** notebook’s `CSVLogger` callback. Overwritten on each run; copy or summarize into `Data/model_training_performance.csv` when a version is finalized. |
| `model_checkpoints/` | Saved Keras models from `ModelCheckpoint` during training. |
| `logs/` | TensorBoard logs for loss/accuracy curves and histograms. |
| `New Research.md` | Research notes, planned experiments, and trial log for future work. |
| `requirements.txt` | Python dependencies for running the notebooks. |

## Workflow
The project follows these key steps:
1. **Data exploration and preparation** — load labels and images, visualize samples, compare RGB distributions between classes.
2. **Model development** — from a simple CNN (V1) to a regularized VGG-style architecture (V3+).
3. **Training and optimization** — augmentation, L2 regularization, learning-rate scheduling, early stopping, and checkpointing.
4. **Evaluation and submission** — validation metrics, ROC/classification reports, and Kaggle `submission.csv` generation.

## Model Progression and Results

The project was developed as a disciplined sequence of controlled experiments. Each version isolated a single change so its effect on generalization could be attributed cleanly. The table below summarizes the trajectory; AUC is the competition metric, so Kaggle scores are reported as ROC AUC.

| Version | Key change | Val acc | Val AUC | Kaggle public / private AUC |
|---------|------------|---------|---------|-----------------------------|
| V4 | VGG-style CNN + augmentation + L2 | 82.5% | — | — |
| V5 | Tuned augmentation | 82.2% | 0.901 | — |
| V6 | In-model Keras augmentation layers | 79.5% | 0.886 | — |
| V7a | Stabilized schedule (constant LR 1e-4) | 75.2% | 0.893 | 0.8980 / 0.8715 |
| V7b | Center-crop-only input (32×32) | 83.6% | 0.906 | 0.8786 / 0.8414 |
| V7c | **Two-stream CNN (local 32×32 + global 96×96)** | **89.6%** | **0.969** | **0.9353 / 0.9188** |

**Engineering narrative.** Versions V4–V6 plateaued near 0.90 validation AUC, with recurring symptoms of validation instability and early-epoch overfitting under aggressive augmentation and a restart-based learning-rate schedule. V7 split the remediation into three targeted experiments: V7a stabilized the training schedule (constant learning rate, AUC-based checkpointing and early stopping), V7b tested an input that cropped to the labeled center region, and V7c introduced a two-stream architecture.

The center-crop hypothesis (V7b) underperformed: despite higher validation accuracy than V7a, it generalized worse on the held-out Kaggle test set (private AUC 0.84 vs 0.87), indicating that discarding peripheral tissue context removed useful signal. V7c resolved this by running two parallel convolutional streams — a **local branch** over the center 32×32 crop (nuclear morphology) and a **global branch** over the full 96×96 patch (surrounding tissue context) — each reduced via global average pooling and concatenated before the dense head.

V7c is the strongest model by a clear margin. It is the first version to break past the 0.90 AUC plateau, reaching a Kaggle public AUC of **0.9353** and private AUC of **0.9188**, with a near-zero train–validation accuracy gap (≈ −0.002) indicating the gain came from improved representational capacity rather than overfitting. The small, expected gap between validation AUC (0.969) and test AUC (~0.92) confirms the result is robust. The principal remaining limitation is positive-class recall (0.71 at a 0.5 decision threshold against 0.98 precision); given the high AUC, threshold calibration from the validation ROC is the recommended next step for any deployment that requires a hard decision.

## Technical Implementation
The project utilizes:
- **TensorFlow/Keras** for model building and training
- **OpenCV** for image processing
- **Pandas/NumPy** for data manipulation
- **Matplotlib** for visualization
- **Scikit-learn** for data splitting and evaluation metrics

### Model Architecture
The architecture evolved from a single VGG-style stack (V4–V7b) to a two-stream design (V7c, best performing). Shared building blocks across all versions:
- Convolutional blocks with increasing filter sizes (32 → 64 → 128)
- Batch normalization after each convolutional layer
- Max pooling layers to reduce dimensionality
- Dropout layers to prevent overfitting
- L2 regularization to improve generalization
- In-model augmentation and rescaling so training/inference preprocessing stay in parity

The best model (V7c) runs two parallel streams over a shared augmentation/rescaling head — a local branch over the center 32×32 crop and a global branch over the full 96×96 patch — each reduced by global average pooling to a 128-d vector, concatenated (256-d) and passed to the dense classification head.

## Key Findings
- Deep learning models can effectively identify cancer in histopathologic images
- Data augmentation significantly improves model performance
- Batch normalization and regularization techniques help prevent overfitting
- Learning rate scheduling improves training stability and final performance
- The model demonstrates strong potential for assisting pathologists in cancer detection

## Conclusion
This project demonstrates the power of deep learning in medical image analysis. The developed CNN model successfully identifies cancer in histopathologic images with high accuracy, showing the clear potential for real-world applications in assisting medical professionals with cancer diagnosis.

The implementation of advanced techniques like data augmentation, batch normalization, and learning rate scheduling proved crucial in achieving robust performance. Future work could explore ensemble methods, more complex architectures, or transfer learning approaches to further improve accuracy.One could also expand on the applications within the medical field as well, as imagry is used in a wide range of diagnosis methods. 

## References
- VGG-16 CNN Model: https://www.geeksforgeeks.org/vgg-16-cnn-model/
- Kaggle Competition: https://www.kaggle.com/competitions/histopathologic-cancer-detection/data
- TCGA Project: https://portal.gdc.cancer.gov/projects/TCGA-BRCA 


## Changelog

**05-29-2026**
- Documented the full V6 → V7a/V7b/V7c experiment progression and added a Model Progression and Results section.
- Recorded Kaggle ROC AUC scores: V7a (0.8980 / 0.8715), V7b (0.8786 / 0.8414), V7c (0.9353 / 0.9188 public / private).
- V7c (two-stream local + global CNN) established as the best-performing model; updated the Model Architecture section accordingly.

**05-27-2026**
- Added `Data/model_training_performance.csv` — versioned training performance log (V4 recorded; future versions append here).
- Documented repository layout and file purposes in this README.

**05-24-2026**
- Updated README.
- Added a `Data/` folder for CSVs and docs; corrected paths in the notebooks.
