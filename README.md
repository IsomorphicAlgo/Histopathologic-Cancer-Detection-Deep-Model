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

## Technical Implementation
The project utilizes:
- **TensorFlow/Keras** for model building and training
- **OpenCV** for image processing
- **Pandas/NumPy** for data manipulation
- **Matplotlib** for visualization
- **Scikit-learn** for data splitting and evaluation metrics

### Model Architecture
The final model architecture includes:
- Multiple convolutional blocks with increasing filter sizes (32 → 64 → 128)
- Batch normalization after each convolutional layer
- Max pooling layers to reduce dimensionality
- Dropout layers to prevent overfitting
- L2 regularization to improve generalization
- Dense layers with appropriate activation functions

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

**05-27-2026**
- Added `Data/model_training_performance.csv` — versioned training performance log (V4 recorded; future versions append here).
- Documented repository layout and file purposes in this README.

**05-24-2026**
- Updated README.
- Added a `Data/` folder for CSVs and docs; corrected paths in the notebooks.
