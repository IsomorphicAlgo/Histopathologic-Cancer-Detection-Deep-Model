# New Research

## Introduction — first attempt (course project)

This work started in **DTSA5511 Deep Learning** (Instructor: Dr. Ying Sun) as a class exploration of the [Histopathologic Cancer Detection](https://www.kaggle.com/competitions/histopathologic-cancer-detection/data) task: binary classification of 96×96 pathology patches (tumor vs non-tumor in the labeled region). The original goal was to learn end-to-end CNN practice on a real medical-imaging-style dataset. 

### What was done beyond “glow” and brightness

**Color and brightness**  as **exploratory analysis**: interpreted **aggregate color** vs **brightness**, and compared **positive vs negative** patches using **per-channel and combined RGB histograms** (plus visual spot-checks of patches). 

- **Data plumbing:** Built train/test path tables from image folders, merged **IDs** with **labels**, and used **parallel loading** (`ThreadPoolExecutor`) to fill numpy arrays from many `.tif` files.
- **Pragmatic training scope:** Trained on a **subset** of images (your notebook caps `N` for feasibility on a laptop).
- **Split protocol:** Held out a fraction of the subset for **validation** (~80/20) after **shuffling** with a fixed random seed.
- **Model:** A **VGG-style** stack—three **convolutional blocks** (filters **32 → 64 → 128**), **3×3** kernels, **ReLU**, **batch normalization**, **max pooling**, and **dropout** (stronger on the final **dense** block than on conv blocks).
- **Training mechanics:** **Binary cross-entropy**, **Adam**, and a **manual epoch loop** using `train_on_batch` / `test_on_batch` with a large **batch size** (500 in the saved code), rather than a high-level `model.fit` pipeline.

---

## New research ideas

Ideas  to try next (hypotheses, architectures, data tweaks, hyperparameters, etc.).

- **data augmentation**, **L2 regularization**, and **callbacks** 

---

## Results of trials

Log outcomes: configuration, dataset slice, metrics, and short conclusions so you can compare runs later.

| Trial / date | Idea / setup | Key metrics | Notes |
|--------------|--------------|-------------|-------|
| | | | |

### Trial log (free form)

- 
