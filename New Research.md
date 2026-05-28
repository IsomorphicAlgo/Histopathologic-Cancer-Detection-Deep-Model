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

### New Neural Structure

#### Swapping to more well-researched structures (DenseNet)

- **data augmentation**, **L2 regularization**, and **callbacks**

### Lowdown — how to implement

#### L2 regularization (weight decay)

**The idea.** Add a penalty proportional to the squared magnitude of the weights to the loss. The optimizer is then pushed to prefer smaller weights, which tends to produce smoother decision boundaries and reduces overfitting.

Per-layer constructor argument: `kernel_regularizer=l2(lambda)`. Modify every layer that has learnable weights (every `Conv2D`, every `Dense`, optionally `BatchNormalization`'s `gamma/beta`).

```python
from tensorflow.keras.regularizers import l2

L2 = 1e-3  # the only *tunable* knob

x = Conv2D(64, (3, 3),
           kernel_regularizer=l2(L2),
           kernel_initializer='he_normal')(x)
# ...
x = Dense(256, kernel_regularizer=l2(L2))(x)
```

- **`lambda` (the strength).** Start at **`1e-4`** for a small model, **`1e-3`** for a deeper one. Bigger λ → stronger pull toward zero → more underfitting risk. Tune in factors of 10.
- **Apply it everywhere or nowhere.** 
- **It only affects training.** 
- **`model.compile(..., loss=...)` reports the combined loss** 
- **Don't double-regularize.** heavy Dropout + BatchNorm, an aggressive L2 on top can stall learning. 
- **Bias terms.** Use `bias_regularizer=l2(...)` only for specific reason; "regularize kernels, leave biases alone."

The train/val accuracy gap should shrink (less overfitting). If train accuracy also drops noticeably, λ is too large.

---

#### Data augmentation

**The idea.** Apply random label-preserving transforms (flips, rotations, shifts, zooms, brightness/contrast jitter, etc.) to each training image on the fly. The model sees a slightly different version of every patch every epoch, so it has to learn the *content* instead of memorizing pixel positions.

**Why it's basically free for this task.** Histopathology patches have no preferred orientation — a tumor rotated 90° is still a tumor — so flips/rotations are guaranteed not to change the label. That makes augmentation here much safer than, say, on natural images of text or faces.

**Where it lives in Keras.** Two reasonable ways:

1. **`ImageDataGenerator` + `flow_from_dataframe`** (what V4 uses, classic API):

   ```python
   from tensorflow.keras.preprocessing.image import ImageDataGenerator

   train_gen = ImageDataGenerator(
       rescale=1./255,          # normalize pixel range
       rotation_range=20,       # degrees
       width_shift_range=0.2,   # fraction of width
       height_shift_range=0.2,
       shear_range=0.2,
       zoom_range=0.2,
       horizontal_flip=True,
       vertical_flip=True,      # safe for pathology
       fill_mode='nearest',
   )
   val_gen = ImageDataGenerator(rescale=1./255)  # no augmentation on val/test
   ```

2. **`tf.keras.layers.*` preprocessing layers** (newer, runs on GPU, baked into the model):

   ```python
   aug = tf.keras.Sequential([
       tf.keras.layers.RandomFlip('horizontal_and_vertical'),
       tf.keras.layers.RandomRotation(0.1),
       tf.keras.layers.RandomZoom(0.2),
   ])
   inputs = Input(shape=(96, 96, 3))
   x = aug(inputs)             # only active during training
   x = tf.keras.layers.Rescaling(1./255)(x)
   # ...rest of the model
   ```

   This second style is nicer because the model file *contains* the preprocessing — no risk of forgetting to normalize at inference.

**What to watch.**
- **Augment training only, never validation or test.** The val/test pipeline should do *exactly* the same deterministic preprocessing the model expects (here: `BGR→RGB` + `/255`) and nothing else. 
- **Train/inference preprocessing parity.** This is the bug that bit V3. Whatever the training generator does (e.g. `rescale=1./255`), the submission code must do the same. If you go with preprocessing layers (`Rescaling`, `RandomFlip`, etc.), this is automatic.
Expect training accuracy to plateau lower and val accuracy to track training more closely. That gap-closing is the goal. With augmentation on you generally want **more epochs**, not fewer, because the effective dataset is larger.
- **Inspect what you're feeding the model.** Always pull one batch out of the generator and `plt.imshow` a few — augmentation bugs (wrong color channel, all-black images, label/image desync) are obvious visually and silent in metrics.

---

### Suggested first experiment for the new notebook

A clean baseline → ablate plan you can drop into the trial log below:

1. **Baseline:** the V3 model with no augmentation, no L2. Train ~5 epochs. Note train acc, val acc, and the gap.
2. **+ L2 only:** add `kernel_regularizer=l2(1e-3)` to every conv + dense. Same epochs. Compare the gap.
3. **+ augmentation only:** baseline model, but feed it through `ImageDataGenerator` with the recipe above. Same epochs.
4. **+ both** (what V4 does). Same epochs.

That gives you four rows in the trial-log table and an honest answer to "did each piece actually help?"

---

## Results of trials

Log outcomes: configuration, dataset slice, metrics, and short conclusions so you can compare runs later.

| Trial / date | Idea / setup | Key metrics | Notes |
|--------------|--------------|-------------|-------|
| | | | |

### Trial log (free form)

- 
