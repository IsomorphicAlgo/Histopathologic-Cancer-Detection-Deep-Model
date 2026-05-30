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

What V6 actually did
V6	V5	V4
Best val accuracy (checkpoint)
79.5% (epoch 3) 82.2% (epoch 3) 82.5% (epoch 6)
Eval val accuracy (restored weights)
79.5% 82.2% 81.5%
Val ROC AUC
0.886 0.901 —
Train–val gap at best checkpoint
~3% ~0% ~-1%
Positive recall
60% 74% 78%


### AI Assessment

In terms of augmentation, early peak, then collapse. the model kept fitting the training stream while validation got worse. 
Where to go next
1. Fix the training schedule (do this before more architecture changes)
CosineDecayRestarts uses first_decay_steps = steps_per_epoch, so the LR restarts every epoch. The big val drop around epoch 9 lines up with that pattern plus overfitting.

V7 experiment: same model, but one of:

Constant LR 1e-4 for 10–15 epochs, or
ReduceLROnPlateau on val_loss (factor 0.5, patience 2–3), or
Cosine decay over the full run (no restarts): e.g. CosineDecay(steps_per_epoch * 20)
Also align callbacks: save and restore on the same metric (val_loss or val_auc), not checkpoint on val_accuracy and early-stop on val_loss.

This is the highest-ROI change because V6’s best weights were already at epoch 3–4.

2. Address class imbalance / threshold
With 60% positive recall, try on the same V6 architecture:

class_weight in model.fit (weights inversely proportional to class frequency), or
Keep training as-is but pick threshold from the validation ROC (often not 0.5 for imbalanced or asymmetric-cost problems)
Quick check after training: plot precision–recall and try thresholds that trade a few points of accuracy for much better recall.

3. Train longer only after the LR fix
Your research notes say augmented training usually wants more epochs, not fewer. V6 peaked at epoch 3 and then unraveled — so “more epochs” only helps after you stop the post-epoch-4 collapse.

Reasonable V7b: 20–25 epochs, early stopping patience 5 (not 10), monitor val_loss.

4. Tune augmentation strength (not the pipeline)
Keep augmentation in the model (that part is working as designed). Next ablation:

Run	Change
V7a
LR fix only (baseline)
V7b
Stronger Keras aug closer to V5: e.g. RandomRotation(0.17), add RandomTranslation(0.1), keep flips
V7c
Mild aug (current V6) + l2_reg=1e-3 (V4’s stronger regularization)
Avoid stacking generator aug on top of layer aug.

### Plan for V7:

We will begin V7 by following the suggestion of 3 subpaths, all coallescing on much needed upgrades for overall performance. 
To begin subpath 7a we will alter the recall.  Because of the imbalance we will not log the val_auc until compile. 
We will also stick to a solid LR 1e-4


V7b:
Cropping2D(cropping=((32, 32), (32, 32)))
Using padding='same' to avoid data shrinkage
right at the input slice of a secondary branch of your model


AI Assessment:

validation metrics prove this crop optimization is working well:

Validation AUC: 0.8933

Validation Accuracy: 75.20%

An AUC near 0.90 indicates the model has strong discriminative power. However, notice the gap during early training epochs:

Epoch 1: Training Accuracy was 75.61%, but Validation Accuracy dropped to 58.02%.

Epoch 2: Training Accuracy increased to 78.89%, while Validation Accuracy remained at 53.71%.

This severe divergence in accuracy alongside fluctuating validation loss (0.8777 to 1.0249) indicates your model experienced unstable validation behavior early on. This is common when using a tightly cropped focus because the model must learn subtle sub-cellular features (like nuclei boundaries) without any wider structural tissue context.


Trial 2 on V7b:
Also combinging prior plan for 7c.

New V7C:
Instead of entirely deleting the outside $64\text{ px}$ of tissue context, top-performing approaches often feed a center-cropped image through one sequence of convolutional layers (to focus on nuclear details), while feeding the full $96 \times 96$ image through a parallel sequence of convolutional layers (to monitor surrounding tissue density).

