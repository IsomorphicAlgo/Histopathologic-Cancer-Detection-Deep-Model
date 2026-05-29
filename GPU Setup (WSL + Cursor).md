# Running this Notebook on the GPU (WSL2 + Cursor)

This project runs TensorFlow on your **NVIDIA RTX 2080** through **WSL2 (Ubuntu 22.04)**,
because TensorFlow dropped native-Windows GPU support after v2.10.

---

## Everyday workflow (do this each time you want to run the notebook)

1. **Open Cursor.**
2. Press `Ctrl+Shift+P` → type **`WSL: Connect to WSL`** (or **`WSL: Reopen Folder in WSL`**) and run it.
   - Bottom-left corner should now show a green/blue **`WSL: Ubuntu-22.04`** badge.
3. **Open the project folder** inside WSL:
   `/mnt/c/Users/Belgarath/Desktop/DTSA5511 Deep Learning/Week 3/Histopathologic Cancer Detection Deep Model`
4. Open the notebook (e.g. `Histopathologic Cancer Detection Deep Model - V7b.ipynb`).
5. **Select the kernel** (top-right of the notebook): choose **`Python (tf-gpu)`**.
6. Run the GPU-check cell (see below). You should see one GPU listed. Done — train away.

> If `Python (tf-gpu)` doesn't appear, click the kernel picker → **Select Another Kernel**
> → **Jupyter Kernel** → refresh. The kernel is registered at
> `~/.local/share/jupyter/kernels/tf-gpu`.

---

## GPU check cell (paste at top of the notebook)

```python
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print("TF:", tf.__version__, "| GPUs:", gpus)
assert gpus, "No GPU detected — check WSL connection / kernel."
```

Expected output:
`TF: 2.x.x | GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`

---

## Quick troubleshooting

| Symptom | Fix |
|---|---|
| No `WSL: Ubuntu-22.04` badge | Install the **WSL** extension in Cursor, then retry `WSL: Connect to WSL`. |
| `nvidia-smi` works on Windows but not in WSL | Update the **Windows** NVIDIA driver; never install a Linux GPU driver inside WSL. |
| `GPUs: []` **and** "Cannot dlopen some GPU libraries" | The CUDA libs aren't on the loader path. They must be in the kernel's `LD_LIBRARY_PATH` (see "Kernel fix" below). Re-run `~/fix-tf-kernel.sh`. |
| `GPUs: []` but no dlopen error | Wrong kernel selected — pick **`Python (tf-gpu)`**, not plain `Python 3.x`. |
| Kernel missing | In WSL: `~/tf-gpu/bin/python3 -m pip install ipykernel && ~/tf-gpu/bin/python3 -m ipykernel install --user --name tf-gpu --display-name "Python (tf-gpu)"`, then re-run the kernel fix below. |
| WSL slow / won't start | In PowerShell: `wsl --shutdown`, then reconnect. |

---

## Kernel fix (why the GPU was invisible at first)

TensorFlow 2.21 in a venv does **not** automatically put the pip-installed CUDA
libraries on the loader path, so the notebook kernel reported `GPUs: []` with a
"Cannot dlopen some GPU libraries" warning. The fix bakes `LD_LIBRARY_PATH` into the
kernel definition so Cursor's kernel always finds them.

A helper script lives at `~/fix-tf-kernel.sh` in WSL. Re-run it any time you
reinstall/upgrade TensorFlow or the kernel stops seeing the GPU:

```bash
bash ~/fix-tf-kernel.sh
```

Then in Cursor: **restart the kernel** (or reselect `Python (tf-gpu)`).

---

## One-time setup (already done — keep for reference / reinstalls)

1. **Install WSL2 + Ubuntu** (PowerShell as Administrator), then reboot:
   ```powershell
   wsl --install -d Ubuntu-22.04
   ```
2. **Create your Linux username/password** when Ubuntu first launches.
3. **Verify GPU passthrough** inside Ubuntu (uses the Windows driver — do not install a Linux driver):
   ```bash
   nvidia-smi
   ```
4. **Create the TensorFlow GPU environment** inside Ubuntu:
   ```bash
   sudo apt update && sudo apt install -y python3-pip python3-venv
   python3 -m venv ~/tf-gpu
   source ~/tf-gpu/bin/activate
   pip install --upgrade pip
   pip install "tensorflow[and-cuda]" ipykernel jupyter
   ```
5. **Confirm detection**:
   ```bash
   python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```
6. **In Cursor**: install the **WSL** extension so `WSL: Connect to WSL` is available.

---

## Notes

- Your files stay on Windows; from WSL they live under `/mnt/c/...`. No need to copy anything.
- Working off `/mnt/c` is fine but slightly slower than the Linux home dir. If data loading
  feels sluggish, consider copying the image data into `~/` inside WSL.
- To update TensorFlow later: `source ~/tf-gpu/bin/activate && pip install -U "tensorflow[and-cuda]"`.
