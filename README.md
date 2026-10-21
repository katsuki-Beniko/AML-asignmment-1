# Bird Species Classifier - Streamlit app (COS30082 Assignment 1)

The hosted version of the web app, built with **Streamlit**. Upload a bird photo (or pick an example),
**choose one of 6 trained models**, and get the **top-5 predicted species** (out of 200 CUB-200 species) with
confidences plus a **Grad-CAM heat-map** showing which part of the image drove the prediction.

(A Gradio version that runs on localhost is in the separate `web_app/` folder.)

## Live app

**Link:** `[paste the Streamlit link here after deploying]`

The first visit after a period of inactivity can take a minute while the server wakes up and loads PyTorch.

## How to use it

1. Upload a photo, or click **Use** under one of the example images.
2. Pick a model in the sidebar (default: the final ConvNeXt-Tiny). Changing it re-runs the same photo.
3. Read the **top-5 predictions**, the **Grad-CAM** image (red = most influential area) and, in the sidebar,
   the model description and its reported test accuracy.

## The 6 models

All pretrained models start from general **ImageNet-1k** weights only. Accuracy is on the provided 1,204 test
images (Top-1 / average per class, as in the report's Table 1).

| Model | How it was trained | Top-1 | Avg/class |
|---|---|---|---|
| ConvNeXt-Tiny - final (default) | fine-tuned, augmentation, CutMix, 288 px, train+validation images; tested at 320 px with flip averaging | 89.4% | 89.2% |
| ConvNeXt-Tiny - fine-tuned | fine-tuned with augmentation (224 px) | 85.3% | 85.3% |
| ConvNeXt-Tiny - from scratch | random weights (under-trained) | 16.4% | 15.8% |
| ResNet50 - final | same recipe as the final ConvNeXt (288 px) | 82.2% | 81.8% |
| ResNet50 - fine-tuned | fine-tuned with augmentation (224 px) | 76.9% | 76.6% |
| ResNet50 - from scratch | random weights | 39.5% | 38.6% |

Weights are stored in half precision (float16) and loaded as float32. Only the selected model is kept in memory.

## Run it on your own computer

Needs Python 3.10 or newer.

```
pip install -r requirements.txt
streamlit run app.py
```

It opens at http://localhost:8501. (`requirements.txt` installs the CPU-only PyTorch, which is a smaller download.)

## Deploy to Streamlit Community Cloud (free)

1. Put this folder's contents in a **public GitHub repository** (the files must be at the repository root,
   including `models/`, `examples/`, `models.json`, `class_names.json`, `.streamlit/`).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. **Create app** -> choose the repository and branch -> main file path `app.py`.
4. Under **Advanced settings**, choose Python **3.12** (or 3.13).
5. Click **Deploy**. The first build installs PyTorch and takes several minutes. Then copy the app link
   into this README and into the report/slides.

## Files

| File | Purpose |
|---|---|
| `app.py` | the Streamlit interface |
| `birdmodel.py` | model loading, prediction and Grad-CAM |
| `models/` | the 6 trained weight files (float16, 48-56 MB each) |
| `models.json` | model list: names, architectures, input sizes, reported accuracies |
| `class_names.json` | the 200 species names |
| `examples/` | six images from the provided test set |
| `requirements.txt`, `.streamlit/config.toml` | packages and app settings |

Data: Caltech-UCSD Birds 200 (Wah et al., 2011). Example images come from the provided test set.
