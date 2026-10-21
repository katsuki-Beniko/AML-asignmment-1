"""Model code for the bird classifier web app: model loading, prediction and Grad-CAM.

Kept separate from the Streamlit interface (app.py) so it can be tested on its own.
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms as T

HERE = Path(__file__).parent
NAMES = json.loads((HERE / "class_names.json").read_text())     # 200 readable species names
MODELS = json.loads((HERE / "models.json").read_text())         # the 6 models offered in the app
BY_LABEL = {m["label"]: m for m in MODELS}
DEFAULT = MODELS[0]["label"]                                    # the final ConvNeXt-Tiny model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def load_net(m):
    """Build the architecture and load the saved weights (stored as float16, loaded as float32)."""
    net = timm.create_model(m["arch"], pretrained=False, num_classes=len(NAMES))
    net.load_state_dict(torch.load(HERE / "models" / m["file"], map_location="cpu"))
    return net.to(DEVICE).eval()


def target_layer(net, arch):
    """Last convolutional stage, used by Grad-CAM."""
    return {"resnet50": lambda: net.layer4[-1], "convnext_tiny": lambda: net.stages[-1]}[arch]()


def preprocess(size):
    return T.Compose([T.Resize(int(size * 1.14)), T.CenterCrop(size), T.ToTensor(), T.Normalize(MEAN, STD)])


def gradcam(net, layer, x, cls):
    """Grad-CAM for class `cls`: weight each channel of `layer` by its mean gradient, sum, ReLU, normalise."""
    acts, grads = {}, {}

    def fwd(mod, inp, out):                                  # keep the activation, grab its gradient
        acts["a"] = out
        out.register_hook(lambda g: grads.__setitem__("g", g))
    h = layer.register_forward_hook(fwd)
    out = net(x.clone().requires_grad_(True))
    net.zero_grad()
    out[0, cls].backward()
    h.remove()
    w = grads["g"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * acts["a"]).sum(1))[0].detach().cpu().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def predict(net, img, m):
    """Return (top-5 [(species, probability)], Grad-CAM overlay image) for a PIL image."""
    img = img.convert("RGB")
    x = preprocess(m["size"])(img)[None].to(DEVICE)
    with torch.no_grad():
        p = net(x).softmax(1)
        if m["flip"]:                                        # average with the horizontally flipped image
            p = (p + net(x.flip(3)).softmax(1)) / 2
    cam = gradcam(net, target_layer(net, m["arch"]), x, int(p.argmax()))
    top = p[0].cpu().topk(5)
    result = [(NAMES[i], float(v)) for v, i in zip(top.values, top.indices)]
    crop = T.CenterCrop(m["size"])(T.Resize(int(m["size"] * 1.14))(img))   # the region the network sees
    heat = Image.fromarray((cam * 255).astype("uint8")).resize(crop.size, Image.BILINEAR)
    rgb = (matplotlib.colormaps["jet"](np.asarray(heat) / 255.0)[..., :3] * 255).astype("uint8")
    return result, Image.blend(crop, Image.fromarray(rgb), 0.45)
