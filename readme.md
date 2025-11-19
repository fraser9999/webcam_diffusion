
# **README.md **

```markdown
# Stable Diffusion 1.5 – Realtime Webcam Scribble Renderer  
**Author:** Hermann Knopp  
**Version:** 0.1 Alpha  

This project uses **Stable Diffusion 1.5** together with **ControlNet Scribble** to capture images from a **webcam in realtime** and transform them using a text prompt.  
The webcam frame is converted into a **Canny edge scribble**, which is then fed into a Stable Diffusion ControlNet pipeline to generate new images.

---

## 🚀 Features

- **Realtime webcam capture** (DirectShow, device 0)
- **Text-prompt–controlled Stable Diffusion rendering**
- **ControlNet Scribble (Canny mode)**
- **Batch rendering support** using `-batch 10`
- **Preview window via PyQt6**
- **Optimized for low VRAM GPUs**
  - Approx. **2–2.5 GB VRAM usage** (tested on RTX 3060)
- **Automatic seed generation**
- **Configurable input/output folders**
- **Rendered images saved as timestamped JPG files**

---

## 🖼 Workflow Overview

1. Start the application  
2. Live webcam image appears  
3. Press **`p`** to enter render mode  
4. Enter a **positive prompt**  
5. Optionally add batch mode:  
```

a cute orange cat -batch 10

````
6. The current webcam frame is saved, converted to a **Canny scribble**, and processed by Stable Diffusion  
7. Result images appear in the PyQt preview window and are written to disk  

---

## 📦 Installation

### 1. Create a Virtual Environment  
Requires **Python 3.10.10 (x64)**.

```bash
python -m venv venv
venv\Scripts\activate
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Copy project files

Place the `webcam_diffusion` directory into your virtual environment.

### 4. Start the renderer

```bash
python -m webcam_diffusion
```

---

## ⚙️ Hardware Requirements

* **GPU:** NVIDIA RTX 3060 or better
* **VRAM usage:**

  * High VRAM mode: ~9 GB
  * Low VRAM mode (default): **2–2.5 GB**

---

## 📁 Folder Structure

At launch, the program asks for custom input/output folders.

* **Input folder:** stores the captured BMP webcam frame
* **Output folder:** stores rendered JPG results (e.g., `test_18052024_153045.jpg`)

---

## 🔧 Model Selection

At startup choose one of the Stable Diffusion 1.5 variants:

| Option | Model                            | VRAM | Description              |
| ------ | -------------------------------- | ---- | ------------------------ |
| **1**  | `nmkd/stable-diffusion-1.5-fp16` | 2 GB | Fast, low VRAM footprint |
| **2**  | `runwayml/stable-diffusion-v1-5` | 4 GB | FP32, higher quality     |

ControlNet model used:

```
lllyasviel/sd-controlnet-scribble
```

---

## 🎛 Controls

| Key   | Action                              |
| ----- | ----------------------------------- |
| **p** | Start rendering / enter prompt mode |
| **q** | Quit webcam preview                 |

---

## 🧩 Batch Rendering

Batch rendering is activated through the prompt:

```
city at night -batch 20
```

Maximum batch size: **1000 images**

---

## 📝 Known Limitations

* Only **Webcam Device 0** is supported
* GUI is limited to a simple preview window
* Some legacy “Quicktime 6” code is present but unused
* Stability may vary depending on VRAM and Diffusers version

---

## 📜 License

This project is experimental and intended for testing and research only.
Do not deploy the generated output in public environments without checking model and dataset licenses.

```

