
# ✅ **README.md (verbessert)**

```markdown
# Stable Diffusion 1.5 – Realtime Webcam Scribble Renderer  
**Autor:** Hermann Knopp  
**Version:** 0.1 Alpha  

Dieses Projekt nutzt **Stable Diffusion 1.5** und **ControlNet Scribble**, um Bilder einer **Webcam in Echtzeit** zu erfassen und diese anschließend mittels eines Prompt-Textes zu transformieren.  
Es wird die Konturen des Webcam-Bildes über **Canny Edge Detection** extrahiert und als ControlNet-Referenz für Stable Diffusion verwendet.

---

## 🚀 Features

- **Realtime Webcam Capture** (DirectShow, Device 0)
- **Prompt-gesteuerte Stable-Diffusion-Transformation**
- **ControlNet Scribble (Canny-Modus)**
- **Batch Rendering** (`-batch 10`)
- **Preview-Fenster** via PyQt6
- **GPU-optimiert (Low VRAM Mode)**  
  - ~2–2.5 GB VRAM Verbrauch (RTX 3060 getestet)
- **Automatische Seed-Generierung**
- **Konfigurierbare Input-/Output-Ordner**
- **Speicherung der generierten Bilder als JPG mit Zeitstempel**

---

## 🖼 Funktionsweise in Kürze

1. Programm starten  
2. Webcam zeigt Livebild  
3. Taste **`p`** drücken → Render-Modus  
4. Prompt eingeben  
5. Optional:  
```

am orange cat -batch 10

````
6. Das aktuelle Webcam-Bild wird gespeichert, in ein **Canny-Scribble** umgewandelt und anschließend mit Stable Diffusion gerendert.  
7. Das Ergebnis erscheint im Qt-Preview-Fenster und wird im Ausgabeverzeichnis gespeichert.

---

## 📦 Installation

### 1. Python Virtual Environment erstellen
Das Projekt benötigt **Python 3.10.10 (x64)**.

```bash
python -m venv venv
venv\Scripts\activate
````

### 2. Dependencies installieren

```bash
pip install -r requirements.txt
```

### 3. Projektdateien kopieren

Das Verzeichnis `webcam_diffusion` in die Umgebung kopieren.

### 4. Programm starten

```bash
python -m webcam_diffusion
```

---

## ⚙️ Hardware-Anforderungen

* **GPU:** NVIDIA RTX 3060 (oder besser)
* **VRAM-Verbrauch:**

  * *High VRAM Modus*: ~9 GB
  * *Low VRAM Modus (Standard)*: 2–2.5 GB

---

## 📁 Ordnerstrukturen

Das Programm fragt beim Start nach Eingabe-/Ausgabeordnern:

* **Input Folder:** Enthält das Webcam-Bild als BMP (wird automatisch erzeugt)
* **Output Folder:** Speichert gerenderte JPG-Ergebnisse (z. B. `test_18052024_153045.jpg`)

---

## 🔧 Modelle

Beim Start wirst du gefragt, welches Stable-Diffusion-Modell geladen werden soll:

| Auswahl | Modell                           | VRAM | Beschreibung                    |
| ------- | -------------------------------- | ---- | ------------------------------- |
| **1**   | `nmkd/stable-diffusion-1.5-fp16` | 2 GB | Schnelle, VRAM-sparende Version |
| **2**   | `runwayml/stable-diffusion-v1-5` | 4 GB | FP32, höhere Qualität           |

ControlNet Modell wird automatisch geladen:

```
lllyasviel/sd-controlnet-scribble
```

---

## 🎛 Bedienung

| Taste | Funktion                 |
| ----- | ------------------------ |
| **p** | Rendering starten        |
| **q** | Webcam-Fenster schließen |

---

## 🧩 Batch Rendering

Die Batch-Funktion wird durch den Zusatz `-batch` im Prompt aktiviert:

```
city at night -batch 20
```

Maximal 1000 Bilder werden erlaubt.

---

## 📝 Bekannte Einschränkungen

* Nur **Webcam Device 0** wird unterstützt
* GUI ist auf ein einfaches Preview-Fenster beschränkt
* Das Alt-Framework „Quicktime 6“ ist deaktiviert, aber Teile darin existieren noch im Code
* Stabilität hängt von VRAM und Diffusers-Version ab

---

## 📜 Lizenz

Dieses Projekt ist experimentell und dient ausschließlich zu Test- und Forschungszwecken.
Bitte betreibe keine Modellausgaben öffentlich ohne Lizenzprüfung.

---

````

---

# ✅ **requirements.txt (empfohlen)**

Basierend auf deinem Code und offiziellen Versionen für Diffusers + ControlNet:

```txt
torch
torchvision
torchaudio
diffusers>=0.20.0
transformers
accelerate
opencv-python
numpy
pillow
PyQt6
qt-thread-updater
````

Optionale Pakete (falls du absolut versionierte Abhängigkeiten willst):

```txt
huggingface-hub
scipy
```

