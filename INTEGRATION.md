
# Integration Guide

## What this package does

It adds ML bubble classification to your existing NEET/KCET OMR system.

Keep your existing:

- `app.py`
- marker detection
- perspective transform
- NEET/KCET JSON templates
- paper-code detection
- answer-key lookup
- scorer

Replace only the old threshold-based bubble classification.

---

## Step 1 — Extract this folder into your project

Your project should look like:

```text
omr-scanner/
├── app.py
├── scanner.py
├── scorer.py
├── config.py
├── templates/
│   ├── neet.json
│   └── kcet.json
├── ml_omr/
│   ├── __init__.py
│   ├── model.py
│   ├── inference.py
│   └── hybrid_reader.py
├── scripts/
│   ├── build_training_dataset.py
│   ├── train_model.py
│   └── test_model.py
├── dataset/
├── models/
└── requirements-ml.txt
```

Use your own existing `templates/neet.json` and `templates/kcet.json`.

---

## Step 2 — Install ML dependencies locally

PowerShell:

```powershell
cd E:\omr-scanner

python -m venv .venv

.venv\Scripts\Activate.ps1

pip install -r requirements-ml.txt
```

Do this for training locally.

---

## Step 3 — Extract the synthetic dataset ZIP

For example:

```text
E:\datasets\omr_synthetic_mobile_dataset\
```

It should contain:

```text
dataset.json
neet\images\
neet\labels\
kcet\images\
kcet\labels\
```

---

## Step 4 — Convert full synthetic sheets to bubble training images

From your project:

```powershell
python scripts\build_training_dataset.py E:\datasets\omr_synthetic_mobile_dataset
```

This fills:

```text
dataset\train\blank
dataset\train\filled
dataset\train\ambiguous

dataset\val\blank
dataset\val\filled
dataset\val\ambiguous
```

The script splits by whole sheets, not by individual bubbles. This helps prevent validation leakage.

---

## Step 5 — Add real mobile data

Synthetic data is bootstrap data.

For production accuracy, add real mobile bubble crops to the same folders.

Important variations:

- Samsung / iPhone / OnePlus / Redmi etc.
- daylight
- indoor warm light
- shadows
- slight motion blur
- black pen
- blue pen
- pencil
- light fills
- heavy fills
- erased marks
- ticks
- crosses

Aim to make validation data come from different sheets / captures than training data.

---

## Step 6 — Train

```powershell
python scripts\train_model.py
```

After training you should get:

```text
models\bubble_classifier.keras
models\class_names.json
```

The terminal will also show validation accuracy.

---

## Step 7 — Test individual crops

```powershell
python scripts\test_model.py dataset\val\filled\SOME_FILE.png
```

Example output:

```text
{
  'label': 'filled',
  'confidence': 0.98,
  ...
}
```

---

## Step 8 — Integrate into scanner.py

At the top of `scanner.py`:

```python
from ml_omr.hybrid_reader import scan_answers_ml
```

Keep your current perspective correction.

After you have a corrected image:

```python
if corrected.ndim == 3:
    gray = cv2.cvtColor(
        corrected,
        cv2.COLOR_BGR2GRAY,
    )
else:
    gray = corrected
```

Keep your existing:

```python
coordinates = generate_bubble_coordinates(
    template
)
```

Replace your old threshold answer scan with:

```python
answers, ml_debug = scan_answers_ml(
    gray=gray,
    coordinates=coordinates,
    crop_radius=int(
        template.get(
            "ml_crop_radius",
            16,
        )
    ),
    filled_confidence=0.70,
    ambiguous_confidence=0.60,
)
```

Then continue with your existing:

```text
paper-code detection
→ answer key loading
→ calculate_score()
```

Do not let ML calculate marks.

---

## Step 9 — Make sure corrected image dimensions match the JSON

For the uploaded templates, use the values from:

```python
template["sheet_width"]
template["sheet_height"]
```

Your perspective warp should output exactly those dimensions.

Without this, the JSON coordinates will not line up.

---

## Step 10 — Local test before Vercel

Run:

```powershell
uvicorn app:app --reload
```

Test:

1. known uploaded image
2. real mobile capture
3. blank sheet
4. several multiple answers
5. pencil
6. blue pen
7. shadow
8. slightly tilted sheet

The ML model should improve bubble classification, but OpenCV alignment still has to work first.

---

## Step 11 — Vercel warning

`tensorflow-cpu` is large and is not a good final Vercel dependency.

Do not add TensorFlow to your existing production `requirements.txt` yet.

First train and verify locally.

After the model is accurate, export it to ONNX or TFLite and use a lightweight runtime for deployment, or host the scanning backend in a persistent Python container.

A good production architecture is:

```text
Vercel
  frontend

      ↓ HTTPS API

Python container
  FastAPI
  OpenCV
  ONNX/TFLite model
  scorer
```

---

## Step 12 — What to measure

Do not judge the model only by bubble classification accuracy.

Measure:

- question-level accuracy
- number of false `MULTIPLE`
- number of false `BLANK`
- paper-code accuracy
- exact full-sheet match

The full-sheet result is what matters.
git status
 git add .
 git commit -m "neet code corrected"
 git push


 uvicorn app:app --reload  