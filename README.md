# Bone Age Estimation from Hand X-rays

An automated deep learning system that predicts a child's skeletal maturity from pediatric hand X-rays. The model takes an X-ray image and patient gender as input, and outputs an estimated bone age range in years and months.

**Live App:** [Launch on Streamlit](YOUR_STREAMLIT_URL_HERE)

---

## Overview

Traditional bone age assessment requires a radiologist to manually compare a child's hand X-ray against a reference atlas — a process that is slow, subjective, and dependent on specialist availability. This project automates that process using a fine-tuned InceptionV3 model trained on the RSNA Bone Age dataset, achieving a Mean Absolute Error of **8.8 months**.

The system is designed as a **decision support tool** — it assists medical professionals and does not replace clinical judgment.

---

## How It Works

1. User uploads a hand X-ray image and selects patient gender
2. The image goes through a preprocessing pipeline: hand localization, cropping, CLAHE contrast enhancement, and Gaussian smoothing
3. The preprocessed image and gender are fed into the model
4. The predicted bone age is displayed as a range in years and months

---

## Model

The model uses **InceptionV3** (pretrained on ImageNet) fine-tuned on the RSNA dataset. It accepts two inputs — the X-ray image and gender — which are processed through separate branches and merged before the regression head.

Training was done in two phases:
- **Phase 1:** Train only the head (frozen base) — MAE 12.15 months
- **Phase 2:** Fine-tune the full model — MAE **8.8 months**

**Model weights are hosted on Hugging Face.** They are loaded automatically at runtime via `urllib.request.urlretrieve`. To find the URL or download the weights manually, open [`Application/app.py`](Application/app.py) and look for the Hugging Face URL at the top of the file.

---

## Results

| Phase | Description | MAE |
|---|---|---|
| Phase 1 | Head-only training (frozen base) | 12.15 months |
| Phase 2 | Full fine-tuning (InceptionV3 unfrozen) | **8.8 months** |

**Inference time:** 1–3 seconds per prediction.

**Test Cases:**

- **Male child (standard quality image):** Predicted bone age range — 13 years 10 months to 14 years 6 months
- **Female child (noisy/low quality image):** Predicted bone age range — 10 years 3 months to 10 years 11 months, demonstrating that the preprocessing pipeline handles poor image quality reliably

For context, Paper 1 in the literature (BoneViewTM / ConvNeXt) reported an MAE of 5.9 months using a commercial, proprietary system. Paper 3 (MobileNetV3, no manual annotations) reported 6.2 months. This project achieves 8.8 months as an open, fully reproducible system built and trained from scratch.

---

## Dataset

The model was trained on the [RSNA Bone Age dataset](https://www.kaggle.com/datasets/kmader/rsna-bone-age), accessed via **KaggleHub** inside Google Colab.

To replicate training, install KaggleHub and use it to pull the dataset:

```bash
pip install kagglehub
```

```python
import kagglehub
path = kagglehub.dataset_download("kmader/rsna-bone-age")
```

The full training notebook is available in the `Notebook/` folder.

---

## Project Structure

```
├── Application/
│   ├── app.py                      # Streamlit app (contains Hugging Face model URL)
│   ├── preprocessing.py            # Image preprocessing pipeline
│   └── normalisation_json.py       # Loads normalization stats for denormalization
│
├── Notebook/
│   └── Bone Age Estimation.ipynb   # Full training notebook (Google Colab)
│
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
pip install -r requirements.txt
streamlit run Application/app.py
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | TensorFlow / Keras, InceptionV3 |
| Image Processing | OpenCV, CLAHE |
| Data | NumPy, Pandas, KaggleHub |
| Interface | Streamlit |
| Training Environment | Google Colab (NVIDIA T4 GPU) |

---

## References

1. Nguyen et al., 2023. High performance for bone age estimation with an artificial intelligence solution. *Diagnostic and Interventional Imaging*, 104(7-8).
2. Thodberg et al., 2022. Autonomous AI in pediatric radiology: BoneXpert for bone age assessment. *Pediatric Radiology*, 52(7).
3. Li et al., 2022. A deep learning-based CAD method of X-ray images for bone age assessment. *Complex & Intelligent Systems*, 8(3).
4. Raju & Venkateswara Rao, 2025. SCLe-Net for Hand Bone Age Estimation. *X-Ray Spectrometry*.
5. Kumaragurubaran et al., 2025. Bone-Age Assessment Using ResNet50 and DenseNet121. *GINOTECH 2025*, IEEE.

---

*Arab Open University, Egypt — Faculty of Computer Studies — 2025–2026*  
*Supervised by Dr. Rafeek Yanni*
