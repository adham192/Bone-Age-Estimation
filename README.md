# Bone Age Estimation from Hand X-rays

An automated deep learning system that predicts a child's skeletal maturity from pediatric hand X-rays. The model takes an X-ray image and patient gender as input, and outputs an estimated bone age range in years and months.

**Live App:** [Launch on Streamlit](https://bone-age-estimation-gp3un9gf7nhfeudztmnzof.streamlit.app/)

---

## Overview

Traditional bone age assessment requires a radiologist to manually compare a child's hand X-ray against a reference atlas a process that is slow, subjective, and dependent on specialist availability. This project automates that process using a fine-tuned **InceptionV3** model trained on the RSNA Bone Age dataset, achieving a Mean Absolute Error of **8.8 months**.

The system is designed as a **decision support tool**, it assists medical professionals and does not replace clinical judgment.

---

## How It Works

1. User uploads a hand X-ray image and selects patient gender
2. The image goes through a preprocessing pipeline: hand localization, cropping, CLAHE contrast enhancement, and Gaussian smoothing
3. The preprocessed image and gender are fed into the model
4. The predicted bone age is displayed as a range in years and months

---

## Model

The model uses **InceptionV3** (pretrained on ImageNet) fine-tuned on the RSNA dataset. It accepts two inputs: the X-ray image and gender, which are processed through separate branches and merged before the regression head.

Training was done in two phases:
- **Phase 1:** Train only the head (frozen base): MAE 12.15 months
- **Phase 2:** Fine-tune the full model MAE **8.8 months**

**Model weights are hosted on Hugging Face.** They are loaded automatically at runtime via `urllib.request.urlretrieve`. To find the URL or download the weights manually, open [`Application/app.py`](Application/app.py) and look for the Hugging Face URL at the top of the file.

---

## Results

| Phase | Description | MAE |
|---|---|---|
| Phase 1 | Head-only training (frozen base) | 12.15 months |
| Phase 2 | Full fine-tuning (InceptionV3 unfrozen) | **8.8 months** |

**Inference time:** 1–3 seconds per prediction.

**Test Cases:**

- **Male child (standard quality image):** Predicted bone age range 13 years 10 months to 14 years 6 months
- **Female child (noisy/low quality image):** Predicted bone age range 10 years 3 months to 10 years 11 months, demonstrating that the preprocessing pipeline handles poor image quality reliably

---

## Dataset

The model was trained on the [RSNA Bone Age dataset](https://www.kaggle.com/datasets/kmader/rsna-bone-age), accessed via **KaggleHub** inside Google Colab.

To replicate training, install KaggleHub and use it to pull the dataset:

```bash
pip install kagglehub
```


The full training notebook is available in the `Notebook/` folder.

---

## Project Structure

```
├── Application/
│   ├── app.py                      
│   ├── preprocessing.py            
│   └── normalisation_json.py      
│
├── Notebook/
│   └── Bone Age Estimation.ipynb   
│
└── requirements.txt
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
