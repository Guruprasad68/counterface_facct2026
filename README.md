# CounterFace

This repository contains code for the paper **"CounterFace: A Synthetic Face Dataset for Fine-Grained Counterfactual Evaluation of Face Recognition Systems"** accepted at ACM FAccT 2026. This paper is a part of the Face Recognition Fairness research at Wisconsin-Privacy and Security Lab [WI-PI Lab](https://github.com/wi-pi) at the University of Wisconsin-Madison.

## Dataset Access

The CounterFace dataset is available for **non-commercial research purposes only**. It cannot be used for training machine learning models.

To request access, contact:
- viswanathanr@wisc.edu
- kfawaz@wisc.edu

## What's in This Repository

This repository includes the scripts that implement our automated counterfactual face generation pipeline, plus the hyperparameters and prompts from our experiments. Use or adapt them to build similar datasets. Because the CounterFace dataset and pipeline details like names used to generate sources are not public, portions of the code use placeholders—you must substitute your own paths, credentials, and settings before running it end to end.

All results are included in the paper (main text or appendix).

## Repository Structure

```
├── candidate_generation/
│   ├── source_face/           # Generate source face images
│   │   ├── generate_source_faces.py
│   │   ├── embedding.py       # FaceNet embedding utilities
│   │   └── helper.py
│   └── modified_face/         # Apply attribute edits with SEGA
│       └── generate_modified_faces.py
│
├── rejection_sampling/
│   ├── generate_counterfactual_data.py   # Filter candidates via specificity
│   └── detectors/
│       ├── attribute_detector/   # VLM-based attribute detection (Claude, GPT-4o, Gemini)
│       └── artifact_detector/    # CLIP-based distortion detection
│
├── fr/
│   ├── aws_compare_faces.py         # Compare face images using AWS Rekognition
│   ├── facepp_compare_faces.py       # Compare face images using Face++ API
│   ├── find_threshold.py             # Find similarity threshold at target FMR
│   └── compute_match_rates.py        # Compute attribute-demographic wise match rates
│
└── utils/
    ├── attribute2edit_caption.json              # SEGA editing prompts per attribute
    ├── attribute2hyperparameters_men_sega.json  # SEGA hyperparams (men)
    ├── attribute2hyperparameters_women_sega.json # SEGA hyperparams (women)
    ├── attribute_conditioning.csv               # Specificity constraint matrix
    ├── best_attribute_detector_attribute_demo_wise.csv  # Best VLM per attribute-demographic
    └── few_shot_config.py                       # Few-shot prompting configuration
```

## Pipeline Overview

1. **Source Face Generation** (`candidate_generation/source_face/`)  
   Generate consistent face images for each identity using Stable Diffusion with similarity-based random seed generation.

2. **Attribute Editing** (`candidate_generation/modified_face/`)  
   Apply facial attribute edits using SEGA (Semantic Guidance for Diffusion).

3. **Rejection Sampling** (`rejection_sampling/`)  
   - Detect facial attributes in source and edited images using a combination of Claude, Gemini, and OpenAI models.
   - Detect generatied artifacts using CLIP embeddings + SVM
   - Filter to keep only images where the intended edit was applied satisfying Validity, Correctness, Specificity criterion.

4. **Face Recognition Evaluation** (`fr/`)  
   - Compare face images using commercial FR systems (AWS Rekognition, and Face++).
   - Find optimal similarity threshold at a target False Match Rate (FMR)
   - Compute attribute-demographic wise match rates and generate evaluation tables

## Supported Attributes

The pipeline supports 20 facial attributes in the paper and can be extended further with simple modifications to the files.

| Category | Attributes |
|----------|------------|
| Accessories | glasses, sunglasses, head_band, facemask, scarf |
| Facial Hair | goatee, mustache, thick_beard |
| Hair Style | buzz_cut, pigtails, shoulder_length_hair |
| Hair Color | blue_hair, red_hair, blond_hair, bald |
| Makeup | heavy_makeup, red_lipstick |
| Expression | smile |
| Skin Tone | light_colored_skin_tone, dark_colored_skin_tone |

## Setup

```bash
# Create conda environment
conda create -n counterface python=3.8 -y
conda activate counterface

# Install dependencies
pip install -r requirements.txt
```

For the VLM-based attribute detectors, you'll also need API keys for Claude, GPT-4o, and/or Gemini. Set these in a `.env` file or as environment variables.

## Citation

Guruprasad Viswanathan Ramesh, Ashish Hooda, Shimaa Ahmed, Harrison J Rosenberg, Ramya Korlakai Vinayak, and Kassem Fawaz. 2026. CounterFace: A Synthetic Face Dataset for Fine-Grained Counterfactual Evaluation of Face Recognition Systems. In The 2026 ACM Conference on Fairness, Accountability, and Transparency (FAccT '26), June 25–28, 2026, Montreal, QC, Canada. ACM, New York, NY, USA, 51 pages. https://doi.org/10.1145/3805689.3812410

## Questions or Issues

Raise an issue or contact the corresponding author for any doubts in the released material.

## License

This code is released for non-commercial research use only.
