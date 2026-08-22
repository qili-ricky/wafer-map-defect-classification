# Wafer Map Defect Pattern Classification

> Research portfolio project — AI for semiconductor manufacturing analytics.

## 1. Project Overview

This project explores machine learning methods for **wafer-level defect pattern classification** using publicly available datasets. The goal is to build a reproducible baseline, analyze real-world challenges such as class imbalance and pattern confusion, and gradually extend the work toward more robust and interpretable models.

### Why this topic

Wafer map defect patterns carry critical information about process failures in semiconductor manufacturing. Classifying these patterns accurately supports root cause analysis, yield improvement, and process monitoring. This project serves as a practical entry point into **AI for metrology / process monitoring**, combining semiconductor domain knowledge with machine learning methods.

### Research questions

- Can a CNN baseline achieve reasonable classification performance on public wafer map data?
- How does **class imbalance** affect model performance, and which mitigation strategies work best?
- Which defect patterns are most easily confused, and why?
- Can model interpretability methods (Grad-CAM, saliency maps) reveal physically meaningful regions?

## 2. Dataset

The primary dataset is the **WM-811K wafer map dataset** (811,457 wafer maps, 9 defect patterns + normal), the most widely used public benchmark in this field.

| Defect Pattern | Description |
|---|---|
| Normal | No defect |
| Center | Defect concentrated at center |
| Donut | Ring-shaped defect |
| Edge-Loc | Defect at edge location |
| Edge-Ring | Ring defect near edge |
| Local | Localized defect cluster |
| Random | Randomly distributed defects |
| Scratch | Linear scratch pattern |
| Near-full | Near-full wafer defect |
| None | Unlabeled |

> **Note:** Raw data is not stored in this repository due to size. Download instructions are in `notebooks/01_data_exploration.ipynb`.

## 3. Project Structure

```
wafer-map-defect-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/            # Original downloaded data (ignored by git)
│   └── processed/      # Preprocessed arrays / tensors (ignored by git)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_imbalance_analysis.ipynb
│   └── 05_interpretability.ipynb
├── src/
│   ├── data/           # Data loading, preprocessing, augmentation
│   ├── models/         # Model architectures (CNN, ResNet, etc.)
│   └── utils/          # Training loops, metrics, visualization
├── reports/
│   ├── figures/        # Generated plots and confusion matrices
│   └── tables/         # Result tables
└── references/         # Paper notes and bibliography
```

## 4. Setup

```bash
# Clone the repository
git clone git@github.com:qili-ricky/wafer-map-defect-classification.git
cd wafer-map-defect-classification

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## 5. Roadmap (12-week plan)

| Phase | Weeks | Goal | Deliverable |
|---|---|---|---|
| **Phase 1: Baseline** | 1–4 | Data exploration, preprocessing, CNN baseline | Working training pipeline + baseline results |
| **Phase 2: Improvement** | 5–8 | Class imbalance handling, data augmentation, model comparison (CNN vs ResNet vs ViT) | Improved results + confusion matrix analysis |
| **Phase 3: Interpretation** | 9–10 | Grad-CAM / saliency maps, error analysis | Interpretability results + discussion |
| **Phase 4: Writing** | 11–12 | Technical report (6–10 pages) | Report PDF in `reports/` |

### Current status: Phase 1 — Project initialization

## 6. Methods (planned)

- **Baseline:** Simple CNN (3–4 conv layers)
- **Comparison:** ResNet-18, Vision Transformer (small)
- **Imbalance handling:** Class weights, oversampling, SMOTE for images, focal loss
- **Augmentation:** Rotation, flip, synthetic wafer map generation
- **Interpretability:** Grad-CAM, confusion matrix, per-class precision/recall

## 7. References (key papers to read)

- Wu et al., "Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets" (WM-811K original paper)
- Nakazawa & Kulkarni, "Wafer Map Defect Pattern Classification and Annotation Using Convolutional Neural Networks"
- Recent works on class imbalance and few-shot wafer map classification

> Notes stored in `references/`.

## 8. Acknowledgments

This is a personal research portfolio project. All data used is from public sources. No proprietary or company-confidential data is involved.
