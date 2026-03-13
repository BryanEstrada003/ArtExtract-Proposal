# ArtExtract-Proposal
GSoC 2025 Evaluation Tasks — ArtExtract @ HumanAI

This repository contains Jupyter notebooks implementing the two evaluation tasks for the ArtExtract GSoC 2025 application.

---

## Task 1 — Convolutional-Recurrent Architecture for Artwork Classification
**Notebook:** [`Task1_Convolutional_Recurrent_Architecture.ipynb`](Task1_Convolutional_Recurrent_Architecture.ipynb)

Builds a **CNN-LSTM** multi-task classifier for predicting **Style**, **Artist**, and **Genre** of paintings from the [WikiArt / ArtGAN dataset](https://github.com/cs-chan/ArtGAN/blob/master/WikiArt%20Dataset/README.md).

Pipeline:
- **VGG16** backbone (ImageNet pre-trained) for robust spatial feature extraction.
- **Bidirectional LSTM** over the 7×7 patch-sequence of CNN feature maps.
- **Attention pooling** to focus on discriminative regions.
- **Multi-task classification heads** (Style / Artist / Genre) with uncertainty-weighted loss.

## Task 2 — Painting Similarity Search
**Notebook:** [`Task2_Painting_Similarity.ipynb`](Task2_Painting_Similarity.ipynb)

Builds a visual similarity search engine for the [National Gallery of Art open dataset](https://github.com/NationalGalleryOfArt/opendata).

Pipeline:
- **DINO ViT-B/16** (self-supervised Vision Transformer) for rich visual and semantic embeddings without the need for labeled data.
- **FAISS (IndexFlatL2)** for scalable, sub-second nearest-neighbour retrieval.
- **Data Verification Pipeline** to ensure strict synchronization and error-handling between dataset metadata and locally downloaded files.
- **Quantitative Evaluation** calculating **Mean Precision@10** and **mAP**, using artist attribution as a mathematical proxy for visual similarity.
- **Qualitative Visual Gallery** for side-by-side inspection of query results and semantic clustering validation.

---

## Setup

```bash
pip install -r requirements.txt
jupyter notebook