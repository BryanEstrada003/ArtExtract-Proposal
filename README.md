# ArtExtract-Proposal
GSoC 2025 Evaluation Tasks — ArtExtract @ HumanAI

This repository contains Jupyter notebooks implementing the two evaluation tasks for the ArtExtract GSoC 2025 application.

---

## Task 1 — Convolutional-Recurrent Architecture for Artwork Classification
**Notebook:** [`Task1_Convolutional_Recurrent_Architecture.ipynb`](Task1_Convolutional_Recurrent_Architecture.ipynb)

Builds a **CNN-LSTM** multi-task classifier for predicting **Style**, **Artist**, and **Genre** of paintings from the [WikiArt / ArtGAN dataset](https://github.com/cs-chan/ArtGAN/blob/master/WikiArt%20Dataset/README.md).

Pipeline:
- **EfficientNet-B3** backbone (ImageNet pre-trained) for spatial feature extraction
- **Bidirectional LSTM** over the 7×7 patch-sequence of CNN feature maps
- **Attention pooling** to focus on discriminative regions
- **Multi-task classification heads** (Style / Artist / Genre) with uncertainty-weighted loss
- **Outlier detection** via cosine distance to class centroids in embedding space
- **t-SNE visualisation** of the learned embedding space

## Task 2 — Painting Similarity Search
**Notebook:** [`Task2_Painting_Similarity.ipynb`](Task2_Painting_Similarity.ipynb)

Builds a visual similarity search engine for the [National Gallery of Art open dataset](https://github.com/NationalGalleryOfArt/opendata).

Pipeline:
- **DINO ViT-B/16** (self-supervised Vision Transformer) for rich visual embeddings
- **FAISS** approximate nearest-neighbour index for scalable retrieval
- **Reciprocal nearest-neighbour re-ranking** to improve precision
- **Pose similarity** module using MediaPipe body keypoints
- **Face similarity** module using FaceNet / ArcFace for portrait queries
- **Precision@K, nDCG@K** evaluation with metadata-based relevance labels
- **t-SNE visualisation** of the NGA collection embedding space

---

## Setup

```bash
pip install -r requirements.txt
jupyter notebook
```

Both notebooks run in **demo mode** automatically when the actual datasets are not present on disk, using synthetic data so the entire pipeline can be inspected end-to-end.

To run with real data:
1. Download the [WikiArt dataset](https://github.com/cs-chan/ArtGAN/blob/master/WikiArt%20Dataset/README.md) and place it at `data/wikiart/`
2. Clone the [NGA opendata repo](https://github.com/NationalGalleryOfArt/opendata) to `data/nga_opendata/`
3. Run the notebooks — images will be downloaded automatically via the NGA IIIF API
