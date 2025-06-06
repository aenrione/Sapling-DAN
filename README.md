# Amazon Reviews Book - DAN & Sapling Similarity Fusion

## Overview

This repository explores the combination of two semantic similarity models — **Deep Averaging Network (DAN)** and **Sapling Similarity** — to improve performance on the **Amazon Reviews Book** dataset. The primary goal is to leverage the complementary strengths of both models in understanding and comparing natural language within book reviews.

## Current Status

* The current implementation is a **weighted (ponderated) combination** of the outputs from DAN and Sapling Similarity models.
* This is demonstrated in the notebook: **`PonderatedResult.ipynb`**.
* The weighting is manually tuned to balance the influence of both models on the final similarity score.

## Work in Progress

We are actively working toward a more **sophisticated fusion approach** that will go beyond simple weighting. The planned implementation may include:

* Learnable fusion layers
* Model stacking or ensembling techniques
* Task-specific fine-tuning on the Amazon Reviews Book dataset
* Evaluation and benchmarking across standard similarity metrics

## Dataset

We use the **Amazon Reviews Book** dataset, which contains user-generated reviews and metadata from books sold on Amazon. This dataset presents unique challenges due to:

* Variability in review length and quality
* Rich sentiment and semantic context
* Domain-specific language

## Future Goals

* Achieve better semantic understanding by merging DAN’s efficiency and Sapling’s contextual depth.
* Improve performance across downstream tasks such as duplicate detection, clustering, and sentiment alignment.

## How to Use

1. Clone this repository.
2. Open and run the notebook `PonderatedResult.ipynb` to view the current fusion method.
3. Follow progress and updates as the full model integration is developed.

## Contributions

We welcome feedback, suggestions, and contributions as we work toward building a more robust hybrid similarity model.


