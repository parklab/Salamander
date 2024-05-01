#### Changelog

All noteable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---
---

## 0.4.0 - 2024-05
### Changed
  - Refactor using the AnnData and MuData data structures

## 0.3.2 - 2024-01
### Fixed
  - Support fixing the model variance of (multimodal) CorrNMF models during training

## 0.3.1 - 2023-12
### Fixed
  - Improve signature, history and embedding plots

## 0.3.0 - 2023-10
### Added
  - Support a sample-weigted KL-divergence loss in KL-NMF
  - Support a sample-weighted sparsity regularization in KL-NMF
  - Support fixing signature and sample biases of (multimodal) CorrNMF models during training

## 0.2.1 - 2023-10
### Fixed
  - Improve CorrNMF model formulation (added signature biases)
  - Improve multimodal exposure plot

## 0.2.0 - 2023-10
### Added
  - Support fixing arbitrary many a priori known signatures during model training
  - Improved performance with just-in-time compiled update rules

## 0.1.0 - 2023-10
### Added
  - First release of the non-negative matrix factorization (NMF) framework. Implemented algorithms: NMF with the generalized Kullback-Leibler divergence [(KL-NMF)](https://proceedings.neurips.cc/paper_files/paper/2000/file/f9d1152547c0bde01830b7e8bd60024c-Paper.pdf), minimum-volume NMF [(mvNMF)](https://arxiv.org/pdf/1907.02404.pdf), a version of correlated NMF [(CorrNMF)](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=87224164eef14589b137547a3fa81f06eef9bbf4), a multimodal version of correlated NMF [(MultimodalCorrNMF)](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=87224164eef14589b137547a3fa81f06eef9bbf4).
  - Install: `pip install salamander-learn==0.1.0`
