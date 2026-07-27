# Fake-News-Misinformation-Detection-Platform

## Overview
This repository implements a hybrid fake-news and misinformation detection platform that combines text-based content analysis with propagation-based network analysis.

The system is designed to detect rumor and non-rumor posts by fusing:
- a DistilBERT-based text classification model for content/features
- a Graph Convolutional Network (GCN) model for propagation and thread structure
- a hybrid fusion model that leverages both representations for improved accuracy

## Project Structure
- `hybridmodel.ipynb` - notebook for training and evaluating the hybrid fusion pipeline.
- `propagation.ipynb` - notebook for building propagation graphs and training the GNN propagation model.
- `hybrid_fusion_model.pth` - saved hybrid fusion model weights.
- `gnn_propagation_model.pth` - saved propagation GNN model weights.
- `classification_report_new.txt` - classification report results for the hybrid model.
- `hybrid_metrics.txt` - evaluation metrics for the hybrid model.
- `hybrid_predictions_new.csv` - hybrid model predictions output.
- `hybrid_probabilities_new.csv` - hybrid model probability scores.
- `hybrid_roc_curve.png` - ROC curve plot for the hybrid classifier.

## Methodology
### Content Model
The text/content branch uses a DistilBERT-based classifier trained on rumor detection data. It extracts semantic text embeddings from the PHEME rumor dataset and classifies messages as rumor or non-rumor.

### Propagation Model
The propagation branch converts conversation threads into graph data and trains a Graph Convolutional Network (GCN). This branch captures structural and propagation behavior in social threads.

### Hybrid Fusion
The final hybrid system fuses text embeddings from DistilBERT with graph embeddings from the GCN. The joint model improves prediction performance over either component alone.

## Dataset
The notebooks reference the PHEME rumor dataset (annotated threads with rumor/non-rumor labels). The dataset is used to create:
- text sequences for the DistilBERT model
- graph structures for the GNN propagation model
- combined feature embeddings for the hybrid fusion model

## Results
The hybrid model achieves strong performance on rumor detection:
- Accuracy: 0.9534
- Precision: 0.9333
- Recall: 0.9378
- F1 Score: 0.9356
- Balanced Accuracy: 0.94998
- MCC: 0.8990
- ROC-AUC: 0.9864

Confusion matrix:
```
[[356, 14],
 [13, 196]]
```

## Getting Started
1. Open `propagation.ipynb` and `hybridmodel.ipynb` in Google Colab or a local Jupyter environment.
2. Install dependencies such as `torch`, `torch_geometric`, `transformers`, `datasets`, `scikit-learn`, `matplotlib`, and `pandas`.
3. Download and prepare the PHEME rumor dataset as shown in the notebooks.
4. Run the notebooks to train, evaluate, and save model checkpoints.

## Notes
- The repository currently includes model artifacts and evaluation outputs, but the underlying dataset is prepared within the notebooks.
- The hybrid approach is intended for research and experimentation in fake-news/misinformation detection using both content and propagation signals.

## Contact
For questions or improvements, open an issue or submit a pull request.
