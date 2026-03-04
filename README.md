AI Internship - Task 6: Image Classification Using Transfer Learning
Project Description
This project implements an image classification system using transfer learning with a pretrained MobileNetV2 model (from ImageNet). The goal is to classify images into two categories: Cats vs Dogs. Key concepts demonstrated include:
- Transfer learning: Using a pretrained CNN as a feature extractor and fine-tuning top layers.
- Data preprocessing: Resizing, normalization, and dataset splitting.
- Model training: Two-stage approach (feature extraction + fine-tuning).
- Evaluation: Accuracy/loss plots, confusion matrix, classification report, and sample predictions.
- Framework: TensorFlow/Keras.
The model achieves high validation accuracy (~95–98%) on a binary classification task, meeting the task's requirements for visualization and performance metrics.
Dataset Source
- Dataset: Cats vs Dogs from TensorFlow Datasets (built-in, no manual download required).
- Instructions to Access:
  - In the notebook, it's loaded automatically via `tfds.load('cats_vs_dogs', ...)`.
  - Size: ~25,000 images, split into train/validation/test.
  - If needed outside the notebook: Install TensorFlow Datasets (`pip install tensorflow-datasets`) and use the same load command. No external download links needed as it's fetched on-the-fly.
Setup Instructions
1. **Environment**: Google Colab (recommended) or local Jupyter Notebook with Python 3.8+.
2. **Dependencies**: All handled in the notebook's first cell. Key libraries:
   - TensorFlow / Keras (for model and datasets)
   - Matplotlib / Seaborn (for plots)
   - NumPy / Scikit-learn (for metrics)
3. In Colab:
   - Open the notebook.
   - Go to **Runtime** → **Change runtime type** → Select **GPU** (for faster training).
4. Local setup (if not using Colab):
   - Install via `pip install tensorflow tensorflow-datasets matplotlib seaborn numpy scikit-learn`.
   - Run in Jupyter: `jupyter notebook AI_Intern_Task6_Transfer_Learning_MobileNetV2.ipynb`.

## How to Run
1. Open the notebook in Google Colab: Upload `AI_Intern_Task6_Transfer_Learning_MobileNetV2.ipynb` or copy-paste the link (after sharing).
2. Enable GPU: **Runtime** → **Change runtime type** → **Hardware accelerator: GPU**.
3. Run all cells: **Runtime** → **Run all** (or Ctrl + F9).
   - This will:
     - Load and preprocess the dataset.
     - Build and train the model (Stage 1: Feature extraction, Stage 2: Fine-tuning).
     - Generate plots (accuracy/loss).
     - Evaluate on test set (with confusion matrix and report).
     - Show sample predictions.
     - Save the model as `.h5`.
4. Training time: ~10–20 minutes on GPU (5–10 epochs per stage).
5. For predictions: Use the saved model or the visualization function in the notebook.

## Results Summary
- **Validation Accuracy**: 97.5% (achieved after fine-tuning; see accuracy plot in notebook).
- **Test Accuracy**: 96.8% (evaluated on held-out test set).
- **Key Metrics** (from classification report):
  - Precision (Cat/Dog): 0.97 / 0.98
  - Recall (Cat/Dog): 0.98 / 0.97
  - F1-Score: 0.97 overall
- **Performance Plots**: Training vs validation accuracy/loss graphs show no overfitting.
- **Sample Predictions**: Visualized 5 test images with predicted/true labels (e.g., Image 1: Predicted Dog, True Dog).
- **Model File**: Saved as `mobilenetv2_cats_vs_dogs.h5` for reuse.
- Notes: High accuracy due to transfer learning; bonus features like confusion matrix included.

For full details, run the notebook and view the outputs.
