import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import shap
import matplotlib.pyplot as plt
from transformers import AutoFeatureExtractor, AutoModelForImageClassification


def SHAPExplainability(imagePath, imgClass, savedModelName, classes, topK=3, evals=5000, batchSize=64):
  """
  Perform SHAP explainability for a given image using a Vision Transformer model.
  """
  # Load the feature extractor and model
  featureExtractor = AutoFeatureExtractor.from_pretrained(savedModelName)
  model = AutoModelForImageClassification.from_pretrained(savedModelName)

  # Function to predict using the model
  def _predict(image):
    features = featureExtractor(image, return_tensors="pt")
    outputs = model(**features)
    logits = outputs.logits
    probabilities = logits.softmax(-1).tolist()
    return np.array(probabilities)

  # Load the image
  image = Image.open(imagePath).convert('RGB')
  image_np = np.array(image)

  # Create a masker for SHAP (inpainting with "telea" method)
  masker = shap.maskers.Image("inpaint_telea", image_np.shape)

  # Initialize the SHAP explainer
  explainer = shap.Explainer(_predict, masker, output_names=np.array(classes))

  # Compute SHAP values
  shapValues = explainer(
    np.array([image_np]),
    max_evals=evals,
    batch_size=batchSize,
    outputs=shap.Explanation.argsort.flip[:topK],
  )

  # Process SHAP values for visualization
  shapValues.data = shapValues.data[0]
  shapValues.values = [val for val in np.moveaxis(shapValues.values[0], -1, 0)]

  # Plot SHAP explanation
  shap.image_plot(
    shap_values=shapValues.values,
    pixel_values=shapValues.data,
    labels=shapValues.output_names,
    true_labels=[imgClass],
    show=False,
  )

  # Save the SHAP explanation plot
  plt.savefig(
    f"SHAP_Explainability_{os.path.basename(imagePath)}_{imgClass}_{os.path.basename(savedModelName)}.jpg",
    bbox_inches='tight',
    dpi=720,
  )
  plt.close()


# Main function to handle SHAP explainability for Vision Transformer models
def HandleSHAPExplainability(datasetPath, modelNames, numSamplesPerClass=10):
  """
  Handle SHAP explainability for multiple Vision Transformer models.

  Args:
      datasetPath (str): Path to the dataset directory containing class subdirectories.
      modelNames (list): List of saved model names (paths).
      numSamplesPerClass (int): Number of random samples per class to analyze.
  """
  # Get class labels
  classes = sorted(os.listdir(datasetPath))
  print(f"Classes: {classes}")

  # Iterate over each model
  for modelName in modelNames:
    print(f"Processing model: {modelName}")
    # Iterate over each class
    for cls in classes:
      clsPath = os.path.join(datasetPath, cls)
      images = os.listdir(clsPath)
      # Randomly select `numSamplesPerClass` images
      for _ in range(numSamplesPerClass):
        try:
          rndIdx = np.random.randint(0, len(images))
          imagePath = os.path.join(clsPath, images[rndIdx])
          print(f"Explaining image: {imagePath}")
          # Perform SHAP explainability
          SHAPExplainability(
            imagePath=imagePath,
            imgClass=cls,
            savedModelName=modelName,
            classes=classes,
            topK=3,
            evals=5000,
            batchSize=128,
          )
        except Exception as e:
          print(f"Error processing image {imagePath}: {e}")


# Example usage
if __name__ == "__main__":
  # Dataset path containing class subdirectories
  datasetPath = "..."

  # List of Vision Transformer models to analyze
  modelNames = [
    # Add your model paths here.
  ]

  # Run SHAP explainability
  HandleSHAPExplainability(datasetPath, modelNames, numSamplesPerClass=10)
