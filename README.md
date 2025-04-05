# Advancing Thyroid Diagnosis: Integrating AI-driven CAD Framework with Numerical Data and Ultrasound Images

## 1. Title

**Advancing Thyroid Diagnosis: Integrating AI-driven CAD Framework with Numerical Data and Ultrasound Images**

## 2. Description

This study presents an advanced computer-aided diagnosis (CAD) framework for thyroid disease that integrates numerical patient data and ultrasound images using cutting-edge technologies like Vision Transformers (ViTs) and SHapley Additive exPlanations (SHAP). The framework employs the Sparrow Search Algorithm (SSA) for optimized feature selection from numerical data and Tree-structured Parzen Estimator (TPE) for hyperparameter tuning. ViTs are used to analyze thyroid ultrasound images, while SHAP provides explainable AI insights into model predictions. Experiments were conducted on two datasets: the thyroid disease patient dataset and the DDTI: Thyroid Ultrasound Images dataset, with performance evaluated through five-fold and ten-fold cross-validation.

The framework achieved remarkable results, with models trained without data augmentation consistently outperforming augmented ones. For the thyroid disease patient dataset, the best-performing model reported 99.71% accuracy, while for the DDTI dataset, ViTs achieved 95.06% accuracy without augmentation. Key features like thyroxine, thyroidsurgery, and thyroid-stimulating hormone (TSH) were identified as critical predictors of thyroid conditions.

This research highlights the potential of AI-driven approaches in healthcare, paving the way for improved diagnostic outcomes and personalized treatment strategies, while also emphasizing the importance of transparency and interpretability in AI-assisted medical decision-making.

## 3. Dataset Information

The study utilized two primary datasets:

1. **Thyroid Disease Patient Dataset**:

   - Includes demographic information, medical history, current health status, and laboratory results like TSH, T3, TT4, T4U, and FTI levels.  
   - Accessible at: [Kaggle - Thyroid Disease Patient Dataset](https://www.kaggle.com/datasets/kapoorprakhar/thyroid-disease-patient-dataset)

2. **DDTI: Thyroid Ultrasound Images Dataset**: 

   - Consists of 99 cases and 134 ultrasound images, with expert annotations and patient information provided in XML files.  
   - Accessible at: [Kaggle - DDTI: Thyroid Ultrasound Images](https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images)

## 4. Code Information

The provided Python code introduces an implementation for generating SHAP (SHapley Additive exPlanations) explainability visualizations to interpret predictions made by Vision Transformer (ViT) models on thyroid ultrasound images. Key functions include:

- **`SHAPExplainability`**: Computes SHAP values for a given image and generates high-resolution visualizations.
- **`HandleSHAPExplainability`**: Manages the execution of SHAP explainability across multiple models and classes.

The code leverages libraries such as Hugging Face's `transformers`, `shap`, and `torch` to process images, compute SHAP values, and generate interpretable visualizations.

## 5. Usage Instructions

### Prerequisites

Ensure the following dependencies are installed:

- Python 3.8 or higher
- Libraries: `torch`, `transformers`, `shap`, `Pillow`, `numpy`, `pandas`, `matplotlib`

Install dependencies using pip:

```bash
pip install torch transformers shap Pillow numpy pandas matplotlib tqdm
```

### Steps to Run the Code
1. **Prepare the Dataset**:  
   Organize your dataset into subdirectories by class labels. For example:

```
dataset/
   ├── class_1/
   │   ├── image1.jpg
   │   ├── image2.jpg
   ├── class_2/
   │   ├── image3.jpg
   │   ├── image4.jpg
```


2. **Update the Script**: 

Modify the `datasetPath` variable in the script to point to your dataset directory. Add the paths to your pre-trained Vision Transformer models in the `modelNames` list.

3. **Run the Script**:

Execute the script using the following command:

```bash
python script_name.py
```

4. **Output**:

SHAP explainability visualizations will be saved as high-resolution `.jpg` files in the working directory.

## 6. Requirements

- **Hardware**: A GPU is recommended for faster inference with Vision Transformer models.
- **Software**: Python 3.8+, with the following libraries installed:

  - `torch`
  - `transformers`
  - `shap`
  - `Pillow`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `tqdm`

Install dependencies using pip:

```bash
pip install torch transformers shap Pillow numpy pandas matplotlib tqdm
```

## 7. Methodology

The methodology involves the following steps:

1. **Data Preprocessing**: 

   - Numerical data is processed using the Sparrow Search Algorithm (SSA) for feature selection.  
   - Hyperparameters are tuned using the Tree-structured Parzen Estimator (TPE).  

2. **Model Training**:  

   - Vision Transformers (ViTs) are trained on thyroid ultrasound images.  
   - Performance is evaluated through five-fold and ten-fold cross-validation.  

3. **SHAP Explainability**: 

   - SHAP values are computed for each image to provide interpretable insights into model predictions.  
   - The provided Python script automates the generation of SHAP visualizations, allowing users to interpret the contribution of each pixel in the image to the final classification decision.

## 8. Citations

If this dataset or code is used in research, please cite the following:

- Thyroid Disease Patient Dataset: [Kaggle - Thyroid Disease Patient Dataset](https://www.kaggle.com/datasets/kapoorprakhar/thyroid-disease-patient-dataset)  
- DDTI: Thyroid Ultrasound Images Dataset: [Kaggle - DDTI: Thyroid Ultrasound Images](https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images)
