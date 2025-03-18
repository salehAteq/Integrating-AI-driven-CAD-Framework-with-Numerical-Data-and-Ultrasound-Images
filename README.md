# Advancing Thyroid Diagnosis: Integrating AI-driven CAD Framework with Numerical Data and Ultrasound Images

## Overview

This study presents an advanced computer-aided diagnosis (CAD) framework for thyroid disease that integrates numerical
patient data and ultrasound images using cutting-edge technologies like Vision Transformers (ViTs) and SHapley Additive
exPlanations (SHAP). The framework employs the Sparrow Search Algorithm (SSA) for optimized feature selection from
numerical data and Tree-structured Parzen Estimator (TPE) for hyperparameter tuning. ViTs are used to analyze thyroid
ultrasound images, while SHAP provides explainable AI insights into model predictions. Experiments were conducted on two
datasets: the thyroid disease patient dataset and the DDTI: Thyroid Ultrasound Images dataset, with performance
evaluated through five-fold and ten-fold cross-validation. The framework achieved remarkable results, with models
trained without data augmentation consistently outperforming augmented ones. For the thyroid disease patient dataset,
the best-performing model reported 99.71% accuracy, while for the DDTI dataset, ViTs achieved 95.06% accuracy without
augmentation. Key features like thyroxine, thyroidsurgery, and thyroid-stimulating hormone (TSH) were identified as
critical predictors of thyroid conditions. This research highlights the potential of AI-driven approaches in healthcare,
paving the way for improved diagnostic outcomes and personalized treatment strategies, while also emphasizing the
importance of transparency and interpretability in AI-assisted medical decision-making.

## Explaination and Implementation

The provided Python code introduces an implementation for generating SHAP (SHapley Additive exPlanations) explainability visualizations to interpret predictions made by Vision Transformer (ViT) models on thyroid ultrasound images. The key function, SHAPExplainability, takes as input the path to an image, its corresponding class label, a pre-trained model, and other parameters such as evaluation steps and batch size. It leverages the Hugging Face transformers library to load a feature extractor and a pre-trained ViT model. The function processes each image through the model to generate prediction probabilities, then computes SHAP values that quantify the contribution of each pixel in the image to the final classification decision. These SHAP values are subsequently visualized using shap.image_plot and saved as high-resolution images for further inspection.  

The HandleSHAPExplainability function manages the execution of SHAP explainability across multiple models and classes. It iterates over a dataset directory containing subdirectories for each class, randomly selects a specified number of images per class, and applies the SHAPExplainability function. This ensures that insights are generated for a diverse set of samples. Notably, error handling is included to gracefully handle issues arising during image processing or model inference. 

To implement this code, users need to provide the path to their dataset, which should be organized into subdirectories by class labels, and specify the paths to the pre-trained models they wish to analyze. This README file would benefit from additional context about the purpose of the code, installation instructions for required dependencies (e.g., torch, transformers, shap, Pillow), and detailed steps for running the script, including example inputs for the dataset path and model names. Furthermore, clarifications regarding hardware requirements (e.g., GPU support for faster inference) and links to relevant documentation for the libraries used would enhance usability. 

## Materials

The study utilized two primary datasets:

Thyroid Disease Patient Dataset: This dataset includes a wide range of attributes such as demographic information (e.g.,
age, sex), medical history (e.g., thyroxine intake, antithyroid medications, past surgeries), current health status (
e.g., presence of goiter, tumor, or hypopituitary conditions), and laboratory results like TSH, T3, TT4, T4U, and FTI
levels. The dataset is accessible at: https://www.kaggle.com/datasets/kapoorprakhar/thyroid-disease-patient-dataset

DDTI: Thyroid Ultrasound Images Dataset: This open-access resource, supported by Universidad Nacional de Colombia,
CIM@LAB, and IDIME, consists of 99 cases and 134 ultrasound images. Each case comes with an XML file containing expert
annotations and patient information. The dataset can be found
at: https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images

## Conclusions of the Study

This study presented a comprehensive CAD framework for thyroid disease diagnosis, integrating numerical patient data and ultrasound images to enhance diagnostic accuracy and interpretability. By leveraging advanced technologies such as Vision Transformers (ViTs) and SHapley Additive exPlanations (SHAP), the framework demonstrated significant improvements in thyroid disease classification. The Sparse Search Algorithm (SSA) was employed for optimized feature selection, while the Tree-structured Parzen Estimator (TPE) was utilized for hyperparameter tuning, resulting in superior performance compared to existing methodologies. Notably, models trained without data augmentation consistently outperformed their augmented counterparts, achieving accuracy scores of up to 95.06% on ultrasound images and 99.71% on numerical data. These results underscore the robustness and reliability of the framework in handling diverse data modalities, making it a powerful tool for clinical decision support and treatment planning. The integration of explainable AI techniques, such as SHAP, provided transparent and interpretable diagnostic results, fostering trust and collaboration among healthcare practitioners and patients. By identifying key diagnostic features like thyroxine, thyroid-stimulating hormone (TSH), and thyroid surgery, the framework not only improved diagnostic accuracy but also enhanced the understanding of thyroid disease pathology, crucial for facilitating patient-centered care and improving health outcomes. 
