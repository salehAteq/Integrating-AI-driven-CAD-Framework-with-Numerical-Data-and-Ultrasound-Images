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
