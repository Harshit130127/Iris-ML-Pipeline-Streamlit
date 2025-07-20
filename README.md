# Iris Flower Classification Web App

## Overview

This project implements a complete end-to-end machine learning pipeline on the Iris dataset and deploys an interactive web application using Streamlit. The workflow covers:

- data loading and cleaning
- feature engineering and preprocessing
- exploratory data analysis (EDA)
- model training, evaluation, and selection
- serialization of the trained model and scaler
- Streamlit frontend for user input, real-time predictions, and visual explanations


## Repository Structure

```
── iris_full_pipeline.ipynb   # jupyter notebook: full ml workflow  
── iris_model.pkl             # pickled randomforest model  
── iris_scaler.pkl            # pickled standardscaler  
── iris_slider_data.csv       # raw feature data for slider defaults  
── app.py                     # streamlit frontend application  
── README.md                  # project documentation  
```


## Requirements

- **python 3.7+**
- **packages:**
    - pandas
    - numpy
    - scikit-learn
    - seaborn
    - matplotlib
    - streamlit

Install dependencies with:

```
pip install pandas numpy scikit-learn seaborn matplotlib streamlit
```


## Usage

### 1. Run the Jupyter Notebook

1. Open a terminal and launch Jupyter Notebook:

```
jupyter notebook iris_full_pipeline.ipynb
```

2. Execute every cell in order. The notebook will:
    - load and inspect the raw iris dataset
    - clean data (drop duplicates, check missing values)
    - encode species labels and engineer a `petal_area` feature
    - scale features to zero mean and unit variance
    - perform EDA (pairplots, heatmap, boxplots)
    - split data into training and test sets
    - train a RandomForest classifier and evaluate performance
    - save `iris_model.pkl`, `iris_scaler.pkl`, and `iris_slider_data.csv`

### 2. Launch the Streamlit App

1. Ensure these files are in your working directory:
    - `app.py`
    - `iris_model.pkl`
    - `iris_scaler.pkl`
    - `iris_slider_data.csv`
2. Run Streamlit:

```
streamlit run app.py
```

3. In the browser interface:
    - adjust sliders (with units) for sepal length, sepal width, petal length, petal width, and petal area
    - click **Predict** to view:
        - predicted Iris species (setosa, versicolor, virginica)
        - bar chart of prediction probabilities
        - annotated feature importance chart highlighting the most influential measurement

## File Descriptions

- **iris_full_pipeline.ipynb:** step-by-step notebook covering data ingestion, preprocessing, EDA, model training, evaluation, and artifact saving
- **iris_model.pkl:** serialized RandomForest model for inference
- **iris_scaler.pkl:** serialized StandardScaler for feature normalization
- **iris_slider_data.csv:** inverse-transformed feature data used to set slider bounds and defaults
- **app.py:** Streamlit application code that loads artifacts, renders input sliders, runs predictions, and displays interactive visualizations


## Highlights

- **complete ml workflow:** from raw data to deployed model
- **interactive ui:** users input measurements and receive instant predictions
- **model interpretability:** feature importance chart with clear annotation of the top feature

*developed as part of a data science internship assignment*

