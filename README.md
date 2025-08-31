# P17_2024-25: Building a Foundation Model for Modelling the World's Oceans

**LSE Department of Statistics - MSc Data Science Capstone Project 2024/2025**

**Candidate Numbers:** 50270, 50450, 51348, 45278

---

## 1. Project Overview

This repository contains the complete codebase and documentation for our capstone project, which focuses on building a practical forecasting pipeline for ocean-colour variables. The primary goal is to provide accurate, short-range predictions to compensate for gaps in daily satellite coverage, a common issue caused by cloud cover, sunglint, and other atmospheric obstructions.

Our research question is: **"Which modelling approaches can accurately forecast ocean variables from satellite observations, while addressing cloud cover gaps and high data dimensionality?"**

We utilize Level-3 (L3) daily data from the **Copernicus Marine Environment Monitoring Service (CMEMS)** for the south-west coast of England, from January 2022 to July 2025. The project involves a comprehensive workflow, including data preprocessing, imputation, exploratory data analysis (EDA), and a comparative evaluation of multiple forecasting models.

### Key Findings

After comparing a range of models—from simple baselines to advanced deep learning architectures—our analysis concluded that the **k-Means + Vector Autoregression (VAR) model** delivered the best performance. It consistently achieved the lowest error rates (SMAPE, RMSE, MAE) across all eight oceanographic variables on the final test set. This approach effectively captures both spatial and temporal dynamics by clustering homogeneous ocean regions before applying a time-series forecast.

---

## 2. Repository Structure

This repository is organized into a series of numbered folders that follow the project's workflow, from data acquisition to final modeling.

-   **1_All_Hands**: Contains project proposals, meeting notes, and draft versions of the final paper.
-   **2_Merging_Datasets**: Scripts and notebooks for downloading, merging, and conducting initial processing of the raw satellite data from CMEMS.
-   **3_EDA**: Notebooks dedicated to Exploratory Data Analysis. This includes generating temporal plots and spatial heatmaps to understand the underlying patterns in the data.
-   **4_Handling_Missing_Data**: Contains the implementation of our two-stage imputation strategy (temporal forward-fill and spatial k-d tree) to handle gaps in the satellite data.
-   **5_Creating_4D_tensors_and_2D_dataframes**: Scripts for transforming the cleaned data into the final formats required by the different models (e.g., 2D dataframes for VAR models and 4D tensors for CNN/GNN models).
-   **6_Models**: The core of the project, containing the implementation, training, and evaluation code for all forecasting models:
    -   Baseline Models (Moving Average, Exponential Smoothing)
    -   Vector Autoregression (VAR) Models
    -   k-Means + VAR
    -   ConvLSTM
    -   Temporal Attention CNN (TACNN)
    -   Edge-Aware GNN + LSTM
-   **model_checkpoints_20250424-122718**: Saved weights and checkpoints for the trained deep learning models to ensure reproducibility.

---

## 3. Getting Started

### Prerequisites

To run the notebooks and scripts in this repository, you will need a Python environment with standard data science libraries. Key dependencies include:
-   `pandas`
-   `numpy`
-   `xarray`
-   `scikit-learn`
-   `statsmodels`
-   `tensorflow` or `pytorch` (for deep learning models)
-   `matplotlib`, `seaborn` (for plotting)

### Datasets

The raw and preprocessed datasets are large and therefore hosted externally.
-   **Raw Dataset:** [Google Drive Link](https://drive.google.com/file/d/1YGkR2g09CjQvV_tAn2e-o12NZiwpPYQE/view?usp=sharing )
-   **Processed Dataset:** [Google Drive Link](https://drive.google.com/file/d/1C885-whkxjRABfQEShP5C-RFOlhOKg65/view?usp=drive_link )

Please download the processed dataset and place it in the relevant directory before running the modeling notebooks.

### How to Run the Code

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/lse-st498/P17_2024-25_ModellingWorldsOceans.git
    cd P17_2024-25_ModellingWorldsOceans
    ```
2.  **Set up the environment:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file should be created for easy setup ).*

3.  **Explore the notebooks:**
    You can run the Jupyter notebooks in order, starting from folder `2_Merging_Datasets` through to `6_Models`, to replicate the entire project workflow.

---

## 4. Final Report

For a detailed explanation of our methodology, data processing techniques, model architectures, and a full analysis of the results, please refer to our final project paper included in this repository as ST498_CapstoneProject_Group17.pdf.

---
