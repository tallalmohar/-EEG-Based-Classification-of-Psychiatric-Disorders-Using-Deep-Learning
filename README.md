# SFU CMPT 340 Project - Group 10
# EEG-Based Classification of Psychiatric Disorders Using Deep Learning (EECPD)

This project applies supervised, semi-supervised, and unsupervised deep learning approaches to classify EEG recordings as normal or abnormal using the [TUH EEG Corpus](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/) dataset.

## Important Links

| [Timesheet](https://1sfu-my.sharepoint.com/:x:/g/personal/hamarneh_sfu_ca/EQ07LycgT7hOkF0sdB02oIIBBLzLK9IjtrYmOcOQgPjRVg?e=m0InGz) | [Slack channel](https://app.slack.com/client/T08645XD55G/C0877AWGWKC) | [Project report](https://www.overleaf.com/project/6772391967aad788ce163cd1) | 
|-----------|---------------|-------------------------|



## Video Demo
Demo [Video](https://youtu.be/5R2VOalZk5Q) of 2 minutes is recorded to explain the project work.


## Table of Contents
1. [Directory Structure](#structure)

2. [Installation](#installation)

3. [Reproducing this project](#repro)




<a name="demo"></a>
## 1. Directory Structure

Each folder contains analysis for each Supervised, Unsupervised and Semi-Supervised.

```bash
2025_1_project_10
Abnormal_vs_Normal_Supervised/
├── config.py                        # All path definitions and constants
├── run_all.py                      # Main script to run preprocessing, training, and explainability
├── convert_edf_to_npy.py           # Converts raw EDF data to NPY format
├── train_model_v2.py               # Trains the CNN-based model on NPY data
├── evaluate_model.py               # Evaluates model on test data and plots metrics
├── explain_model.py                # Runs saliency and Grad-CAM visualizations
├── define_model.py                 # CNN architecture definition
├── saliency_maps.py                # Saliency map and top feature extraction
├── cam_explain.py                  # Grad-CAM implementation for EEG
├── process_metadata_iteratively.py # EDF metadata summarization
├── load_data_edf.py / load_data_npy.py  # EEG dataset loaders for EDF and NPY
├── load_trained_model.py          # Utility to load saved model
├── saved_models/
│   └── eeg_classifier.pth          # Trained model weights
├── XAI_Visuals/
│   ├── Saliency/                   # Saliency maps per class
│   ├── GradCAM/                    # Grad-CAM visualizations
│   ├── Feature_Importance/         # Top EEG channels from saliency
│   ├── Loss/                       # Loss and accuracy training curves
│   └── Metrics/                    # Confusion matrix, ROC curve
Unsupervised/
├── unsupervised-model.py/ipynb     # KMeans-based clustering pipeline
├── unsupervised-exploration.py     # Preprocessing and visualizations
├── Subset-Visualizations/          # PCA, t-SNE, cluster plots
abnorm_vs_norm_SSupervised/
├── plots/                        # Visual outputs from semi-supervised training
│   ├── accuracy_curve.png        # Accuracy across training epochs
│   ├── conf_ssl.png              # Confusion matrix of SSL model
│   ├── loss_curve.png            # Training and validation loss curve
│   └── saliency_map.png          # Saliency map for feature interpretation
├── saved_models/
│   └── eeg_classifier.pth        # Trained semi-supervised model
├── src/
│   ├── ssl_preprocess.py          # Preprocessing EEG data for SSL
│   ├── runall.py                  #Main Script for SSl Model       
│   ├── evaluation/
│   │   └── eval_ssl.py           # Evaluation script for SSL model
│   ├── training/
│   │   ├── config.py             # SSL configuration file
│   │   └── train_ssl.py          # SSL training script
Miscellaneous Files
├── requirements.txt / .yml         # Environment setup
├── edf_metadata_summary_*.txt      # Dataset summaries (train/test)
├── requirement_check.py            # Script to check library dependencies
```

<a name="installation"></a>
## 2. Installation

To install the project, clone the repository, navigate into the project folder, and set up the environment using the provided requirements.yml. Then activate the environment to run the code.

```bash
git clone https://github.com/your-username/2025_1_project_10.git
cd 2025_1_project_10
conda env create -f requirements.yml
conda activate eeg_env
```
If you are unable to use yml file, we have also added requirements.txt file.

<a name="repro"></a>
## 3. Reproduction
Data can be downloaded from [TUH EEG Corpus](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/). Please contact the university to get credentials to access the data.

### Supervised Learning
Before running the whole pipeline, update all paths and NUM_EPOCHS in config.py to point to your local dataset and project directory structure.
Once paths and number of epochs are set, run the following:
```bash
conda activate eeg_env
python run_all.py
```
This will sequentially execute:
- convert_edf_to_npy.py — Converts .edf files to .npy format.
- train_model_v2.py — Trains a CNN model using the EEG data.
- evaluate_model.py — Evaluates the model and saves performance metrics.
- explain_model.py — Generates saliency maps and top feature visualizations.

Outputs will be saved under XAI_Visuals/, saved_models/, and Metrics/.

To only test on the saved model, update data path in config.py and run the following:
```bash
conda activate eeg_env
python evaluate_model.py
```

To get only Saliency and CAM figures, update the path to EEG_FILE and run the following:
```bash
conda activate eeg_env
python explain_model.py
```
All figures will be saved in XAI_Visuals.

### Unsupervised Learning
Before running the scripts or notebooks, make sure to update the relevant EEG data file paths in each file.

To run the full clustering model and save the visualizations:
```bash
conda activate eeg_env
python unsupervised-model.py
```
This will: 
- Perform unsupervised clustering on the EEG data.
- Save all plots in the current working directory.
- Print clustering results and metrics to the console.

Update file paths on lines 169, 170, 211, and 212 of unsupervised-model.py before running.

To run exploratory visualizations interactively:

```bash
conda activate eeg_env
python unsupervised-exploration.py
```
This will:
- Load and visualize EEG data step-by-step.
- Display each plot (code will pause until you close the plot window).
- Print intermediate results to the console.

Update file paths on lines 149 and 150 of unsupervised-exploration.py before running.

To run the notebooks interactively:
- Open unsupervised-model.ipynb or unsupervised-exploration.ipynb in JupyterLab or Jupyter Notebook.
- Update EEG data file paths as needed.

Run all cells to generate clustering results and visualizations directly within the notebook.
### Semi-Supervised Learning
(abnorm_vs_norm_SSupervised)
Before running the full pipeline, make sure to:

Update all paths in config.py (e.g., dataset, output, model save directory).

Set the number of epochs in train_ssl.py if you want to modify training duration.

To run the full pipeline:
```bash
conda activate eeg_env
python src/runall.py
```
This will sequentially execute:

-ssl_preprocessy.py – Verifies and loads .npy EEG data.
-train_ssl.py – Trains the CNN-based semi-supervised model.
-eval_ssl.py – Evaluates model performance and generates visualizations

To evaluate only (using the trained model):
Update test data paths in config.py, then run:

```bash
conda activate eeg_env
python src/evaluation/eval_ssl.py
```
To generate Saliency and CAM visualizations only:
Ensure test data is correctly referenced in eval_ssl.py, then run:

```bash
conda activate eeg_env
python src/evaluation/eval_ssl.py
```
All visualizations will be saved in the plots/ folder.
