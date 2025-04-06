# Fake Review Detection Using Machine Learning and Deep Learning

This project focuses on detecting fake reviews using a combination of machine learning models and a deep learning-based LSTM model. The dataset used for this project is `fake reviews dataset.csv`, which contains labeled reviews indicating whether they are fake or genuine.

## Features

- **Preprocessing**:
  - Text data is preprocessed by converting to lowercase and removing stop words using NLTK.
  - TF-IDF vectorization is applied to convert text data into numerical features.

- **Machine Learning Models**:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest
  - XGBoost

- **Deep Learning Model**:
  - LSTM (Long Short-Term Memory) implemented using PyTorch.

- **Evaluation**:
  - Models are evaluated using accuracy, classification reports, and confusion matrices.
  - Confusion matrices are visualized using Seaborn heatmaps.

## Workflow

1. **Data Loading**:
   - The dataset is loaded from `fake reviews dataset.csv`.

2. **Data Preprocessing**:
   - Text reviews are cleaned and preprocessed.
   - Labels are encoded into binary values (e.g., 0 for genuine, 1 for fake).

3. **Feature Extraction**:
   - TF-IDF vectorization is applied to convert text data into numerical features.

4. **Model Training and Evaluation**:
   - Machine learning models are trained using `GridSearchCV` for hyperparameter tuning.
   - The LSTM model is trained using PyTorch with a custom architecture.

5. **Results**:
   - Accuracy scores for all models are saved and compared.
   - Confusion matrices and classification reports are generated for detailed evaluation.

## Requirements

- Python 3.7+
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - nltk
  - seaborn
  - matplotlib
  - torch
  - xgboost

## How to Run

1. Install the required libraries:
   ```bash
   pip install pandas numpy scikit-learn nltk seaborn matplotlib torch xgboost
   ```

2. Place the dataset file `fake reviews dataset.csv` in the project directory.

3. Run the Jupyter Notebook or Python script to train and evaluate the models.

4. View the results, including accuracy scores and confusion matrices.

## Results

- The project compares the performance of machine learning models and the LSTM model.
- Accuracy scores and confusion matrices are used to evaluate the effectiveness of each model.
