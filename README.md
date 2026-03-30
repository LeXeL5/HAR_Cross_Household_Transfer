# HAR_Cross_Household_Transfer

**Human Activity Recognition from Continuous Ambient Sensor Data**  
*Cross‑household transfer learning with XGBoost*

This project explores building a universal activity recognition model that can be quickly adapted to a new home using only a small amount of labeled data. It leverages the CASAS dataset from UCI Machine Learning Repository, containing sensor events from 30 different households.

## Goal

Develop a scalable approach to recognize daily activities in a new household by fine‑tuning a model pre‑trained on data from other households. The aim is to achieve high accuracy while minimizing the need for manual labeling in the target environment.

## Dataset

- **Source**: [UCI Machine Learning Repository – CASAS](https://archive.ics.uci.edu/ml/datasets/CASAS+Human+Activity+Recognition)
- **Description**: Sensor event sequences collected from 30 real apartments. Each record is a 30‑event window with 37 extracted features and an activity label.
- **Size**: 13,956,534 records, 37 attributes.
- **Characteristics**:
  - Strong class imbalance
  - Time‑based dependencies (overlapping windows)
  - Different sensor configurations per household
  - No missing values


## Technologies

- Python 3.9+
- pandas, numpy
- matplotlib, seaborn
- scikit‑learn
- xgboost
- (optional) spark for scaling experiments

## Approach

1. **Data Preprocessing**  
   - Load 30 CSV files, add `household_id` column.
   - Drop constant features (e.g., `numDistinctSensors`).
   - Encode activity labels consistently across households.

2. **Train/Test Split**  
   - For each household, use first 80% of time‑ordered records for training, last 20% for testing.
   - Target household: `target_train` (80%) and `target_test` (20%).
   - Other households: combine their training sets into `universal_train`, test sets into `universal_test`.

3. **Models**  
   - **Private model**: Trained only on `target_train`. Baseline for maximum achievable accuracy with full labeling.
   - **Universal model**: Trained on `universal_train` (all other households). Assesses generalizability without adaptation.
   - **Fine‑tuned models**: Universal model is sequentially fine‑tuned on increasing subsets (1%, 5%, 10%, 25%, 50%, 100%) of `target_train`.

4. **Evaluation**  
   - Fixed test set: `target_test` (last 20% of target household) for all experiments.
   - Metrics: Weighted F1‑score, confusion matrices.
   - Additional evaluation of universal model on `universal_test`.

## Key Results (Expected)

- Fine‑tuning the universal model on only **25%** of the target household’s labeled data achieves accuracy comparable to a private model trained on **100%** of that household’s data.
- XGBoost scales efficiently with parallel threads, reducing training time significantly.
- Feature importance analysis reveals how the universal model relies on general features (e.g., time‑based), while fine‑tuning adapts to household‑specific sensor patterns.
