# Student Score Prediction Based on Study Habits

**Question:** Can we predict a student’s final exam score using study hours and attendance data?  
**Summary:** This project builds a Linear Regression model (scikit-learn) that predicts a student's *Final_Score* from two predictors: **Hours_Studied** and **Attendance**.  
It includes data preprocessing, visualization (matplotlib + seaborn), training/testing split, model fitting, evaluation (R² & MAE), and a CLI to make new predictions.

---

## Project Structure

```
.
├── data
│   └── student_scores.csv        # sample dataset (you can replace/extend)
├── outputs                       # auto-created: plots, metrics, model, predictions
├── src
│   └── train_and_predict.py      # main script
├── requirements.txt
└── README.md
```

## Setup

### 1) Create & activate a virtual environment (recommended)
```bash
# Windows (Powershell)
python -m venv .venv
.\.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

## Run the Project

### A) Train, evaluate, and predict (default 4 hours, 80% attendance)
```bash
python src/train_and_predict.py
```

### B) Custom inputs
```bash
python src/train_and_predict.py --hours 4 --attendance 80
```

### C) Use a different CSV
```bash
python src/train_and_predict.py --csv path/to/your.csv
```

### Outputs
- `outputs/scatter_hours_vs_score.png` – scatter plot (Hours vs Score, colored by Attendance)
- `outputs/correlation_heatmap.png` – correlation heatmap
- `outputs/linear_regression_model.joblib` – saved model
- `outputs/metrics.json` – R² and MAE
- `outputs/prediction.json` – predicted score for the provided inputs

## Sample Dataset (data/student_scores.csv)

```
Hours_Studied,Attendance,Final_Score
5,90,85
3,60,55
6,95,90
2,50,45
8,98,95
4,70,65
7,92,88
1,40,35
9,96,97
10,99,99
```

## Methodology

1. **Import & Clean Data:** Drop NAs; keep sensible bounds (Hours ≥ 0; Attendance, Final_Score in [0, 100]).  
2. **Visualize:** Scatter plot (Hours vs Final Score, colored by Attendance) + correlation heatmap.  
3. **Train/Test Split:** 80/20 split with `random_state=42`.  
4. **Model:** `LinearRegression()` from scikit-learn.  
5. **Evaluate:** Report **R²** (coefficient of determination) and **MAE** (average absolute error).  
6. **Predict:** CLI predicts score for (hours, attendance), default `4` & `80` as required.

## Expected Output (as per assignment)
When run with `--hours 4 --attendance 80`, the script prints a prediction and saves metrics to `outputs/metrics.json`.

## Notes
- You may extend the dataset with real records; the model will automatically retrain.  
- For academic submission, include screenshots of the plots from `outputs/` and paste metrics/prediction into your report.