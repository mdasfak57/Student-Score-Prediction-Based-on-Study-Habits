import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from joblib import dump

def load_data(csv_path: str) -> pd.DataFrame:
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        # Fallback mini dataset
        df = pd.DataFrame({
            "Hours_Studied": [5,3,6,2,8,4,7,1,9,10],
            "Attendance": [90,60,95,50,98,70,92,40,96,99],
            "Final_Score": [85,55,90,45,95,65,88,35,97,99],
        })
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Basic cleaning & sensible bounds
    df = df.dropna(subset=["Hours_Studied", "Attendance", "Final_Score"])
    df = df[(df["Hours_Studied"] >= 0) &
            (df["Attendance"].between(0, 100)) &
            (df["Final_Score"].between(0, 100))]
    df = df.reset_index(drop=True)
    return df

def visualize(df: pd.DataFrame, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)

    # Scatter plot
    plt.figure()
    sns.scatterplot(x="Hours_Studied", y="Final_Score", hue="Attendance", data=df)
    plt.title("Study Hours vs Final Score (colored by Attendance)")
    plt.savefig(os.path.join(outdir, "scatter_hours_vs_score.png"), bbox_inches="tight")
    plt.close()

    # Correlation heatmap
    plt.figure()
    corr = df[["Hours_Studied", "Attendance", "Final_Score"]].corr()
    sns.heatmap(corr, annot=True)
    plt.title("Feature Correlation Heatmap")
    plt.savefig(os.path.join(outdir, "correlation_heatmap.png"), bbox_inches="tight")
    plt.close()

def train_and_evaluate(df: pd.DataFrame, outdir: str):
    X = df[["Hours_Studied", "Attendance"]].values
    y = df["Final_Score"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    os.makedirs(outdir, exist_ok=True)
    dump(model, os.path.join(outdir, "linear_regression_model.joblib"))

    metrics = {"r2": float(r2), "mae": float(mae)}
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return model, metrics

def predict(model, hours: float, attendance: float, outdir: str):
    new_data = np.array([[hours, attendance]], dtype=float)
    pred = model.predict(new_data)[0]
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "prediction.json"), "w") as f:
        json.dump({"hours": hours, "attendance": attendance, "predicted_score": float(round(pred, 2))}, f, indent=2)
    return pred

def main():
    parser = argparse.ArgumentParser(description="Student Score Prediction (Linear Regression)")
    parser.add_argument("--csv", type=str, default=os.path.join("data", "student_scores.csv"), help="Path to CSV data")
    parser.add_argument("--hours", type=float, default=4.0, help="Study hours for prediction")
    parser.add_argument("--attendance", type=float, default=80.0, help="Attendance percentage for prediction")
    parser.add_argument("--outputs", type=str, default=os.path.join("outputs"), help="Directory to save outputs")
    args = parser.parse_args()

    df = load_data(args.csv)
    df = clean_data(df)

    visualize(df, args.outputs)
    model, metrics = train_and_evaluate(df, args.outputs)

    prediction = predict(model, args.hours, args.attendance, args.outputs)

    print("\n=== Model Performance ===")
    print(f"R2 Score: {metrics['r2']:.4f}")
    print(f"Mean Absolute Error: {metrics['mae']:.4f}")

    print("\n=== Example Prediction ===")
    print(f"Predicted score for {args.hours} study hours & {args.attendance}% attendance: {prediction:.2f}")

if __name__ == "__main__":
    main()