#!/usr/bin/env python
# coding: utf-8

# In[69]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Make sure output folder exists
outdir = "outputs"
os.makedirs(outdir, exist_ok=True)

# Load data (either from CSV or manual sample)
df = pd.read_csv(r"C:\Users\mdasf\Downloads\student-score-prediction-github\student-score-prediction\data\student_scores.csv")

# Scatter plot
plt.figure()
sns.scatterplot(
    x="Hours_Studied",
    y="Final_Score",
    hue="Attendance",
    data=df
)
plt.title("Study Hours vs Final Score (Attendance as color)")
plt.savefig(os.path.join(outdir, "scatter_hours_vs_score.png"), bbox_inches="tight")
plt.close()
visualize(df)
model, r2, mae = train_and_evaluate(df)
prediction = predict(model, hours=4, attendance=80)

print("=== Model Performance ===")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print("\n=== Example Prediction ===")
print(f"Predicted score for 4 study hours & 80% attendance: {prediction}")

if __name__ == "__main__":
    main()


# In[72]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Make sure output folder exists
outdir = "outputs"
os.makedirs(outdir, exist_ok=True)

# Sample dataset (replace with your CSV if you have one)
df = pd.read_csv(r"C:\Users\mdasf\Downloads\student-score-prediction-github\student-score-prediction\data\student_scores.csv")

# --- Scatter Plot ---
plt.figure()
sns.scatterplot(
    x="Hours_Studied",
    y="Final_Score",
    hue="Attendance",
    data=df,
    palette="viridis"
)
plt.title("Study Hours vs Final Score (Attendance as color)")
plt.savefig(os.path.join(outdir, "scatter_hours_vs_score.png"), bbox_inches="tight")
plt.show()

# --- Correlation Heatmap ---
plt.figure()
corr = df[["Hours_Studied", "Attendance", "Final_Score"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.savefig(os.path.join(outdir, "correlation_heatmap.png"), bbox_inches="tight")
plt.show()



# In[ ]:




