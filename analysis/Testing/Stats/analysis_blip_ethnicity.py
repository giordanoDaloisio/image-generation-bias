import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import statistics

df_results = pd.DataFrame(columns=["Type", "Accuracy", "Weighted F1"])

# Generals
data_g3 = pd.read_csv("Testing/Stats/ehtnicity_labellings/G3_ethnicities.csv")
g3_blip = data_g3["Blip Labelling"].tolist()
g3_manual = data_g3["Manual Labelling"].tolist()
score_1_g3 = accuracy_score(g3_manual, g3_blip)
score_2_g3 = f1_score(g3_manual, g3_blip, average="weighted")
df_results.loc[len(df_results.index)] = ["General_3", score_1_g3, score_2_g3]

data_g2 = pd.read_csv("Testing/Stats/ehtnicity_labellings/G2_ethnicities.csv")
g2_blip = data_g2["Blip Labelling"].tolist()
g2_manual = data_g2["Manual Labelling"].tolist()
print(g2_blip, g2_manual)
score_1_g2 = accuracy_score(g2_manual, g2_blip)
score_2_g2 = f1_score(g2_manual, g2_blip, average="weighted")
df_results.loc[len(df_results.index)] = ["General_2", score_1_g2, score_2_g2]

data_g2 = pd.read_csv("Testing/Stats/ehtnicity_labellings/G_xl_ethnicities.csv")
g2_blip = data_g2["Blip Labelling"].tolist()
g2_manual = data_g2["Manual Labelling"].tolist()
print(g2_blip, g2_manual)
score_1_g2 = accuracy_score(g2_manual, g2_blip)
score_2_g2 = f1_score(g2_manual, g2_blip, average="weighted")
df_results.loc[len(df_results.index)] = ["General_XL", score_1_g2, score_2_g2]


# SE
data_se3 = pd.read_csv("Testing/Stats/ehtnicity_labellings/SE3_ethnicities.csv")
se3_blip = data_se3["Blip Labelling"].tolist()
se3_manual = data_se3["Manual Labelling"].tolist()
score_1_se3 = accuracy_score(se3_manual, se3_blip)
score_2_se3 = f1_score(se3_manual, se3_blip, average="weighted")
df_results.loc[len(df_results.index)] = ["SE_3", score_1_se3, score_2_se3]


data_se2 = pd.read_csv("Testing/Stats/ehtnicity_labellings/SE2_ethnicities.csv")
se2_blip = data_se2["Blip Labelling"].tolist()
se2_manual = data_se2["Manual Labelling"].tolist()
score_1_se2 = accuracy_score(se2_manual, se2_blip)
score_2_se2 = f1_score(se2_manual, se2_blip, average="weighted")
df_results.loc[len(df_results.index)] = ["SE_2", score_1_se2, score_2_se2]

data_se2 = pd.read_csv("Testing/Stats/ehtnicity_labellings/SE_xl_ethnicities.csv")
se2_blip = data_se2["Blip Labelling"].tolist()
se2_manual = data_se2["Manual Labelling"].tolist()
score_1_se2 = accuracy_score(se2_manual, se2_blip)
score_2_se2 = f1_score(se2_manual, se2_blip, average="weighted")
df_results.loc[len(df_results.index)] = ["SE_XL", score_1_se2, score_2_se2]

# Overall
overall_score_g1 = statistics.mean([score_1_g3, score_1_g2])
overall_score_g2 = statistics.mean([score_2_g3, score_2_g2])
df_results.loc[len(df_results.index)] = [
    "All General",
    overall_score_g1,
    overall_score_g2,
]

overall_score_se1 = statistics.mean([score_1_se3, score_1_se2])
overall_score_se2 = statistics.mean([score_2_se3, score_2_se2])
df_results.loc[len(df_results.index)] = ["All SE", overall_score_se1, overall_score_se2]

# all sd3
overall_score_sd3_1 = statistics.mean([score_1_g3, score_1_se3])
overall_score_sd3_2 = statistics.mean([score_2_g3, score_2_se3])
df_results.loc[len(df_results.index)] = [
    "SD3",
    overall_score_sd3_1,
    overall_score_sd3_2,
]
# all sd2
overall_score_sd2_1 = statistics.mean([score_1_g2, score_1_se2])
overall_score_sd2_2 = statistics.mean([score_2_g2, score_2_se2])
df_results.loc[len(df_results.index)] = [
    "SD2",
    overall_score_sd2_1,
    overall_score_sd2_2,
]

overall_score_1 = statistics.mean([overall_score_g1, overall_score_se1])
overall_score_2 = statistics.mean([overall_score_g2, overall_score_se2])
df_results.loc[len(df_results.index)] = [
    "Overall Accuracy",
    overall_score_1,
    overall_score_2,
]


df_results.to_csv("Testing/Stats/blip_ethnicity_analysis_f1.csv")
