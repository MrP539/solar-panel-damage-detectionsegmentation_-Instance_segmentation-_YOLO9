## conda environment :Car_Counting_LPR_Project


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"D:\machine_learning_AI_Builders\บท4\Segmentation\Instance_segmentation\runs\segment\train\results.csv")
df.columns = df.columns.str.strip()

plt.figure(figsize=(6,6))

plt.plot(df["epoch"],df["train/cls_loss"],label="train/cls_loss",color = "blue",marker="x")
plt.plot(df["epoch"],df["val/cls_loss"],label="val/cls_loss",color = "red",marker="x")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title('Train and Validation Loss by Epoch')
plt.show()
