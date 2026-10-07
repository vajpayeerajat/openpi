import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("train_logs.csv")

# Exponential moving average
alpha = 0.9  # smaller = smoother
train_smooth = data["train_loss"].ewm(alpha=alpha).mean()
val_smooth = data["val_loss"].ewm(alpha=alpha).mean()

plt.figure(figsize=(8, 5))

plt.plot(data["step"][1:], train_smooth[1:], label="train_loss")
plt.plot(data["step"][1:], val_smooth[1:], label="val_loss")

plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.grid(alpha=0.2)
for step in data["step"].iloc[4::6]:
    plt.axvline(x=step, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)

plt.tight_layout()
plt.savefig("train-val-curve.png", dpi=200)
plt.show()
