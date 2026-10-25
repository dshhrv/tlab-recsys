from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


batch_size = 256
data_path = Path("data/LastFM/train.csv")
output_path = Path("distribution_ml1m.png")
table_path = Path("exact_user_batch_message_counts.csv")

data = pd.read_csv(data_path, usecols=["user_id"])
data = data.iloc[: len(data) // batch_size * batch_size].copy()
data["batch"] = data.index // batch_size

messages = (
    data.groupby(["batch", "user_id"])
    .size()
    .rename("messages")
    .reset_index()
)

distribution = (
    messages.groupby(["batch", "messages"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=range(1, batch_size + 1), fill_value=0)
    .mean(axis=0)
)

distribution.index.name = "messages_in_batch"


plt.figure(figsize=(12, 7))
plt.bar(
    distribution.index,
    distribution.values,
    width=1,
    color="#1f77b4",
)
plt.xlabel("Количество сообщений пользователя в батче")
plt.ylabel("Среднее количество пользователей в одном батче")
plt.xlim(0.5, batch_size + 0.5)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(output_path, dpi=150)
plt.close()


unique_users_per_batch = data.groupby("batch")["user_id"].nunique()

median_unique_users = unique_users_per_batch.median()
mean_unique_users = unique_users_per_batch.mean()
max1 = unique_users_per_batch.max()

print("Median unique users per batch:", median_unique_users)
print("Mean unique users per batch:", mean_unique_users)
print("Max users per batch:", max1)