import pandas as pd
import matplotlib.pyplot as plt

CUTOFFS = {"llama3:8b": "2023-03", "llama3.1:8b": "2023-12"}

df = pd.read_csv("results/predictions.csv")
df["year"] = df["month"].str[:4].astype(int)

def informedness(g, model):
    up_true = (g.loc[g["actual"] == "UP", model] == "UP").mean()
    up_false = (g.loc[g["actual"] == "DOWN", model] == "UP").mean()
    return up_true - up_false

def informedness_se(g, model):
    up = g.loc[g["actual"] == "UP", model] == "UP"
    down = g.loc[g["actual"] == "DOWN", model] == "UP"
    return (up.mean() * (1 - up.mean()) / len(up)
            + down.mean() * (1 - down.mean()) / len(down)) ** 0.5


rows = []
for model, cutoff in CUTOFFS.items():
    period = df["month"].le(cutoff).map({True: "before", False: "after"})
    for name, g in df.groupby(period):
        rows.append({
            "model": model,
            "period": name,
            "n": len(g),
            "said_up": (g[model] == "UP").mean(),
            "accuracy": (g[model] == g["actual"]).mean(),
            "informedness": informedness(g, model),
            "se": informedness_se(g, model),
        })

summary = pd.DataFrame(rows)
print(summary.round(3))
summary.to_csv("results/summary.csv", index=False)

# Is the before/after gap bigger than noise?
for model in CUTOFFS:
    s = summary[summary["model"] == model].set_index("period")
    gap = s.loc["before", "informedness"] - s.loc["after", "informedness"]
    gap_se = (s.loc["before", "se"] ** 2 + s.loc["after", "se"] ** 2) ** 0.5
    print(f"{model}: gap = {gap:.3f}, z = {gap / gap_se:.1f}")


yearly_table = pd.DataFrame({m: df.groupby("year").apply(lambda g: informedness(g, m))
                             for m in CUTOFFS})
print("\nInformedness by year:")
print(yearly_table.round(3))


by_year = df.groupby("year")["actual"].apply(lambda s: (s == "UP").mean()).to_frame("actual_up")
for m in CUTOFFS:
    by_year[m + "_said_up"] = df.groupby("year")[m].apply(lambda s: (s == "UP").mean())
print("\nShare UP by year (actual vs. said):")
print(by_year.round(3))
by_year.to_csv("results/by_year.csv")


fig, ax = plt.subplots(figsize=(8, 5))
for model in CUTOFFS:
    yearly = df.groupby("year").apply(lambda g: informedness(g, model))
    ax.plot(yearly.index, yearly.values, marker="o", label=model)
ax.axhline(0, color="gray", ls="--", label="no knowledge")
ax.axvline(2023.2, ls=":", color="C0")   # llama3 cutoff (Mar 2023)
ax.axvline(2023.9, ls=":", color="C1")   # llama3.1 cutoff (Dec 2023)
ax.set_xticks(range(2021, 2026))
ax.set_xlabel("Year")
ax.set_ylabel("Informedness")
ax.set_title("Does the model know what actually happened? (by year)")
ax.legend()
plt.savefig("results/accuracy_by_year.png", dpi=150, bbox_inches="tight")
print("\nChart saved")