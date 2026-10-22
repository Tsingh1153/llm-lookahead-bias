import sys
import pandas as pd
import ollama

MODELS = ["llama3:8b", "llama3.1:8b"]

df = pd.read_csv("data/ground_truth.csv")

#test
if "--test" in sys.argv:
    df = df.sample(40, random_state=42)

def ask(model, ticker, month):
    date = pd.to_datetime(month).strftime("%B %Y")
    prompt = f"Was {ticker} stock up or down in {date}? Answer with one word: UP or DOWN."
    reply = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0, "num_predict":5},
    )
    text = reply["message"]["content"].strip().upper()
    if text.startswith("UP"):
        return "UP"
    if text.startswith("DOWN"):
        return "DOWN"
    return "UNCLEAR"

for model in MODELS:
    answers = []
    for i, row in enumerate(df.itertuples()):
        answers.append(ask(model, row.ticker, row.month))
        if i % 100 == 0:
            print(model, i, "/", len(df))
    df[model] = answers

df.to_csv("results/predictions.csv", index=False)
print(df.head(10))