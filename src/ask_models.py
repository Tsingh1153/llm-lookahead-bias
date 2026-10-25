import sys
import time
import pandas as pd
import ollama
import json

MODELS = ["llama3:8b", "llama3.1:8b"]
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": ["UP", "DOWN"]}},
    "required": ["answer"]
}

SYSTEM = ("You are a financial historian. You answer questions about "
          "past stock market performance from your knowledge.")

TEST = "--test" in sys.argv

df = pd.read_csv("data/ground_truth.csv")

#test
if TEST:
    df = df.sample(40, random_state=42)

def ask(model, ticker, month):
    date = pd.to_datetime(month).strftime("%B %Y")
    prompt = (f"Was {ticker} stock up or down in {date}? Respond in JSON.")
    reply = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt}],
        format=SCHEMA,
        options={"temperature": 0, "num_predict":20},
    )
    raw = reply["message"]["content"]
    try:
        return json.loads(raw)["answer"],raw
    except (json.JSONDecodeError, KeyError):
        return "UNCLEAR", raw
        
start = time.time()
for model in MODELS:
    labels, raws = [], []
    for i, row in enumerate(df.itertuples()):
        label, raw = ask(model, row.ticker, row.month)
        labels.append(label)
        raws.append(raw)
        if i % 100 == 0:
            print(model, i, "/", len(df))
    df[model] = labels
    df[model + "_raw"] = raws

out = "results/test_predictions.csv" if TEST else "results/predictions.csv"
df.to_csv(out, index=False)

print(df[MODELS].apply(lambda col: col.value_counts()))
print(f"Took {time.time() - start:.0f} seconds")