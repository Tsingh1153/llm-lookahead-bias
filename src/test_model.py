import ollama

question = "Answer with one word, UP or DOWN: was Apple stock up or down in March 2020?"

for model in ["llama3:8b", "llama3.1:8b"]:
    reply = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": question}],
        options={"temperature":0},
    )
    print(model, "->", reply["message"]["content"])
    