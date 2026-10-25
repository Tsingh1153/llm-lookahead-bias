# Lab Notes — LLM Look-Ahead Bias

## Setup
- Python 3.15, VS Code, virtual environment (.venv)
- Models run locally with Ollama:
  - llama3:8b — training cutoff March 2023
  - llama3.1:8b — training cutoff December 2023
- Settings: temperature 0 (deterministic), num_predict 5

## 2026-10-20 — Sanity check
- Asked both models: "Was Apple stock up or down in March 2020?"
- Both answered DOWN (correct, AAPL fell ~7% in the COVID crash).
- Note: a single famous event proves nothing; need a large sample.

## 2026-10-20 — Ground truth (Phase 2)
- 30 large-cap stocks, Jan 2021 – Dec 2025, monthly returns via yfinance
- Prices adjusted for splits/dividends (auto_adjust=True)
- 1,800 rows (60 months x 30 stocks), no missing data
- UP share = 0.544 → "always UP" baseline is 54.4%, not 50%
- Decision: months with exactly 0 return labeled DOWN (check how many occur)
- Limitation: survivorship bias — all stocks are still large today,
  likely inflating the UP share. Doesn't affect before/after-cutoff comparison.

## 2026-10-21 — Query test 1 (Phase 3)
- Ran ask_models.py --test on first 20 rows
- Code works; no UNCLEAR answers
- Problem: first 20 rows are all January 2021 — only one month tested
- Observation: both models answered UP on all 10 printed rows,
  including 5 stocks that actually fell → strong UP bias
- Possible cause: position bias — prompt lists "UP" first
- Decision: switch test to random sample of 40 (random_state=42);
  count UP/DOWN per model before running the full set
- Idea for later: swap word order in prompt ("DOWN or UP") as a robustness check

## 2026-10-21 — Query test 2 (Phase 3)
- Random sample of 40 rows (random_state=42)
- New finding: UNCLEAR answers cluster on recent dates
  - Both models: UNCLEAR on 2024–2025 dates
  - llama3.1 also UNCLEAR on several 2023 dates where llama3 still answered UP
  - 2021 dates answered normally
- Problem: refusals cluster after the cutoff → confound.
  Counting UNCLEAR as wrong would fake a look-ahead effect;
  dropping them would remove and skew post-cutoff data.
- Decision: prompt now asks for a best guess; save raw replies;
  track refusal rate separately as its own result.
  
  ## 2026-10-22 — Raw replies for UNCLEAR
- They are true refusals, not parsing failures:
  - llama3:8b: "I'm a large language model, I don't..."
  - llama3.1:8b: "I'm unsure."
- "Best guess" wording not enough to prevent refusals
- Decision: use Ollama structured outputs (JSON schema, enum UP/DOWN)
  → constrained decoding makes refusal impossible; model still chooses the answer
- Trade-off: lose refusal signal in main run → measure refusal rate
  separately as a side result