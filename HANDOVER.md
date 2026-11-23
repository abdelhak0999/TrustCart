# TrustCart – Project handover (state on 2026-11-23)

> Give this whole file to an AI assistant to continue the project. Everything below is factual and tested unless marked "TODO".

## 1. The project

ING4 Data Mining, **Projet 20 : "Plateforme de Caisse Autonome sans Friction (Smart Retail)"**.
Goal: validate a customer's basket automatically at a self-checkout and detect anomalies (theft).
Deliverables: technical report, documented source code, live demo.
App name chosen: **TrustCart**.

### Architecture (decided, do not change without reason)

```
  INPUTS                 MODELS                   DECISION                 AGENT
┌──────────────┐     ┌────────────────┐
│ Customer path│────►│ RNN (LSTM)     │──┐
│ LOOK/PICK/PAY│     │ ✅ READY       │  │     ┌──────────────────┐     ┌────────────────────┐
└──────────────┘     └────────────────┘  ├────►│ Logistic         │────►│ Explanation agent  │
┌──────────────┐     ┌────────────────┐  │     │ regression       │     │ (1 LLM: Gemini)    │
│ Checkout     │────►│ CNN            │──┤     │ ⬜ NOT STARTED   │     │ ✅ BUILT           │
│ photo        │     │ ⬜ NOT STARTED │  │     │ → VALIDATED/ALERT│     │ writes messages,   │
└──────────────┘     └────────────────┘  │     └──────────────────┘     │ answers questions, │
┌──────────────┐                         │      🟡 for now: ALERT if    │ NEVER decides      │
│ Cart data    │─────────────────────────┘         RNN risk ≥ 0.476     └─────────┬──────────┘
└──────────────┘                                                                  ▼
                                                                    Streamlit app (✅ BUILT)
```

Key principles:
- **The 3 models decide; the LLM agent only explains.** It must never change a decision (ethics: stopping a customer).
- One agent, not one per model. No LangChain/LangGraph: the OpenAI-compatible client talks to Gemini (free key) or Ollama.
- The customer screen always uses fixed neutral wording (never accuses). Only staff messages come from the LLM.
- If the LLM fails (no key, quota, network), the app falls back to pre-written French templates and never crashes.
- User-facing language: **French**. Code and comments: English.

### Team split
- **RNN + app + agent**: done (this handover).
- **CNN** (product recognition, RPC dataset on Kaggle `diyer22/retail-product-checkout-dataset`): a teammate, not started. The user does NOT want the AI to do the CNN for now.
- **Logistic regression**: not started (needs RNN + CNN outputs).

## 2. Folder map

```
Projet/
├── HANDOVER.md                  ← this file
├── requirements.txt             ← app dependencies (Python 3.13)
├── .env.example                 ← copy to .env; GEMINI_API_KEY goes there (never commit .env)
├── .gitignore
├── notebooks/
│   ├── rnn.ipynb                ← full RNN pipeline, 80 cells, phases 0–14 (runs on Kaggle)
│   └── cnn.ipynb                ← empty, teammate's
├── kaggle/rnn/                  ← copy of rnn.ipynb + kernel-metadata.json used to run it on Kaggle via CLI
├── models/rnn/                  ← trained RNN, ready to use
│   ├── rnn_model.keras          ← Keras 3.13 / TF 2.21 model
│   ├── rnn_inference.py         ← SessionScorer: raw events → calibrated risk (score), raw_score, is_anomaly
│   ├── rnn_meta.json            ← threshold, test metrics, robustness, calibration
│   ├── encoders.json            ← preprocessing contract
│   ├── injection_config.json    ← thresholds used to simulate thefts
│   ├── rnn_scores.parquet       ← RNN score of every session + split (train/val/test) → logistic regression data
│   └── item_prices.csv          ← synthetic price per item (shared by all parts)
├── app/
│   ├── app.py                   ← Streamlit demo
│   ├── data/demo_sessions.json  ← 5 real test sessions (normal, conceal, alibi, spree, false_alarm)
│   └── trustcart/
│       ├── config.py            ← paths + agent backend selection from .env
│       ├── rnn.py               ← loads models/rnn, path_risk(), alert_threshold(), reliability()
│       ├── cnn.py               ← STAND-IN: manual item entry. Teammate replaces items_seen()
│       ├── decision.py          ← STAND-IN rule. Teammate implements _logistic_regression()
│       ├── visit.py             ← events → readable facts, theft cues, timeline (French)
│       └── agent.py             ← ExplanationAgent: Gemini/Ollama + template fallback
├── outputs/rnn_v5/              ← last Kaggle run outputs (checkpoints, big files, gitignored)
├── context/                     ← README.md, models/ (modele_*.txt), use-case/ (scenario_*.txt): French explanations
├── .venv/                       ← Python 3.14 env with kaggle CLI + uv (tools only)
└── .venv-app/                   ← Python 3.13 env for the app (TensorFlow needs ≤ 3.13)
```

## 3. How to run things

```bash
cd ~/Desktop/ING4/DM/Projet

# The demo app (http://localhost:8501)
.venv-app/bin/streamlit run app/app.py

# Recreate the app env if needed
.venv/bin/uv venv --python 3.13 .venv-app
.venv/bin/uv pip install --python .venv-app/bin/python -r requirements.txt

# Re-run the RNN notebook on Kaggle (private kernel hamziabdelhak/trustcart-rnn-pipeline, CPU)
cp notebooks/rnn.ipynb kaggle/rnn/rnn.ipynb        # (strip outputs first if large)
.venv/bin/kaggle kernels push -p kaggle/rnn
.venv/bin/kaggle kernels status hamziabdelhak/trustcart-rnn-pipeline
.venv/bin/kaggle kernels output hamziabdelhak/trustcart-rnn-pipeline -p outputs/rnn_vN
```

Kaggle token is in `~/.kaggle/access_token` (private). **It was pasted in a chat once: revoke it (Kaggle → Settings → API) and create a new one.**
Kaggle gotcha: GPU batch runs stay QUEUED while the user's interactive GPU session is open. The RNN runs fine on CPU (~20 min with robustness phase).

Gemini: free key from aistudio.google.com → put `GEMINI_API_KEY=...` in `.env`. Endpoint used: `https://generativelanguage.googleapis.com/v1beta/openai/`, model `gemini-3.8-flash` (change with `GEMINI_MODEL`). `gemini-2.5-*` are deprecated. Gemini returns HTTP 400 (not 401) for a bad key; agent.py handles it.

## 4. The RNN – what was done (notebooks/rnn.ipynb)

Data: **Retailrocket** e-commerce events (Kaggle `retailrocket/ecommerce-dataset`), adapted to an in-store story:
view → LOOK, addtocart → PICK, transaction → PAY, item root category → aisle (23 aisles + 0 unknown), synthetic prices per item (log-normal per aisle tier).

| Phase | Content | Key result |
|---|---|---|
| 1 | EDA | 2.76 M events, 97 % views, 137 days |
| 2 | Clean | 460 duplicates; bots = top 0.1 % activity AND no purchase intent (532 removed; first version wrongly removed 819 real shoppers) |
| 3 | Aisles via category tree | 90.6 % coverage, 23 active aisles; 5.6 % items changed category (kept latest) |
| 4 | Sessions (30 min gap) | 30,198 sessions kept (≥1 PICK, ≥3 events), cut to last 50 events |
| 5 | Store mapping + profile | dt p50 70 s, p99 1450 s; expensive > 82.6 €, cheap ≤ 9.8 € |
| 6 | Inject thefts (copies of real sessions, same length) | 3,020 anomalies (9.1 %): conceal, alibi, spree. Bugs fixed: alibi length, theft inserted inside a checkout |
| 7 | Encoding | action/aisle ids + log1p(dt), log1p(price), right padding, encode_session() reference |
| 8 | Split | 70/15/15 grouped by orig_id (a session and its theft copy never split), stratified |
| 9 | Model | Embeddings + LSTM(64, return_sequences) + state at last real step (Dot with one-hot last_step input) + Dense. Keras 3 Concatenate merges masks with OR, so mask_zero leaked padding; fixed by design, padding test diff = 0 |
| 10 | Training | class weights, early stopping on val PR-AUC, best epoch 11 |
| 11 | Evaluation (test) | ROC-AUC 0.996, **PR-AUC 0.966**, precision 0.917, recall 0.872, false alarms 0.79 %. Baselines: length-only ROC 0.63, session aggregates PR-AUC 0.761 → order matters |
| 12 | Error analysis | missed thefts mostly "conceal" |
| 13 | Export | model, meta, scorer; exported scorer == notebook (diff 1.8e-7) |
| 14 | Robustness | 4 seeds: PR-AUC **0.967 ± 0.003**, false alarms 0.74 % ± 0.19. Leave-one-pattern-out recall: alibi 85 %, conceal 14 %, spree 19 % (weak on unseen theft types). 25 % of false alarms carry a theft cue vs 1 % of accepted. Platt calibration: ECE 0.044 → 0.004, threshold 0.476 on calibrated score |

Senior-standard grade given: **8.5/10**. Main limits: thefts are simulated; model recognises known patterns, not new ones.

Open hypothesis (unverified): 75 % of false alarms have no theft cue; maybe because theft copies had LOOK events removed to keep lengths equal, so "few LOOKs" may look suspicious.

## 5. TODO (in suggested order)

1. **Add the Gemini key** in `.env`, test the assistant live in the app.
2. **README.md** for teammates (how to run, where to plug models) + **git init / first GitHub commit**. Note: `.gitignore` ignores `models/`, `*.parquet`, `*.csv`; the app needs `models/rnn/` → allow it (e.g. replace `models/` by `models/*` then `!models/rnn/`, and allow `models/rnn/*.parquet|*.csv`) or share models separately.
3. **CNN (teammate)**: implement `app/trustcart/cnn.py::items_seen()` returning `CheckoutView(item_ids, confidence, "cnn")`; set `AVAILABLE = True`. RPC images contain product classes, not Retailrocket item ids → a mapping table RPC class ↔ items (or ↔ aisle/price) is needed.
4. **Logistic regression**: features defined in `decision.py::features()` (rnn_risk, items_picked_not_presented, cnn_confidence, cart_total, visit_duration_min). Train on `models/rnn/rnn_scores.parquet` (use calibrated `rnn_score`, respect the existing train/val/test split) + simulated CNN outputs; implement `_logistic_regression()`; set `LOGISTIC_REGRESSION_AVAILABLE = True`.
5. Optional RNN improvements (→ 9/10): check the "few LOOKs" hypothesis; add an LSTM autoencoder trained on normal sessions to catch unseen theft types.
6. Report: use the phase table above, the robustness results, and the architecture diagram.

## 6. How the user likes to work
- Explain step by step, phase by phase; they want to learn, not only get code.
- Number notebook cells counting every cell from the top (markdown included).
- Kaggle sessions restart often: cells must reload checkpoints; give explicit run order.
- Senior standards: tests/asserts, no leakage, honest limitations.
- Ask before big scope changes; the user decides (e.g. no CNN by the AI for now).
