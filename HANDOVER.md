# TrustCart – Project handover (state on 2026-11-23)

> Give this whole file to an AI assistant to continue the project. Everything below is factual and tested unless marked "TODO".

## 1. The project

ING4 Data Mining, **Projet 20 : "Plateforme de Caisse Autonome sans Friction (Smart Retail)"**.
Goal: validate a customer's basket automatically at a self-checkout and detect anomalies (theft).
Deliverables: technical report, documented source code, live demo.
App name chosen: **TrustCart**.

### Architecture (decided, do not change without reason)

```
  INPUTS                 MODELS                   DECISION                 AGENTS
┌──────────────┐     ┌────────────────┐
│ Customer path│────►│ RNN (LSTM)     │──┐
│ LOOK/PICK/PAY│     │ ✅ READY       │  │     ┌──────────────────┐     ┌────────────────────┐
└──────────────┘     └────────────────┘  ├────►│ Logistic         │────►│ Agent RNN          │
┌──────────────┐     ┌────────────────┐  │     │ regression       │     │ Agent CNN          │
│ Checkout     │────►│ CNN MobileNetV2│──┤     │ ✅ READY         │     │ Agent Décision     │
│ photo/manual │     │ ✅ READY       │  │     │ → VALIDATED/ALERT│     │ Coordinateur       │
└──────────────┘     └────────────────┘  │     └──────────────────┘     │ explain, never     │
┌──────────────┐                         │      threshold = 0.5044      │ decide             │
│ Cart data    │─────────────────────────┘      (chosen on val)         └─────────┬──────────┘
└──────────────┘                                                                  ▼
                                                                     Streamlit app (✅ BUILT)
```

Key principles:
- **The 3 models decide; the 4 agents only explain.** The logistic regression produces 100 % of the
  VALIDATED/ALERT outcomes. No agent can override or re-open a decision (ethics: stopping a customer).
- **One agent per model + one coordinator.** `RNNAgent` → `PathReport`, `CNNAgent` → `CheckoutReport`,
  `DecisionAgent` → `DecisionReport`, `CoordinatorAgent` orchestrates, writes the staff briefing and
  routes Q&A. LangGraph only (no LangChain): each agent is a compiled `infer/decide -> narrate`
  StateGraph; the coordinator runs a checkout graph (RNN ∥ CNN -> Decision -> briefing) and a Q&A graph
  (LLM router with keyword fallback -> `Send` to specialists -> synthesise). LLM calls use the plain OpenAI-compatible `openai`
  client (Gemini or Ollama).
- The customer screen always uses fixed neutral wording (never accuses). Only staff messages come from
  the LLM (or the offline templates).
- If the LLM fails (no key, quota, network, unexpected error), every agent falls back to pre-written
  French templates and never crashes. `AGENT_MODE=templates` forces fully offline mode.
- User-facing language: **French**. Code and comments: English.

### Team split
- **RNN + app + agents**: done.
- **CNN** (product recognition, RPC dataset on Kaggle `diyer22/retail-product-checkout-dataset`): done
  and wired into `app/trustcart/cnn.py`.
- **Logistic regression**: done and wired into `app/trustcart/decision.py`.

## 2. Folder map

```
Projet/
├── HANDOVER.md                  ← this file
├── requirements.txt             ← app dependencies (Python 3.13)
├── .env.example                 ← copy to .env; GEMINI_API_KEY goes there (never commit .env)
├── .gitignore
├── notebooks/
│   ├── rnn.ipynb                ← full RNN pipeline, 80 cells, phases 0–14 (runs on Kaggle)
│   └── cnn.ipynb                ← CNN pipeline (RPC, MobileNetV2)
├── kaggle/rnn/                  ← copy of rnn.ipynb + kernel-metadata.json used to run it on Kaggle via CLI
├── kaggle/cnn/                  ← copy of cnn.ipynb + kernel-metadata.json
├── models/rnn/                  ← trained RNN, ready to use
│   ├── rnn_model.keras          ← Keras 3.13 / TF 2.21 model
│   ├── rnn_inference.py         ← SessionScorer: raw events → calibrated risk (score), raw_score, is_anomaly
│   ├── rnn_meta.json            ← threshold, test metrics, robustness, calibration
│   ├── encoders.json            ← preprocessing contract
│   ├── injection_config.json    ← thresholds used to simulate thefts
│   ├── rnn_scores.parquet       ← RNN score of every session + split (train/val/test) → logistic regression data
│   └── item_prices.csv          ← synthetic price per item (shared by all parts)
├── models/cnn/                  ← trained MobileNetV2 (17 retail supercategories)
│   ├── cnn_model.keras
│   ├── cnn_meta.json
│   └── sample_crops/            ← 17 verified test crops + index.json
├── models/decision/             ← exported logistic regression
│   ├── decision_meta.json       ← coefficients, scaler, threshold, val/test metrics
│   └── decision_features.parquet← audit of features/probabilities per split
├── app/
│   ├── app.py                   ← Streamlit demo (coordinator + per-agent trace)
│   ├── data/demo_sessions.json  ← 5 real test sessions (normal, conceal, alibi, spree, false_alarm)
│   └── trustcart/
│       ├── config.py            ← paths, agent backend/mode, per-agent model overrides
│       ├── rnn.py               ← loads models/rnn, path_risk(), alert_threshold(), reliability()
│       ├── cnn.py               ← CNN inference (raw [0,255] RGB; preprocessing is in-graph)
│       ├── decision.py          ← deterministic logistic regression (VALIDATED/ALERT)
│       ├── visit.py             ← events → readable facts, theft cues, timeline (French)
│       ├── agent.py             ← legacy single agent (compatibility shim)
│       └── agents/              ← multi-agent layer
│           ├── base.py          ← shared contracts + zero-crash LLM fallback
│           ├── rnn_agent.py     ← PathReport
│           ├── cnn_agent.py     ← CheckoutReport
│           ├── decision_agent.py← DecisionReport
│           ├── coordinator.py   ← orchestration, briefing, Q&A routing
│           └── facts.py         ← allowed facts + fixed customer message
├── scripts/
│   ├── train_decision.py        ← zero-leakage logistic-regression training/export
│   └── verify_agents.py         ← offline checks: scenarios, routing, fallbacks, CNN
├── outputs/rnn_v5/              ← last Kaggle run outputs (checkpoints, big files, gitignored)
├── context/                     ← README.md, models/ (modele_*.txt), use-case/ (scenario_*.txt): French explanations
├── .venv/                       ← Python 3.14 env with kaggle CLI + uv (tools only)
├── .venv-app/                   ← Python 3.13 env for the app (TensorFlow needs ≤ 3.13)
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

## 5. Status and next steps

Completed:
1. **CNN**: `app/trustcart/cnn.py` loads `models/cnn/cnn_model.keras`, classifies the 17 RPC
   supercategories and returns a `CheckoutView` with mean confidence. It feeds raw RGB `[0, 255]`
   because the exported Keras graph already contains `preprocess_input`. v2 (kaggle/cnn/cnn.ipynb,
   kernel version 5): train/val/test accuracy 0.9946/0.9947/0.9942, temperature scaling baked in.
2. **Logistic regression**: `scripts/train_decision.py` derives four features (rnn_risk, items_picked_not_presented, cart_total, visit_duration_min; no CNN-confidence proxy) from
   `models/rnn/rnn_scores.parquet` + `outputs/rnn_v5/sessions_labeled.parquet`, fits on `train`,
   selects the threshold on `val` (best F1), evaluates once on `test`, and exports
   `models/decision/decision_meta.json`. `app/trustcart/decision.py` applies it exactly.
3. **Multi-agent layer**: `app/trustcart/agents/` contains `RNNAgent`, `CNNAgent`, `DecisionAgent`
   and `CoordinatorAgent`, with parallel orchestration, per-agent reports, staff briefing, Q&A routing
   and offline French fallbacks. `app/app.py` displays the per-agent trace.

Verification:
```bash
.venv-app/bin/python scripts/verify_agents.py   # offline: scenarios, routing, fallbacks, CNN, LR
.venv-app/bin/streamlit run app/app.py
```

Next steps (optional):
1. **Add the Gemini key** in `.env` and test the live narration (`AGENT_MODE=hybrid`).
2. **RNN improvements** (→ 9/10): check the "few LOOKs" hypothesis; add an LSTM autoencoder trained on
   normal sessions to catch unseen theft types.
3. **CNN improvements**: the test accuracy is 59.1 % on 17 fine-grained retail categories; more crops
   per class or a stronger backbone would help. The demo intentionally presents the CNN as a fallible
   optical sensor, not a decision-maker.
4. **Report**: use the phase table above, the RNN/CNN/LR metrics, and the multi-agent architecture diagram.

## 6. How the user likes to work
- Explain step by step, phase by phase; they want to learn, not only get code.
- Number notebook cells counting every cell from the top (markdown included).
- Kaggle sessions restart often: cells must reload checkpoints; give explicit run order.
- Senior standards: tests/asserts, no leakage, honest limitations.
- Ask before big scope changes; the user decides (e.g. no CNN by the AI for now).
