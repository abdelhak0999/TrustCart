# 🛒 TrustCart – Plateforme de Caisse Autonome sans Friction

> **ING4 Data Mining – Projet 20 : Smart Retail**  
> Système de validation automatisée de paniers d'achat en caisse autonome et détection d'anomalies (dissimulation, rafle, vol).

---

## 1. Vue d'ensemble de l'architecture

Le système repose sur un découpage strict en trois couches : **Perception & Modèles**, **Décision Déterministe**, et **Explication par Agent LLM**.

```
  ENTRÉES                  MODÈLES                  DÉCISION                  AGENT EXPLICATIF
┌──────────────┐     ┌────────────────┐
│ Parcours     │────►│ RNN (LSTM)     │──┐
│ LOOK/PICK/PAY│     │ ✅ DISPONIBLE  │  │     ┌──────────────────┐     ┌───────────────────────┐
└──────────────┘     └────────────────┘  ├────►│ Régression       │────►│ Agent d'explication   │
┌──────────────┐     ┌────────────────┐  │     │ logistique       │     │ (Gemini / Ollama)     │
│ Photo de la  │────►│ CNN (vision)   │──┤     │ ⬜ EN COURS      │     │ ✅ DISPONIBLE         │
│ caisse       │     │ ⬜ EN COURS    │  │     │ → VALIDE / ALERTE│     │ Rédige les synthèses, │
└──────────────┘     └────────────────┘  │     └──────────────────┘     │ répond aux questions, │
┌──────────────┐                         │      🟡 Règle transitoire:   │ NE DÉCIDE JAMAIS      │
│ Données du   │─────────────────────────┘         Alerte si RNN ≥ 0.48 └──────────┬────────────┘
│ panier       │                                                                   ▼
└──────────────┘                                                    Application Streamlit (✅ PRÊTE)
```

### Principes éthiques et techniques fondamentaux :
- **Les modèles statistiques décident, le LLM explique** : L'agent d'IA générative ne prend ni ne modifie jamais une décision d'alerte (sécurité et éthique d'interpellation client).
- **Communication bilingue/séparée** : L'écran client affiche un message strictement neutre et factuel. Seul le personnel de sécurité reçoit les briefings explicatifs détaillés.
- **Résilience (Zero Crash)** : Si l'API Gemini est inaccessible (quota, réseau, absence de clé), l'application bascule automatiquement et de manière transparente sur des gabarits hors ligne pré-écrits.

---

## 2. Installation & Démarrage rapide

### Prérequis
- Linux ou macOS (ou WSL2 sur Windows)
- **Python 3.13** (requis pour la compatibilité TensorFlow / Keras 3.13)
- Gestionnaire de paquets `uv` (recommandé) ou `venv` standard

### 1. Cloner le projet et créer l'environnement

```bash
# Cloner le dépôt
git clone <URL_DU_REPO>
cd Projet

# Créer l'environnement virtuel avec Python 3.13
uv venv --python 3.13 .venv-app
# OU avec python standard : python3.13 -m venv .venv-app

# Installer les dépendances
.venv-app/bin/pip install -r requirements.txt
```

### 2. Configuration des secrets (`.env`)

Copiez le modèle d'environnement :
```bash
cp .env.example .env
```

Éditez le fichier `.env` pour y renseigner votre clé gratuite Google AI Studio :
```ini
GEMINI_API_KEY=AIzaSy...votre_cle_ici
GEMINI_MODEL=gemini-3.8-flash
AGENT_BACKEND=gemini
```
*(Note : Si vous ne disposez pas de clé API, laissez vide. L'application utilisera automatiquement les templates hors ligne).*

### 3. Lancer la démonstration Streamlit

```bash
.venv-app/bin/streamlit run app/app.py
```
L'interface est accessible sur [http://localhost:8501](http://localhost:8501).

---

## 3. Guide d'intégration pour l'équipe (Où brancher vos modules)

### A. Reconnaissance Produits (CNN / Vision)
- **Dataset** : Kaggle `diyer22/retail-product-checkout-dataset` (RPC).
- **Fichier à implémenter** : `app/trustcart/cnn.py`
  - Compléter la fonction `items_seen(manual_item_ids)` pour appeler votre modèle CNN entraîné.
  - Retourner une instance de `CheckoutView(items=[...], confidence=0.95, source="cnn")`.
  - Passer le flag `AVAILABLE = True` dans `cnn.py`.
  - Prévoir une table de correspondance entre les classes RPC et les identifiants d'articles du projet (`item_prices.csv`).

### B. Fusion & Décision (Régression Logistique)
- **Données d'entraînement** : `models/rnn/rnn_scores.parquet` contient pour chaque session réelle et simulée son score RNN calibré (`rnn_score`) et son split officiel (`train`, `val`, `test`).
- **Features définies** (voir `app/trustcart/decision.py`) :
  1. `rnn_risk` : Score calibré de probabilité d'anomalie [0, 1].
  2. `items_picked_not_presented` : Nombre d'articles pris en rayon mais absents de la caisse.
  3. `cnn_confidence` : Score de confiance du modèle de vision [0, 1].
  4. `cart_total` : Valeur marchande totale des articles présentés.
  5. `visit_duration_min` : Durée totale de la visite en magasin.
- **Fichier à implémenter** : Remplacer la règle provisoire dans `app/trustcart/decision.py::_logistic_regression()` et activer `LOGISTIC_REGRESSION_AVAILABLE = True`.

---

## 4. Organisation du projet

```
Projet/
├── README.md                  ← Ce document
├── HANDOVER.md                ← Carnet d'état complet, métriques et historique
├── requirements.txt           ← Dépendances de l'application (Python 3.13)
├── .env.example               ← Modèle des variables d'environnement
├── .gitignore                 ← Configuration Git (exclut les secrets et données lourdes)
│
├── app/                       ← Application Streamlit et cœur TrustCart
│   ├── app.py                 ← Point d'entrée de la démo
│   ├── data/demo_sessions.json← 5 sessions réelles du jeu de test (normal, vol, fausse alerte)
│   └── trustcart/             ← Logique métier et interfaces des modèles
│       ├── config.py          ← Gestion de la configuration et des backends
│       ├── rnn.py             ← Inférence du parcours client (LSTM)
│       ├── cnn.py             ← Interface du modèle de vision (stand-in manuel)
│       ├── decision.py        ← Moteur décisionnel (règle actuelle / future régression)
│       ├── visit.py           ← Traduction des événements en indices factuels (français)
│       └── agent.py           ← Agent explicatif (Gemini, fallback templates)
│
├── models/                    ← Artefacts de modèles
│   └── rnn/                   ← Modèle LSTM entraîné et métadonnées
│       ├── rnn_model.keras    ← Modèle Keras 3.13
│       ├── rnn_meta.json      ← Métriques, seuil d'alerte (0.476), calibration Platt
│       ├── rnn_scores.parquet ← Base pour l'entraînement de la régression logistique
│       └── item_prices.csv    ← Grille tarifaire unifiée par article
│
├── notebooks/                 ← Expérimentations Jupyter / Kaggle
│   ├── rnn.ipynb              ← Pipeline complet du RNN (80 cellules, phases 0 à 14)
│   └── cnn.ipynb              ← Notebook réservé au modèle de vision
│
└── context/                   ← Documentation métier et fiches de présentation
```

---

## 5. Performances et métriques actuelles (Modèle RNN)

- **Architecture** : Embeddings + LSTM (64 unités) avec masquage Keras 3 sans fuite temporelle.
- **Calibration** : Régression de Platt (ECE ramenée de 0.044 à 0.004).
- **Seuil d'alerte** : `0.476` sur le score calibré.
- **Performances sur le jeu de test** :
  - **PR-AUC** : `0.966` (validation croisée robuste sur 4 seeds : `0.967 ± 0.003`)
  - **ROC-AUC** : `0.996`
  - **Précision** : `91.7 %`
  - **Rappel** : `87.2 %`
  - **Taux de fausses alertes** : `0.74 % ± 0.19`
