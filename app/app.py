"""TrustCart demo: run with  .venv-app/bin/streamlit run app/app.py"""
import json

import pandas as pd
import streamlit as st

from trustcart import cnn, decision, rnn
from trustcart.agents import CoordinatorAgent
from trustcart.config import DEMO_SESSIONS_PATH
from trustcart.visit import ACTION_LABELS, format_duration, summarize

st.set_page_config(page_title="TrustCart", page_icon="🛒", layout="wide")

SCENARIOS = {
    "normal": ("Client normal", "Fait ses courses, présente et paie ses articles."),
    "conceal": ("Dissimulation", "Prend un article cher, s'arrête longtemps, ne le présente pas en caisse."),
    "alibi": ("Achat alibi", "Garde un article cher et ne paie qu'un petit article pris juste avant la caisse."),
    "spree": ("Rafle", "Prend plusieurs articles en quelques secondes dans le même rayon."),
    "false_alarm": ("Fausse alerte", "Client honnête que le modèle signale à tort (cas réel du jeu de test)."),
}

CATEGORY_LABELS = {
    "alcohol": "Alcools", "candy": "Bonbons", "canned_food": "Conserves",
    "chocolate": "Chocolat", "dessert": "Desserts", "dried_food": "Épicerie sèche",
    "dried_fruit": "Fruits secs", "drink": "Boissons", "gum": "Chewing-gums",
    "instant_drink": "Boissons instantanées", "instant_noodles": "Nouilles instantanées",
    "milk": "Produits laitiers", "personal_hygiene": "Hygiène", "puffed_food": "Snacks soufflés",
    "seasoner": "Assaisonnements", "stationery": "Papeterie", "tissue": "Papier / mouchoirs",
}


@st.cache_resource
def load_models():
    """Warm the RNN and build the coordinator once per Streamlit server (cached)."""
    rnn.load_scorer()                     # warm the RNN once per server
    return CoordinatorAgent()             # orchestrates RNN, CNN and decision agents


@st.cache_data
def load_demo_sessions() -> dict:
    """The five demo sessions from the test split, indexed by scenario kind."""
    with open(DEMO_SESSIONS_PATH) as f:
        return {s["kind"]: s for s in json.load(f)}


coordinator = load_models()
sessions = load_demo_sessions()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🛒 TrustCart")
    st.caption("Caisse autonome sans friction")
    kind = st.radio("Scénario de démonstration", list(SCENARIOS), format_func=lambda k: SCENARIOS[k][0])
    st.caption(SCENARIOS[kind][1])

    st.divider()
    st.subheader("État du système")
    st.markdown("✅ **Parcours client (RNN)** : prêt")
    st.markdown("✅ **Reconnaissance produits (CNN)** : 17 catégories"
                if cnn.AVAILABLE else "⬜ **Reconnaissance produits (CNN)** : indisponible")
    st.markdown("✅ **Décision (régression logistique)** : prête"
                if decision.LOGISTIC_REGRESSION_AVAILABLE else
                "🟡 **Décision** : règle provisoire")
    st.markdown(f"🤖 **Coordinateur** : {coordinator.source}")
    st.caption(f"Mode agents : `{coordinator.mode}`")

    with st.expander("Détails des modèles"):
        rnn_rel = rnn.reliability()
        st.caption(
            f"**RNN** — rappel {rnn_rel['recall']:.0%}, fausses alertes "
            f"{rnn_rel['false_alarm_rate']:.2%} (test)."
        )
        cnn_meta = cnn.metadata()["test_metrics"]
        st.caption(
            f"**CNN** — accuracy {cnn_meta['accuracy']:.1%}, macro-F1 {cnn_meta['macro_f1']:.3f}, "
            f"ECE {cnn_meta['ece']:.3f} (test)."
        )
        if decision.LOGISTIC_REGRESSION_AVAILABLE:
            dec_meta = decision.metadata()
            st.caption(
                f"**Décision** — F1 {dec_meta['test']['f1']:.3f}, PR-AUC "
                f"{dec_meta['test']['pr_auc']:.3f}, seuil {dec_meta['threshold']:.0%} (test)."
            )
        st.caption("Les agents expliquent ; les modèles décident.")

# Results belong to one scenario: start fresh when it changes
if st.session_state.get("scenario") != kind:
    st.session_state.scenario = kind
    st.session_state.result = None
    st.session_state.chat = []

session = sessions[kind]
events = session["events"]
visit = summarize(events, rnn.cue_thresholds())

st.header(SCENARIOS[kind][0])

# ---------------------------------------------------------------- 1. path and 2. checkout
left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("1. Parcours du client")
    st.dataframe(
        pd.DataFrame([{
            "Action": ACTION_LABELS[e["action"]],
            "Article": e["itemid"],
            "Rayon": str(e["aisle"]) if e["aisle"] else "inconnu",
            "Prix (€)": e["price"],
            "Depuis l'action précédente": format_duration(e["dt_sec"]) if i else "début",
        } for i, e in enumerate(events)]),
        hide_index=True, width="stretch",
    )
    st.caption(f"Durée de la visite : {format_duration(visit.duration_min * 60)}")

with right:
    st.subheader("2. Passage en caisse")
    candidates = list(dict.fromkeys(visit.picked + visit.paid))
    presented = st.multiselect(
        "Articles posés en caisse",
        options=candidates,
        default=list(dict.fromkeys(visit.paid)),
        format_func=lambda i: visit.items[i].label,
        help="Articles déclarés par le client. Chacun est photographié : seul ce que le CNN "
             "reconnaît compte comme présenté pour la décision.",
        key=f"presented_{kind}",
    )
    camera = st.toggle("Caméra CNN active", value=cnn.AVAILABLE, disabled=not cnn.AVAILABLE,
                       help="Désactivée : saisie manuelle, le CNN n'intervient pas dans la décision.",
                       key=f"camera_{kind}")
    swaps = {}
    if camera and presented:
        st.caption("Photos prises par la caméra (catégorie du catalogue de chaque article) :")
        st.caption(" · ".join(f"{visit.items[i].item_id} → "
                              f"{CATEGORY_LABELS.get(cnn.catalogue_category(i), cnn.catalogue_category(i))}"
                              for i in presented))
        with st.expander("Simuler une substitution (fraude à l'étiquette)"):
            swapped_item = st.selectbox("Article déclaré", [None] + presented,
                                        format_func=lambda i: "aucun" if i is None else visit.items[i].label,
                                        key=f"swap_item_{kind}")
            if swapped_item is not None:
                categories = [c for c in cnn.categories() if c != cnn.catalogue_category(swapped_item)]
                swaps[swapped_item] = st.selectbox(
                    "Produit réellement posé sur le plateau", categories,
                    format_func=lambda c: CATEGORY_LABELS.get(c, c), key=f"swap_cat_{kind}")
    if st.button("Passer en caisse", type="primary", width="stretch"):
        with st.spinner("Analyse du parcours, photos CNN et décision…"):
            st.session_state.result = coordinator.run_checkout(
                events, presented, camera=camera, swaps=swaps or None)
            st.session_state.chat = []

result = st.session_state.result
if result is None:
    st.info("Choisissez les articles posés en caisse, puis cliquez sur **Passer en caisse**.")
    st.stop()

# ---------------------------------------------------------------- 3. outcome
st.divider()
st.subheader("3. Résultat")
is_alert = result.is_alert
customer_col, staff_col, model_col = st.columns(3, gap="large")

with customer_col:
    st.markdown("**Écran client**")
    box = st.error if is_alert else st.success
    box(f"### {result.customer.title}\n{result.customer.body}")

with staff_col:
    st.markdown("**Tablette du personnel**")
    with st.container(border=True):
        st.markdown(("🔴 **Vérification recommandée**" if is_alert else "🟢 **Rien à signaler**"))
        st.write(result.briefing.text)
        caption = f"Rédigé par : {result.briefing.agent_name} · {result.briefing.source}"
        if result.briefing.note:
            caption += f" ({result.briefing.note})"
        st.caption(caption)

with model_col:
    st.markdown("**Ce que les modèles ont vu**")
    st.metric("Risque du parcours (RNN)", f"{result.path.risk:.0%}",
              delta=f"seuil RNN {result.path.threshold:.0%}", delta_color="off")
    probability = result.decision.decision.probability
    threshold = result.decision.decision.threshold
    st.metric("Probabilité d'anomalie (décision)", f"{probability:.0%}" if probability is not None else "n/a",
              delta=f"seuil {threshold:.0%}" if threshold is not None else None, delta_color="off")
    if probability is not None:
        st.progress(min(probability, 1.0))
    facts = result.decision.facts
    st.markdown(f"Articles pris : **{len(facts['articles_pris'])}** · confirmés par le CNN : "
                f"**{len(facts['articles_poses_en_caisse'])}** · non présentés : "
                f"**{len(facts['articles_pris_non_presentes'])}**")
    for cue in facts["indices_dans_le_parcours"]:
        st.markdown(f"- {cue}")
    st.caption(result.decision.decision.reason)

# ---------------------------------------------------------------- per-agent trace
with st.expander("Activité des agents (traçabilité)", expanded=False):
    st.caption("Chaque agent expose uniquement les faits de son modèle. La décision finale reste "
               "celle de la régression logistique.")
    for activity in result.activities:
        with st.container(border=True):
            summary_col, metrics_col = st.columns([3, 2], gap="large")
            with summary_col:
                st.markdown(f"**{activity.agent}** — {activity.role}")
                st.write(activity.summary)
                caption = f"Source : {activity.source}"
                if activity.note:
                    caption += f" ({activity.note})"
                st.caption(caption)
            with metrics_col:
                for label, value in activity.metrics.items():
                    st.markdown(f"- **{label}** : {value}")

    st.markdown("**Signaux du modèle de décision**")
    st.dataframe(pd.DataFrame([result.decision.features]), hide_index=True, width="stretch")

    st.markdown("**Vision en caisse**")
    st.write(result.checkout.vision_status)
    view = result.checkout.view
    if view.predictions:
        verified = set(view.item_ids)
        st.dataframe(pd.DataFrame([{
            "Article déclaré": item_id,
            "Catégorie attendue": CATEGORY_LABELS.get(cnn.catalogue_category(item_id),
                                                      cnn.catalogue_category(item_id)),
            "Photo": prediction.source,
            "Catégorie reconnue (CNN)": CATEGORY_LABELS.get(prediction.label, prediction.label),
            "Confiance": f"{prediction.confidence:.1%}",
            "Confirmé": "✅" if item_id in verified else "❌",
        } for item_id, prediction in zip(view.declared_ids, view.predictions)]),
            hide_index=True, width="stretch")

# ---------------------------------------------------------------- 4. questions
st.divider()
st.subheader("4. Questions au personnel — routage multi-agents")
st.caption("Exemples : Pourquoi cette alerte ? Quels articles manquent ? Quelle est la pause la plus "
           "longue ? Le modèle est-il fiable ? Comment aborder le client ?")

for message in st.session_state.chat:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant":
            caption = f"{message.get('agent', 'Agent')} · {message.get('source', '')}"
            if message.get("note"):
                caption += f" ({message['note']})"
            st.caption(caption)

if question := st.chat_input("Posez une question sur cette visite"):
    with st.chat_message("user"):
        st.write(question)
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat]
    reply = coordinator.answer(question, result, history)
    with st.chat_message("assistant"):
        st.write(reply.text)
        caption = f"{reply.agent_name} · {reply.source}"
        if reply.note:
            caption += f" ({reply.note})"
        st.caption(caption)
    st.session_state.chat += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": reply.text, "agent": reply.agent_name,
         "source": reply.source, "note": reply.note},
    ]
