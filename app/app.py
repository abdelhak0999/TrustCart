"""TrustCart demo: run with  .venv-app/bin/streamlit run app/app.py"""
import json

import pandas as pd
import streamlit as st

from trustcart import cnn, decision, rnn
from trustcart.agent import ExplanationAgent, build_facts, customer_message
from trustcart.config import DEMO_SESSIONS_PATH, agent_settings
from trustcart.visit import ACTION_LABELS, format_duration, summarize

st.set_page_config(page_title="TrustCart", page_icon="🛒", layout="wide")

SCENARIOS = {
    "normal": ("Client normal", "Fait ses courses, présente et paie ses articles."),
    "conceal": ("Dissimulation", "Prend un article cher, s'arrête longtemps, ne le présente pas en caisse."),
    "alibi": ("Achat alibi", "Garde un article cher et ne paie qu'un petit article pris juste avant la caisse."),
    "spree": ("Rafle", "Prend plusieurs articles en quelques secondes dans le même rayon."),
    "false_alarm": ("Fausse alerte", "Client honnête que le modèle signale à tort (cas réel du jeu de test)."),
}


@st.cache_resource
def load_models():
    rnn.load_scorer()                     # warm the model up once per server
    return ExplanationAgent(agent_settings())


@st.cache_data
def load_demo_sessions() -> dict:
    with open(DEMO_SESSIONS_PATH) as f:
        return {s["kind"]: s for s in json.load(f)}


agent = load_models()
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
    st.markdown("⬜ **Reconnaissance produits (CNN)** : pas encore entraîné, saisie manuelle")
    st.markdown("🟡 **Décision (régression logistique)** : pas encore entraînée, règle provisoire")
    st.markdown(f"🤖 **Assistant** : {agent.source}")

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
        help="La reconnaissance des produits (CNN) n'est pas encore disponible : les articles posés "
             "sont saisis ici. Par défaut, ce sont les articles payés dans le parcours.",
        key=f"presented_{kind}",
    )
    if st.button("Passer en caisse", type="primary", width="stretch"):
        with st.spinner("Analyse du parcours…"):
            checkout = cnn.items_seen(presented)
            risk, threshold = rnn.path_risk(events), rnn.alert_threshold()
            final = decision.decide(risk, threshold, visit, checkout)
            facts = build_facts(final, risk, threshold, visit, checkout, events)
            st.session_state.result = {
                "risk": risk, "threshold": threshold, "decision": final, "facts": facts,
                "customer": customer_message(final, facts), "briefing": agent.staff_briefing(facts),
            }
            st.session_state.chat = []

result = st.session_state.result
if result is None:
    st.info("Choisissez les articles posés en caisse, puis cliquez sur **Passer en caisse**.")
    st.stop()

# ---------------------------------------------------------------- 3. outcome
st.divider()
st.subheader("3. Résultat")
is_alert = result["decision"].outcome == decision.ALERT
customer_col, staff_col, model_col = st.columns(3, gap="large")

with customer_col:
    st.markdown("**Écran client**")
    box = st.error if is_alert else st.success
    box(f"### {result['customer'].title}\n{result['customer'].body}")

with staff_col:
    st.markdown("**Tablette du personnel**")
    with st.container(border=True):
        st.markdown(("🔴 **Vérification recommandée**" if is_alert else "🟢 **Rien à signaler**"))
        st.write(result["briefing"].text)
        caption = f"Rédigé par : {result['briefing'].source}"
        if result["briefing"].note:
            caption += f" ({result['briefing'].note})"
        st.caption(caption)

with model_col:
    st.markdown("**Ce que les modèles ont vu**")
    st.metric("Risque du parcours (RNN)", f"{result['risk']:.0%}",
              delta=f"seuil d'alerte {result['threshold']:.0%}", delta_color="off")
    st.progress(min(result["risk"], 1.0))
    facts = result["facts"]
    st.markdown(f"Articles pris : **{len(facts['articles_pris'])}** · posés en caisse : "
                f"**{len(facts['articles_poses_en_caisse'])}**")
    for cue in facts["indices_dans_le_parcours"]:
        st.markdown(f"- {cue}")
    st.caption(result["decision"].reason + " Décision : " + facts["qui_decide"] + ".")

# ---------------------------------------------------------------- 4. questions
st.divider()
st.subheader("4. Questions à l'assistant")
st.caption("Exemples : Pourquoi cette alerte ? Quels articles manquent ? Comment aborder le client ? Le modèle est-il fiable ?")

for message in st.session_state.chat:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if question := st.chat_input("Posez une question sur cette visite"):
    with st.chat_message("user"):
        st.write(question)
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat]
    reply = agent.answer(question, result["facts"], history)
    with st.chat_message("assistant"):
        st.write(reply.text)
        st.caption(reply.source + (f" ({reply.note})" if reply.note else ""))
    st.session_state.chat += [{"role": "user", "content": question},
                              {"role": "assistant", "content": reply.text}]
