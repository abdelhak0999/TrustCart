TrustCart is a frictionless self-checkout system that follows each shopper from the moment they enter the store to the moment they leave. Three models each handle one phase of the visit:

▎ - an RNN (LSTM) analyzes the sequence of actions in the aisles and gives the path an anomaly score,

▎ - a CNN recognizes the products placed at the checkout, with no manual scanning needed,

▎ - a logistic regression combines both signals to approve the payment in seconds, or discreetly call a security agent when something doesn't add up.

▎

▎ Honest customers pay and leave in about 10 seconds, and suspicious cases are flagged with an explainable decision.

## Énoncé (Projet 20 : Plateforme de Caisse Autonome sans Friction — Smart Retail)

- **Dataset :** Retail Product Recognition + Customer Shopping Sequence (Kaggle).
- **Objectif :** valider automatiquement le panier client et détecter les anomalies de passage en caisse.
- **Techniques :** CNN (reconnaissance optique des articles déposés), RNN (suivi de la cohérence séquentielle du parcours client), Régression Logistique (validation du paiement ou appel d'un agent de sécurité).
- **Livrables :** rapport technique, code source documenté, démonstration.
