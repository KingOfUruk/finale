# Pipelines ML / IA

Ce dossier décrit comment entraîner et consommer les modèles ajoutés à l'application.

## Installation des dépendances

Les scripts utilisent principalement `pandas`, `numpy`, `scikit-learn`, `joblib` (et `shap` en option).  
Assurez-vous que l'environnement virtuel contient au minimum :

```bash
pip install pandas numpy scikit-learn joblib shap
```

## Entraînement des modèles

Un utilitaire CLI centralise l'entraînement :

```bash
python -m ml.train_all --start-year 2020
```

- `--start-year` est optionnel et permet de restreindre les données (`FAIT_POINTAGE`, `FAIT_remuneration`) à partir d'une année donnée.
- Les modèles sont stockés dans `ml/models/` (créé automatiquement).

Le script entraîne successivement :

1. **Prévision staffing** (`staffing_forecast.pkl`) – prédit l’effectif attendu par service/semaine.
2. **Benchmark rémunération** (`compensation_benchmark.pkl`) – estime le salaire attendu par collaborateur.
3. **Prévision masse salariale** (`payroll_forecast.pkl`) – anticipe la masse salariale par service/mois.

Chaque entraînement retourne un indicateur (MAE, R²…) dans la sortie console.

## Endpoints REST

Le blueprint `ml_api` expose plusieurs routes :

- `POST /api/ml/train` : relance l’entraînement complet. Payload optionnel `{"start_year": 2020}`.
- `GET  /api/ml/staffing_forecast?year=2024` : prévisions d’effectifs par service/semaine.
- `GET  /api/ml/compensation_benchmark?limit=100` : liste des écarts salaire réel vs attendu.
- `GET  /api/ml/payroll_forecast?year=2025` : projection de masse salariale sur 12 mois.
- `POST /api/ml/scenarios/payroll` : simulation d’impact (embauches, augmentations, heures sup).

Ces endpoints s’appuient sur les modèles stockés dans `ml/models/`. Relancez `POST /api/ml/train` dès que de nouvelles données sont disponibles.

## Intégration UI

Les réponses JSON peuvent être branchées sur de nouveaux onglets/graphes dans les dashboards existants :

- Staffing : heatmap ou tableau de planification.
- Benchmark : liste priorisée des services/collaborateurs à auditer.
- Masse salariale : graphiques prévisionnels et alertes de dérive.
- Simulation : formulaire interactif “what‑if”.

## Scheduling

- Utiliser `cron`/Airflow (ou un simple `systemd timer`) pour exécuter `python -m ml.train_all` chaque mois.
- Les modèles étant idempotents, le script peut être relancé sans risque ; les fichiers `*.pkl` sont écrasés.

## Notes

- Les requêtes SQL s’appuient sur `DIM_EMPLOYEe_nouveau`, `FAIT_POINTAGE`, `FAIT_remuneration`. Adapter si vos tables évoluent.
- Le module `ml/scenario_simulator.py` illustre un calcul d’impact simple. Ajustez la logique métier (coût horaire, règles d’heures sup…) selon vos besoins.
