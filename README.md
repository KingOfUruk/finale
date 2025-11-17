# Plateforme RH / Paie

Portail d'analyse RH et paie construit sur Flask. L'application regroupe plusieurs
modules (authentification sécurisée, payroll, analytics RH, performance employé,
API ML) connectés à Oracle Autonomous Database via wallet/DSN.

## Prérequis

- Python 3.10+
- Wallet Oracle (`Wallet_PFE/` existe déjà mais vous pouvez le remplacer par le vôtre)
- Accès réseau vers `adb.eu-paris-1.oraclecloud.com:1522`

## Lancer l'application en local

1. Renseignez `.env` avec vos identifiants Oracle/Redis (exemple dans `docs/setup.md`).
2. Exécutez le script helper (créera `.venv`, installera les dépendances et chargera `.env`) :

   ```bash
   ./scripts/run_local.sh
   ```

   Utilisez `SKIP_PIP_INSTALL=1 ./scripts/run_local.sh` si vous ne souhaitez pas
   réinstaller les dépendances à chaque lancement.

3. L'application écoute sur <http://127.0.0.1:5000>. Vérifiez `/healthz` et
   `/readyz` pour l'état des dépendances (Oracle, Redis, Celery).

## Déploiements

- **Vercel (serverless)** : voir `docs/deploying-to-vercel.md`. La configuration
  `vercel.json` route tout le trafic vers `api/index.py`, lequel ré-exporte
  l'application Flask existante. Fournissez les secrets Oracle/Redis via le
  dashboard Vercel et assurez-vous que le wallet (`Wallet_PFE`) accompagne le
  build si vous utilisez l'ADB.
- **Railway** : `Dockerfile` est détecté automatiquement par Railway. Le guide
  `docs/deploying-to-railway.md` décrit l'import du dépôt, la configuration des
  variables d'environnement, l'utilisation du wallet et le lancement éventuel
  d'un service Celery séparé.
- **Render** : `render.yaml` et `runtime.txt` décrivent le service Flask + le
  worker Celery. Ajoutez les secrets dans le dashboard et fournissez le wallet
  (volume ou script base64).
- **Fly.io** : `Dockerfile` + `fly.toml` permettent un déploiement containerisé.
  Définissez les secrets (`fly secrets set …`) puis `fly deploy`. Le wallet est
  intégré à l'image ou injecté via un secret base64 selon vos besoins.

Variables optionnelles utiles :
- `HR_ANALYTICS_LOOKBACK_YEARS` (défaut : 3) limite l'historique chargé pour le
  dashboard RH.
- `HR_ANALYTICS_REFRESH_SECONDS` (défaut : 900) cadence le rafraîchissement
  asynchrone des jeux de données.
- `HR_ANALYTICS_EMPLOYEE_LIMIT` (défaut : 200) borne le nombre d’employés
  renvoyés dans l’annuaire.

Pour plus de détails (init utilisateurs, Redis, Celery, Prometheus), consultez
`docs/setup.md`.
