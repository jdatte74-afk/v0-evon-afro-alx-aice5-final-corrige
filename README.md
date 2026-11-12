# EVON AFRO — ALX AiCE-5

**Projet :** tableau de bord interactif de segmentation des clients automobiles.  
**Auteur / nom du projet :** EVON AFRO.  
**Algorithme :** Gradient Boosting (scikit-learn), exporté pour une prédiction dans le navigateur.

## 1. Contenu du dossier

- `index.html` : interface web responsive, filtres, graphiques et simulateur de prédiction.
- `data.js` : données de synthèse utilisées par les visualisations.
- `model.js` : modèle Gradient Boosting exporté et fonction `predict()` utilisée par le simulateur.
- `data.csv` : jeu de données source (8 068 lignes, 11 colonnes).
- `metrics.json` : métriques d'évaluation, matrice de confusion et diagnostic des données.
- `train_model.py` : script pour reproduire l'entraînement, la validation et l'export du modèle.
- `requirements.txt` : dépendances Python nécessaires à la reproduction.

## 2. Données et préparation

Le jeu de données contient 8 068 profils et les quatre segments A, B, C et D. La colonne `ID` est exclue des variables explicatives. Les valeurs manquantes sont imputées à l'intérieur du pipeline : modalité la plus fréquente pour les variables catégorielles et médiane pour les variables numériques. L'encodage one-hot est également ajusté dans chaque pli de validation.

Le diagnostic détecte 417 lignes ayant des caractéristiques identiques lorsque l'identifiant est ignoré. Elles sont signalées, mais conservées : il peut s'agir de clients distincts présentant le même profil.

Le fichier fourni ne contient pas les 2 627 nouveaux clients mentionnés dans le descriptif initial. Aucune prédiction pour ces nouveaux clients n'est donc revendiquée dans cette livraison.

## 3. Modèle et évaluation

- Modèle : `GradientBoostingClassifier(random_state=42)`.
- Évaluation : validation croisée stratifiée à 5 plis (`StratifiedKFold`, mélange activé, graine 42), avec prédictions hors-échantillon.
- Exactitude : **53,79 %**.
- F1 macro : **52,54 %**.
- Exactitude équilibrée : **52,85 %**.
- Référence majoritaire : **28,11 %**.

Le simulateur du navigateur utilise les arbres exportés depuis le modèle entraîné. Les probabilités affichées sont des probabilités brutes non calibrées : elles donnent une indication et ne doivent pas être interprétées comme une certitude.

Le fonctionnement de `model.js` a été contrôlé par comparaison avec `predict_proba` de scikit-learn sur des profils du jeu de données ; les probabilités reproduites concordent.

## 4. Limites d'interprétation commerciale

Le jeu de données ne contient ni ventes, ni prix, ni marges, ni résultats de campagnes, ni choix de produits P1 à P5. Les recommandations commerciales présentées dans le tableau de bord sont donc des hypothèses à tester avec les équipes et les données métier, et non des conclusions sur la rentabilité ou la performance réelle des produits.

Une validation sur des données récentes et représentatives du marché ciblé est nécessaire avant toute utilisation opérationnelle.

## 5. Lancer le projet localement

1. Décompresser le dossier.
2. Ouvrir `index.html` dans un navigateur récent.
3. Une connexion Internet est nécessaire pour charger Chart.js depuis jsDelivr.
4. Tester les filtres, les graphiques et le simulateur de prédiction.

## 6. Reproduire le modèle (facultatif)

Depuis le dossier du projet :

```bash
pip install -r requirements.txt
python train_model.py
```

Le script lit `data.csv` depuis le dossier du projet et régénère `model.js`.

## 7. Publier le tableau de bord pour obtenir un lien public

Le dossier est un site statique : il ne nécessite ni serveur applicatif, ni commande de compilation.

### Option conseillée : Netlify Drop

1. Depuis Chrome, ouvrir `https://app.netlify.com/drop`.
2. Choisir le fichier ZIP final fourni avec ce projet. Netlify Drop accepte les fichiers ZIP ; le ZIP de livraison contient `index.html` directement à sa racine.
3. Attendre la publication. Si Netlify demande une connexion, créer un compte gratuit ou se connecter.
4. Netlify fournit une adresse publique se terminant par `netlify.app`.
5. Ouvrir cette adresse dans une fenêtre privée ou sur un autre téléphone pour vérifier l'accès public, les graphiques et le simulateur.

### Alternative : Vercel

1. Ouvrir `https://vercel.com` et se connecter.
2. Importer le dossier du site ou un dépôt GitHub contenant les fichiers.
3. Pour ce site statique, ne renseigner aucune commande de build ; publier le dossier racine contenant `index.html`.
4. Vercel fournit une adresse publique se terminant par `vercel.app`.

**Important :** le ZIP seul n'est pas encore un lien web public. Le lien `netlify.app` ou `vercel.app` doit être créé après publication, puis testé avant de le soumettre à ALX.
