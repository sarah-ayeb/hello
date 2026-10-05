# Classification automatique des réclamations clients

Pipeline NLP multilingue (français / anglais) pour classer automatiquement les réclamations clients.

Le projet compare deux approches :

* **Baseline :** TF-IDF + Logistic Regression
* **Modèle avancé :** DistilBERT multilingue fine-tuné

L'évaluation utilise plusieurs méthodes :

* Validation croisée
* Intervalles de confiance par bootstrap
* Test de McNemar
* Analyse des erreurs
* Évaluation globale et par langue

---

## Sommaire

* [1. Limites du projet](#1-limites-du-projet)
* [2. Vue d'ensemble du pipeline](#2-vue-densemble-du-pipeline)
* [3. Structure du notebook](#3-structure-du-notebook)
* [4. Format des données](#4-format-des-données)
* [5. Configuration](#5-configuration)
* [6. Modèles](#6-modèles)
* [7. Méthodologie d'évaluation](#7-méthodologie-dévaluation)
* [8. Analyse des erreurs](#8-analyse-des-erreurs)
* [9. Artefacts produits](#9-artefacts-produits)

---

# 1. Limites du projet

Les résultats doivent être interprétés en tenant compte des limites du dataset.

## 1.1 Données issues du scraping

Le dataset a été construit à partir de commentaires et d'avis récupérés par scraping.

Les textes ne sont donc pas toujours des réclamations structurées.

Certains commentaires ressemblent plutôt à :

* des témoignages ;
* des récits d'expérience ;
* des textes narratifs.

Le type de réclamation peut parfois être difficile à identifier.

Le signal important peut aussi être noyé dans un texte long.

Cela peut rendre l'annotation et la classification plus difficiles.

---

## 1.2 Déséquilibre entre les catégories

Les catégories ne contiennent pas toutes le même nombre d'exemples.

Certaines classes sont beaucoup plus représentées que d'autres.

Pour limiter cet effet, le projet utilise des poids de classe équilibrés :

```python
class_weight="balanced"
```

Cependant, le déséquilibre reste une limite importante du dataset.

---

## 1.3 Catégories parfois trop larges

Certaines catégories regroupent plusieurs types de réclamations.

Cela peut créer des frontières floues entre les classes.

Par exemple, deux réclamations différentes peuvent appartenir à une même catégorie générale.

Cela peut expliquer certaines erreurs de classification.

---

## 1.4 Entités ajoutées artificiellement

Les textes issus du scraping contiennent rarement des informations structurées comme :

* une ville ;
* un montant ;
* une date ;
* un numéro de suivi.

Certaines entités ont donc été générées avec Claude puis ajoutées aux textes, **en amont du notebook**, lors de la constitution du dataset. Cette étape n'apparaît pas dans le pipeline présenté ici : le notebook charge directement le CSV déjà enrichi.

L'objectif était d'enrichir les données tout en gardant un style naturel.

Cette étape représente cependant une limite méthodologique.

Les entités générées peuvent créer des formats artificiels qui ne représentent pas parfaitement les vrais textes utilisateurs.

Pour réduire les risques de fuite d'information, certaines entités sont masquées ou exclues des features utilisées par les modèles.

Malgré cela, un biais résiduel reste possible.

---

## 1.5 Confusion entre catégories proches

Certaines catégories sont proches sur le plan sémantique.

Les modèles peuvent donc confondre plusieurs types de réclamations.

Ces erreurs sont normales lorsque les frontières entre les catégories ne sont pas clairement définies.

---

## 1.6 Une seule catégorie par texte

Le projet utilise une classification **mono-label**.

Cela signifie qu'un seul label est attribué à chaque texte.

Cependant, une même réclamation peut contenir plusieurs problèmes.

Par exemple :

* un retard de livraison ;
* un problème de remboursement.

Dans ce cas, une seule catégorie ne représente pas toujours toute la situation.

Une classification **multi-label** pourrait donc être une évolution intéressante du projet.

---

### Résumé des limites

Les scores obtenus doivent être interprétés comme une estimation des performances sur un dataset imparfait et partiellement synthétique.

Ils ne représentent pas nécessairement les performances finales d'un système utilisé en production.

La méthodologie d'évaluation a donc été renforcée pour mieux mesurer :

* la stabilité des modèles ;
* l'incertitude des résultats ;
* les différences entre les langues ;
* les performances par catégorie ;
* les erreurs de classification.

---

# 2. Vue d'ensemble du pipeline

```text
Chargement du dataset
        │
        ▼
EDA
(distribution, langue, valeurs manquantes)
        │
        ▼
Nettoyage et prétraitement
        │
        ▼
Suppression des quasi-doublons
        │
        ▼
Split stratifié
(catégorie × langue)
        │
        ▼
Gestion du déséquilibre des classes
        │
        ├──────────────────────────┐
        ▼                          ▼
TF-IDF + Logistic Regression   DistilBERT multilingue
        │                          │
        ▼                          ▼
Validation croisée              Fine-tuning
et recherche d'hyperparamètres
        │                          │
        └──────────────┬───────────┘
                       ▼
              Évaluation finale
                       │
                       ▼
       Tests statistiques et bootstrap
                       │
                       ▼
              Analyse des erreurs
                       │
                       ▼
                Export des artefacts
```

---

# 3. Structure du notebook

## 3.1 Installation des dépendances

Installation des bibliothèques nécessaires :

* `langdetect`
* `statsmodels`
* `transformers`
* `datasets`
* `accelerate`

Certaines bibliothèques sont déjà disponibles dans l'environnement Colab.

---

## 3.2 Configuration du projet

Les paramètres principaux sont regroupés dans une configuration centralisée.

Cela permet de :

* faciliter la reproductibilité ;
* éviter les valeurs dispersées dans le code ;
* modifier facilement les paramètres du pipeline.

Les principaux paramètres sont :

* seed ;
* chemins des fichiers ;
* noms des colonnes ;
* proportions du split ;
* seuil de similarité ;
* paramètres du TF-IDF.

---

## 3.3 Chargement des données

Le dataset peut être chargé :

* directement dans Google Colab ;
* depuis Google Drive.

Le fichier doit contenir au minimum :

* une colonne contenant le texte ;
* une colonne contenant la catégorie cible.

---

## 3.4 Analyse exploratoire des données (EDA)

L'EDA permet d'analyser :

* la taille du dataset ;
* les valeurs manquantes ;
* les doublons ;
* la distribution des catégories ;
* la distribution des langues ;
* les risques de fuite d'information.

La langue est détectée avec `langdetect`.

Les textes sont classés en :

* français (`fr`) ;
* anglais (`en`) ;
* autre (`other` dans le code) ;
* `unknown` lorsque la langue ne peut pas être détectée.

Une analyse croisée entre les catégories et les langues permet aussi de vérifier la répartition des données.

---

## 3.5 Détection des risques de fuite

Le pipeline vérifie la présence :

* de montants ;
* de numéros de tracking.

Cette étape permet d'identifier les informations qui pourraient donner directement un indice sur la catégorie.

Ces informations sont ensuite masquées ou exclues des features.

---

# 4. Prétraitement des textes

Deux méthodes de nettoyage sont utilisées.

## 4.1 Nettoyage commun

Le nettoyage commun comprend :

* normalisation Unicode ;
* masquage des numéros de tracking ;
* masquage des passages censurés ;
* normalisation des espaces.

Le but est de réduire le bruit sans supprimer le sens du texte.

---

## 4.2 Nettoyage pour TF-IDF

Le texte est davantage nettoyé pour le modèle classique.

Les principales opérations sont :

* conversion en minuscules ;
* suppression de certains caractères ;
* suppression de la ponctuation inutile.

Cette approche est adaptée à un modèle basé sur les mots.

---

## 4.3 Nettoyage pour DistilBERT

Le nettoyage est volontairement minimal.

La casse et la ponctuation sont conservées.

Elles peuvent contenir des informations utiles pour le modèle Transformer.

---

# 5. Suppression des quasi-doublons

Les textes très similaires peuvent provoquer une fuite entre les différents ensembles de données.

Pour éviter ce problème :

1. Les textes sont vectorisés avec TF-IDF.
2. La similarité cosinus est calculée.
3. Les textes ayant une similarité supérieure à `0.92` sont considérés comme des quasi-doublons.
4. Ces doublons sont traités avant le split.

Cela permet d'éviter qu'un texte presque identique soit présent à la fois dans le train et dans le test.

---

# 6. Séparation des données

Le dataset est séparé en **environ 70 % pour l'entraînement, 15 % pour la validation et 15 % pour le test**.

Cette séparation est obtenue à partir des paramètres suivants :

```python
test_size = 0.15
val_size = 0.15
```

La séparation utilise une stratification basée sur :

```text
categorie_finale × langue
```

Cette méthode permet de conserver une distribution similaire des catégories et des langues dans chaque ensemble.

Les petites strates sont signalées afin d'identifier les situations où les résultats peuvent être instables.

Le **test set reste totalement séparé** pendant l'entraînement et le développement des modèles. Il est utilisé uniquement lors de l'évaluation finale.


# 7. Gestion du déséquilibre des classes

Les poids de classe sont calculés à partir du train set.

Ils sont utilisés pour :

* la Logistic Regression ;
* la fonction de perte de DistilBERT.

L'objectif est de donner plus d'importance aux catégories minoritaires.

---

# 8. Modèles utilisés

## 8.1 Baseline : TF-IDF + Logistic Regression
TF-IDF → GridSearchCV → validation croisée 5 plis → choix des hyperparamètres (C, L1, L2) → meilleur modèle.

Donc ici, la validation croisée sert notamment à rechercher les meilleurs hyperparamètres.

La baseline utilise :

* TF-IDF ;
* des n-grammes de 1 à 2 mots ;
* un vocabulaire limité à 30 000 termes ;
* Logistic Regression ;
* des poids de classe équilibrés.

Le modèle est évalué avec :

* une validation croisée à 5 plis ;
* une recherche d'hyperparamètres avec `GridSearchCV`.

Les paramètres testés comprennent :

* la valeur de `C` ;
* la pénalité `L1` ;
* la pénalité `L2`.

Le meilleur modèle est sauvegardé avec son vectoriseur.

---

## 8.2 DistilBERT multilingue

Le modèle utilisé est :

```text
distilbert-base-multilingual-cased
```

Il est adapté aux textes en français et en anglais.

Le modèle peut aussi gérer des textes contenant plusieurs langues.

### Paramètres principaux

* Longueur maximale : `256 tokens`
* Batch size : `16`
* Nombre d'époques : `8`
* Validation croisée : `5 plis`

La tokenisation est réalisée avec Hugging Face.

Le padding dynamique est utilisé pour réduire la consommation de mémoire.

---

## 8.3 Fine-tuning
Train set → validation croisée 5 plis pour mesurer la stabilité → entraînement final sur le train complet → validation set pour suivre l'entraînement → sélection du meilleur checkpoint selon eval_loss → test final.

Donc ici, la validation croisée sert surtout à évaluer la stabilité du modèle, tandis que le validation set sert au suivi de l'entraînement et à la sélection du checkpoint.
Le modèle est fine-tuné sur le dataset de réclamations.

Les différentes parties du modèle peuvent utiliser des learning rates différents.

Les couches pré-entraînées sont mises à jour plus doucement.

La tête de classification apprend avec un learning rate plus élevé.

Cette méthode permet de mieux adapter le modèle au dataset.

---

Le meilleur checkpoint de DistilBERT est sélectionné selon la **`eval_loss`** observée sur le validation set.

Le **F1-macro** est également calculé et suivi afin d'analyser les performances du modèle, mais il n'est pas utilisé comme critère principal pour sélectionner le meilleur checkpoint.

Cette distinction permet de conserver une procédure de sélection cohérente avec la fonction de perte optimisée pendant le fine-tuning, tout en surveillant une métrique particulièrement pertinente pour un problème de classification potentiellement déséquilibré.
  
# 9. Méthodologie d'évaluation

Les deux modèles sont évalués avec les mêmes métriques.

Cela garantit une comparaison équitable.

## Métriques utilisées

* **F1-macro**
* **F1-weighted**
* **Balanced Accuracy**

Les performances sont mesurées :

* globalement ;
* par langue ;
* par catégorie.

---

## 9.1 Validation croisée

Une validation croisée à 5 plis est utilisée pour mesurer la stabilité des modèles.

Pour **DistilBERT**, la validation croisée est réalisée **uniquement sur le train set** afin d'évaluer la stabilité du modèle selon différentes séparations des données d'entraînement.

Après cette étape, un **modèle final est entraîné sur l'ensemble du train set**.

Le **validation set** est utilisé pendant l'entraînement du modèle final afin de suivre les performances et de sélectionner le meilleur checkpoint.

Le **test set reste totalement séparé** jusqu'à l'évaluation finale. Il n'est utilisé ni pour l'entraînement, ni pour le choix des hyperparamètres, ni pour la sélection du meilleur checkpoint.

Cette organisation permet de limiter les risques de fuite de données et de conserver une estimation indépendante des performances finales.

---

## 9.2 Bootstrap

Des intervalles de confiance à 95 % sont calculés avec le bootstrap.

Cette méthode permet d'obtenir une estimation de l'incertitude autour du score obtenu.

Le score n'est donc pas présenté comme une valeur absolue.

---

## 9.3 Test de McNemar

Le test de McNemar compare les erreurs des deux modèles sur les mêmes exemples.

Il permet de vérifier si la différence entre les modèles est statistiquement significative.

---

## 9.4 Diagnostic par catégorie

Les métriques sont aussi calculées pour chaque catégorie.

Cela permet d'identifier les catégories où les modèles rencontrent le plus de difficultés.

Un score global peut parfois cacher de mauvaises performances sur une classe minoritaire.

---

# 10. Évaluation finale

Les deux modèles sont évalués sur le test set.

Le test set n'est jamais utilisé pour :

* entraîner les modèles ;
* choisir les hyperparamètres ;
* sélectionner le meilleur modèle.

Les résultats sont regroupés dans un tableau comparatif.

Les métriques principales sont :

* F1-macro global ;
* F1-weighted global ;
* Balanced Accuracy ;
* F1-macro par langue.

Le tableau final est exporté dans :

```text
reports/comparaison_finale_modeles.csv
```

---

# 11. Analyse des erreurs

L'analyse des erreurs comprend :

* les matrices de confusion ;
* les catégories souvent confondues ;
* les exemples mal classés ;
* l'analyse par langue.

Les erreurs sont analysées pour comprendre :

* pourquoi certaines catégories sont confondues ;
* si les catégories sont trop proches ;
* si le texte contient plusieurs types de réclamations ;
* si la langue influence les performances.

---

# 12. Format des données attendu

Le dataset doit être un fichier CSV.

Il doit contenir au minimum :

| Colonne            | Description                               |
| ------------------ | ----------------------------------------- |
| `texte`            | Texte de la réclamation ou du commentaire |
| `categorie_finale` | Catégorie à prédire                       |

Les colonnes suivantes peuvent également être présentes :

* `ville`
* `montant`
* `montant_num`

Ces colonnes sont exclues des features utilisées par les modèles.

---

# 13. Configuration

Les principaux paramètres du pipeline sont centralisés.

| Paramètre                       |  Valeur par défaut | Rôle                     |
| ------------------------------- | -----------------: | ------------------------ |
| `seed`                          |                 42 | Reproductibilité         |
| `text_col`                      |            `texte` | Colonne du texte         |
| `target_col`                    | `categorie_finale` | Colonne cible            |
| `test_size`                     |               0.15 | Taille du test set       |
| `val_size`                      |               0.15 | Taille du validation set |
| `min_stratum_size_warning`      |                 30 | Seuil d'alerte           |
| `near_dup_similarity_threshold` |               0.92 | Seuil des quasi-doublons |
| `tfidf_ngram_range`             |             (1, 2) | N-grammes                |
| `tfidf_min_df`                  |                  2 | Fréquence minimale       |
| `tfidf_max_features`            |             30 000 | Taille du vocabulaire    |

---

# 14. Artefacts produits

## Modèles

```text
models/tfidf_vectorizer.joblib
models/logistic_regression.joblib
models/distilbert_checkpoints/   # checkpoints intermédiaires (un par époque, sauvegardés pendant l'entraînement)
models/distilbert_final/         # modèle final retenu (meilleur checkpoint) + tokenizer, prêt pour l'inférence
```

## Rapports

```text
reports/figures/
reports/comparaison_finale_modeles.csv
```

Les figures comprennent notamment :

* la distribution des catégories ;
* la distribution des langues ;
* la heatmap catégorie × langue ;
* les matrices de confusion.

---

## Archives

Les artefacts peuvent être regroupés dans :

```text
reclamations_outputs.zip
reclamations_reports.zip
```

Ces fichiers peuvent être téléchargés directement depuis Google Colab.

---

# Conclusion

Ce projet propose une comparaison entre une approche classique et une approche basée sur les Transformers pour la classification automatique de réclamations clients.

La baseline **TF-IDF + Logistic Regression** permet d'obtenir une solution simple et rapide.

Le modèle **DistilBERT multilingue** permet d'explorer une approche plus avancée.

L'évaluation ne repose pas uniquement sur un score global.

Elle prend également en compte :

* la stabilité des modèles ;
* les différences entre les langues ;
* les performances par catégorie ;
* l'incertitude statistique ;
* les erreurs de classification.

Cette approche permet d'obtenir une analyse plus complète des performances et des limites du système.
