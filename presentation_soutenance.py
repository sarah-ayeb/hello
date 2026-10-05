#!/usr/bin/env python3
"""
Génération de présentation PowerPoint pour soutenance de stage
Projet : Classification et routage automatique de réclamations clients
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
import os

# Création de la présentation
prs = Presentation()
prs.slide_width = Inches(10)
prs.slide_height = Inches(7.5)

# Palettes de couleurs
COLOR_DARK_BLUE = RGBColor(25, 55, 109)
COLOR_LIGHT_BLUE = RGBColor(68, 114, 196)
COLOR_ACCENT = RGBColor(192, 0, 0)
COLOR_LIGHT_GRAY = RGBColor(242, 242, 242)
COLOR_WHITE = RGBColor(255, 255, 255)

def add_title_slide(prs, title, subtitle):
    """Crée une diapositive titre"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank layout
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = COLOR_DARK_BLUE
    
    # Titre
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(2.5), Inches(9), Inches(1.5))
    title_frame = title_box.text_frame
    title_frame.word_wrap = True
    title_p = title_frame.paragraphs[0]
    title_p.text = title
    title_p.font.size = Pt(54)
    title_p.font.bold = True
    title_p.font.color.rgb = COLOR_WHITE
    title_p.alignment = PP_ALIGN.CENTER
    
    # Sous-titre
    subtitle_box = slide.shapes.add_textbox(Inches(0.5), Inches(4.2), Inches(9), Inches(2))
    subtitle_frame = subtitle_box.text_frame
    subtitle_frame.word_wrap = True
    subtitle_p = subtitle_frame.paragraphs[0]
    subtitle_p.text = subtitle
    subtitle_p.font.size = Pt(28)
    subtitle_p.font.color.rgb = COLOR_LIGHT_BLUE
    subtitle_p.alignment = PP_ALIGN.CENTER

def add_content_slide(prs, title, content_func=None):
    """Crée une diapositive de contenu avec titre"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = COLOR_WHITE
    
    # Barre de titre
    title_bar = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(10), Inches(0.8))
    title_bar.fill.solid()
    title_bar.fill.fore_color.rgb = COLOR_DARK_BLUE
    title_bar.line.color.rgb = COLOR_DARK_BLUE
    
    # Texte du titre
    title_frame = title_bar.text_frame
    title_frame.text = title
    title_p = title_frame.paragraphs[0]
    title_p.font.size = Pt(40)
    title_p.font.bold = True
    title_p.font.color.rgb = COLOR_WHITE
    title_frame.margin_left = Inches(0.4)
    title_frame.margin_top = Inches(0.1)
    
    if content_func:
        content_func(slide)
    
    return slide

def add_bullet_points(slide, bullet_list, start_top=1.2, start_left=0.5):
    """Ajoute une liste de points de balle"""
    text_box = slide.shapes.add_textbox(Inches(start_left), Inches(start_top), Inches(9), Inches(5.5))
    text_frame = text_box.text_frame
    text_frame.word_wrap = True
    
    for i, bullet in enumerate(bullet_list):
        if i == 0:
            p = text_frame.paragraphs[0]
        else:
            p = text_frame.add_paragraph()
        p.text = bullet
        p.level = 0
        p.font.size = Pt(18)
        p.font.color.rgb = RGBColor(0, 0, 0)
        p.space_before = Pt(6)
        p.space_after = Pt(6)

def add_title_and_subtitle(prs):
    """Slide 1 : Titre et sous-titre"""
    add_title_slide(prs, 
        "Classification et Routage Automatique\nde Réclamations Clients",
        "Soutenance de Stage Data Science / Data Engineering")

def add_context_slide(prs):
    """Slide 2 : Contexte et objectifs"""
    def content(slide):
        bullets = [
            "🎯 Objectif : Automatiser la classification de réclamations clients",
            "📊 Finalité : Préparer le routage vers les départements appropriés",
            "🌐 Défi multilingue : Français et Anglais",
            "📈 Approche : Comparaison ML classique vs. Transformers modernes",
            "🔐 Contrainte importante : Prévention du data leakage"
        ]
        add_bullet_points(slide, bullets)
    
    add_content_slide(prs, "Contexte et Objectifs", content)

def add_dataset_slide(prs):
    """Slide 3 : Vue d'ensemble des données"""
    def content(slide):
        bullets = [
            "📌 Source : Réclamations clients en français et anglais",
            "📁 Étapes de traitement :",
            "    • Chargement : 2 961 réclamations",
            "    • Après déduplication (0.92 similarité) : 2 814 réclamations",
            "",
            "📊 Répartition des données :",
            "    • Entraînement (train) : 1 969 (70%)",
            "    • Validation (val) : 422 (15%)",
            "    • Test : 423 (15%)"
        ]
        add_bullet_points(slide, bullets)
    
    add_content_slide(prs, "Dataset et Analyse Exploratoire - Tailles", content)

def add_distribution_slide(prs):
    """Slide 4 : Distribution des langues et catégories"""
    def content(slide):
        # Langues
        bullets = [
            "🗣️ Répartition linguistique :",
            "    • Français : 2 125 textes (73.2%)",
            "    • Anglais : 836 textes (28.8%)",
            "",
            "📂 6 catégories de réclamations :",
            "    • Livraison / Point relais (42.0% en FR, 11.7% en EN)",
            "    • Autre, Facturation, Qualité de service",
            "    • Retard, Sinistre colis",
            "",
            "⚠️ Déséquilibre des classes : ratio max/min ≈ 3.3x"
        ]
        add_bullet_points(slide, bullets)
    
    add_content_slide(prs, "Dataset - Distribution Linguistique et Catégories", content)

def add_viz_slide_1(prs):
    """Slide 5 : Visualisation distribution catégories"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = COLOR_WHITE
    
    # Barre titre
    title_bar = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(10), Inches(0.8))
    title_bar.fill.solid()
    title_bar.fill.fore_color.rgb = COLOR_DARK_BLUE
    title_bar.line.color.rgb = COLOR_DARK_BLUE
    
    title_frame = title_bar.text_frame
    title_frame.text = "Distribution des Catégories"
    title_p = title_frame.paragraphs[0]
    title_p.font.size = Pt(40)
    title_p.font.bold = True
    title_p.font.color.rgb = COLOR_WHITE
    title_frame.margin_left = Inches(0.4)
    title_frame.margin_top = Inches(0.1)
    
    # Image
    img_path = "notebook_reclamations/reports/figures/distribution_categories.png"
    if os.path.exists(img_path):
        slide.shapes.add_picture(img_path, Inches(1), Inches(1.2), width=Inches(8))

def add_viz_slide_2(prs):
    """Slide 6 : Visualisation catégorie x langue"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = COLOR_WHITE
    
    # Barre titre
    title_bar = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(10), Inches(0.8))
    title_bar.fill.solid()
    title_bar.fill.fore_color.rgb = COLOR_DARK_BLUE
    title_bar.line.color.rgb = COLOR_DARK_BLUE
    
    title_frame = title_bar.text_frame
    title_frame.text = "Répartition : Catégorie × Langue"
    title_p = title_frame.paragraphs[0]
    title_p.font.size = Pt(40)
    title_p.font.bold = True
    title_p.font.color.rgb = COLOR_WHITE
    title_frame.margin_left = Inches(0.4)
    title_frame.margin_top = Inches(0.1)
    
    # Image
    img_path = "notebook_reclamations/reports/figures/categorie_x_langue.png"
    if os.path.exists(img_path):
        slide.shapes.add_picture(img_path, Inches(0.5), Inches(1.2), width=Inches(9))

def add_preprocessing_slide(prs):
    """Slide 7 : Pipeline de prétraitement"""
    def content(slide):
        bullets = [
            "🔄 Pipeline de prétraitement complet :",
            "",
            "1️⃣ Vérification qualité (valeurs manquantes, doublons)",
            "2️⃣ Détection de langue (langdetect)",
            "3️⃣ Normalisation Unicode et espaces",
            "4️⃣ Masquage des informations sensibles (tracking, montants)",
            "5️⃣ Détection des quasi-doublons (TF-IDF + cosinus ≥ 0.92)",
            "6️⃣ Split stratifié (catégorie × langue)"
        ]
        add_bullet_points(slide, bullets, start_top=0.95)
    
    add_content_slide(prs, "Étape 1 - Prétraitement des Données", content)

def add_data_leakage_slide(prs):
    """Slide 8 : Prévention du data leakage"""
    def content(slide):
        bullets = [
            "⚠️ Enjeu critique : Prévention du data leakage",
            "",
            "Variables de fuite identifiées :",
            "    • Numéros de tracking : 98.4% des textes",
            "    • Montants financiers : jusqu'à 89.8% (Sinistre colis)",
            "    • Ces variables peuvent directement révéler la catégorie",
            "",
            "✅ Actions correctives :",
            "    • Masquage systématique : __TRACKING__, __CENSURE__",
            "    • Exclusion des colonnes sensibles du modèle",
            "    • Déduplication avant split train/val/test"
        ]
        add_bullet_points(slide, bullets, start_top=0.95)
    
    add_content_slide(prs, "Prévention du Data Leakage", content)

def add_baseline_slide(prs):
    """Slide 9 : Modèle baseline TF-IDF + Logistic Regression"""
    def content(slide):
        bullets = [
            "📌 Approche classique ML - Baseline robuste",
            "",
            "Pipeline :",
            "    Texte → TF-IDF → Logistic Regression → Catégorie",
            "",
            "Configuration TF-IDF :",
            "    • N-grammes : 1-2",
            "    • Vocabulaire : 30 000 termes max",
            "    • 25 892 termes réels extraits",
            "",
            "Optimisation :",
            "    • GridSearchCV : C ∈ {0.01, 0.1, 1.0, 3.0, 10.0}",
            "    • Pénalités : L1, L2",
            "    • ✅ Meilleurs paramètres trouvés : C=10, penalty=L1",
            "",
            "Performance validation : F1-macro ≈ 0.7872"
        ]
        add_bullet_points(slide, bullets, start_top=0.95)
    
    add_content_slide(prs, "Modèle 1 - TF-IDF + Logistic Regression", content)

def add_distilbert_slide(prs):
    """Slide 10 : Modèle avancé DistilBERT"""
    def content(slide):
        bullets = [
            "🤖 Approche moderne - Transformers multilingues",
            "",
            "Modèle : distilbert-base-multilingual-cased",
            "    • Prédéfini sur 101 langues",
            "    • Économe (distillé vs BERT complet)",
            "",
            "Tokenisation : Hugging Face Transformers",
            "    • Max length : 256 tokens",
            "    • Padding dynamique (efficacité mémoire)",
            "",
            "Fine-tuning configuré :",
            "    • Batch size : 16",
            "    • Epochs : 8",
            "    • Learning rates discriminants (base < top < head)",
            "    • Loss pondérée par classe + Label smoothing",
            "    • Early stopping (patience=2)"
        ]
        add_bullet_points(slide, bullets, start_top=0.95)
    
    add_content_slide(prs, "Modèle 2 - DistilBERT Multilingue", content)

def add_evaluation_slide(prs):
    """Slide 11 : Méthodologie d'évaluation"""
    def content(slide):
        bullets = [
            "📊 Stratégie d'évaluation robuste et complète",
            "",
            "Métriques principales :",
            "    • F1-macro (pondération équitable par classe)",
            "    • F1-weighted (pondération par fréquence)",
            "    • Balanced Accuracy",
            "",
            "Niveaux d'analyse :",
            "    • Global (tous les textes)",
            "    • Par langue (FR vs EN)",
            "    • Par catégorie (6 classes)",
            "",
            "Techniques statistiques :",
            "    • Validation croisée 5-fold → stabilité du modèle",
            "    • Bootstrap CI 95% → incertitude estimée",
            "    • Test de McNemar → différence significative",
            "    • Matrice de confusion → analyse des erreurs"
        ]
        add_bullet_points(slide, bullets, start_top=0.95)
    
    add_content_slide(prs, "Méthodologie d'Évaluation", content)

def add_comparison_slide(prs):
    """Slide 12 : Comparaison des modèles"""
    def content(slide):
        text_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.2), Inches(9), Inches(5.5))
        text_frame = text_box.text_frame
        text_frame.word_wrap = True
        
        # Tableau simplifié
        data = [
            ("Critère", "TF-IDF + LR", "DistilBERT"),
            ("Approche", "ML Classique", "Transformer"),
            ("Vocab", "25 892 termes", "Subword BPE"),
            ("F1-macro (val)", "0.7872", "À évaluer"),
            ("Interprétabilité", "⭐⭐⭐", "⭐⭐"),
            ("Vitesse", "⭐⭐⭐", "⭐"),
            ("Multilingue", "⭐⭐", "⭐⭐⭐"),
            ("Contextuel", "Non", "Oui"),
        ]
        
        for i, (criterion, tfidf, distilbert) in enumerate(data):
            row_top = 1.2 + (i * 0.5)
            
            # Critère (à gauche)
            p1 = text_frame.paragraphs[0] if i == 0 else text_frame.add_paragraph()
            p1.text = f"{criterion:<20} | {tfidf:<20} | {distilbert}"
            p1.font.size = Pt(14) if i == 0 else Pt(13)
            p1.font.bold = i == 0
            p1.space_before = Pt(4)
            p1.space_after = Pt(4)
    
    add_content_slide(prs, "Comparaison des Modèles", content)

def add_ner_intro_slide(prs):
    """Slide 13 : Introduction au NER"""
    def content(slide):
        bullets = [
            "🏷️ Extraction d'Entités Nommées (NER)",
            "",
            "Rôle dans le projet :",
            "    • Identifier les informations structurées dans les textes",
            "    • Enrichir le contexte pour la classification",
            "    • Auditer la qualité des annotations",
            "",
            "Entités ciblées :",
            "    1️⃣ Numéro de tracking (TRK-XXXX, TN123456...)",
            "    2️⃣ Montant financier (€, DT...)",
            "    3️⃣ Date (JJ/MM/AAAA...)",
            "",
            "Exemple : ",
            '    📌 \"Colis TRK-5645 retardé depuis 25/09 pour 120 DT\"',
            "    → Extrait : TRK-5645 (TRACKING), 25/09 (DATE), 120 DT (AMOUNT)"
        ]
        add_bullet_points(slide, bullets, start_top=0.95)
    
    add_content_slide(prs, "Extraction d'Entités Nommées (NER)", content)

def add_ner_data_slide(prs):
    """Slide 14 : Données et modèles NER"""
    def content(slide):
        bullets = [
            "📦 Données annotées au format spaCy",
            "",
            "Ensembles d'entraînement NER :",
            "    • train.spacy - Données d'entraînement",
            "    • dev.spacy - Données de validation",
            "    • test.spacy - Données de test",
            "",
            "Modèles NER sauvegardés :",
            "    • model-best - Meilleur checkpoint",
            "    • model-last - Dernier checkpoint",
            "",
            "Rapports d'audit :",
            "    • audit_ner_a_completer.csv",
            "    • ner_sous_annotation_date.csv",
            "    • ner_sous_annotation_montant.csv",
            "    • ner_sous_annotation_numero_tracking.csv",
            "",
            "⚠️ Audit : Identification des annotations incomplètes"
        ]
        add_bullet_points(slide, bullets, start_top=0.95)
    
    add_content_slide(prs, "Données et Modèles NER", content)

def add_routing_slide(prs):
    """Slide 15 : Classification et routage"""
    def content(slide):
        bullets = [
            "🎯 De la classification au routage automatique",
            "",
            "Flux de traitement :",
            "",
            "    📝 Réclamation reçue",
            "         ↓",
            "    🔍 Extraction d'entités (NER)",
            "         ↓",
            "    🤖 Classification (TF-IDF ou DistilBERT)",
            "         ↓",
            "    📂 Détermine la catégorie",
            "         ↓",
            "    🚀 Routage vers département approprié",
            "         ↓",
            "    ✅ Traitement optimisé et rapide",
            "",
            "💡 Objectif fonctionnel : Automatiser le tri initial"
        ]
        add_bullet_points(slide, bullets, start_top=0.95)
    
    add_content_slide(prs, "Classification et Routage Automatique", content)

def add_artifacts_slide(prs):
    """Slide 16 : Artefacts produits"""
    def content(slide):
        bullets = [
            "📦 Artefacts du projet sauvegardés",
            "",
            "Modèles entraînés :",
            "    • baseline_tfidf_logreg_pipeline.joblib",
            "    • distilbert_cv_fold{1-5}/ (validation croisée)",
            "    • ner/model-best/ et model-last/",
            "",
            "Données traitées :",
            "    • train.csv, val.csv, test.csv (splits stratifiés)",
            "    • reclamations_dedup.csv (après déduplication)",
            "    • données NER au format spaCy",
            "",
            "Rapports et visualisations :",
            "    • distribution_categories.png",
            "    • categorie_x_langue.png",
            "    • audit_ner*.csv",
            "    • comparaison_finale_modeles.csv"
        ]
        add_bullet_points(slide, bullets, start_top=0.95)
    
    add_content_slide(prs, "Artefacts Produits", content)

def add_limitations_slide(prs):
    """Slide 17 : Limitations du projet"""
    def content(slide):
        bullets = [
            "⚠️ Limitations à considérer",
            "",
            "Données :",
            "    • Source : Scraping → textes non structurés",
            "    • Entités partiellement synthétiques (générées avec Claude)",
            "    • Déséquilibre des classes : ratio 3.3x",
            "    • Classification mono-label (une seule catégorie par texte)",
            "",
            "Méthodologie :",
            "    • Dataset de taille modérée (~2800 exemples)",
            "    • Catégories parfois trop larges ou floues",
            "    • Différences de performance FR vs EN",
            "",
            "Conclusions :",
            "    ✅ Résultats : estimation sur dataset imparfait",
            "    ✅ À valider en production avec données réelles"
        ]
        add_bullet_points(slide, bullets, start_top=0.95)
    
    add_content_slide(prs, "Limitations et Considérations", content)

def add_perspectives_slide(prs):
    """Slide 18 : Perspectives et améliorations futures"""
    def content(slide):
        bullets = [
            "🚀 Perspectives d'amélioration",
            "",
            "Court terme :",
            "    • Déployer le meilleur modèle en environnement de test",
            "    • Valider les performances avec données réelles",
            "    • Optimiser l'API d'inférence",
            "",
            "Moyen terme :",
            "    • Classification multi-label (plusieurs catégories)",
            "    • Explainabilité du modèle (LIME, SHAP)",
            "    • Intégration du pipeline NER + classification",
            "    • Monitoring des performances en production",
            "",
            "Long terme :",
            "    • Apprentissage actif (re-annotation continue)",
            "    • Adaptation cross-domaine",
            "    • Système de routage intelligent avec SLA"
        ]
        add_bullet_points(slide, bullets, start_top=0.95)
    
    add_content_slide(prs, "Perspectives Futures", content)

def add_conclusion_slide(prs):
    """Slide 19 : Conclusion"""
    def content(slide):
        bullets = [
            "✅ Résumé des réalisations",
            "",
            "Pipeline complet ML d'expérimentation :",
            "    • Prétraitement robuste avec gestion du data leakage",
            "    • Deux modèles contrastés (ML classique vs Transformers)",
            "    • Évaluation rigoureuse multi-critères",
            "    • Extraction d'entités pour enrichissement contextuel",
            "",
            "Infrastructure d'entraînement :",
            "    • Validation croisée & hyperparameter tuning",
            "    • Modèles sauvegardés et reproductibles",
            "    • Audit de qualité des données et annotations",
            "",
            "✨ Prêt pour déploiement et validation en production"
        ]
        add_bullet_points(slide, bullets, start_top=0.95)
    
    add_content_slide(prs, "Conclusion", content)

def add_thank_you_slide(prs):
    """Dernière slide : Remerciements"""
    add_title_slide(prs,
        "Merci de votre attention",
        "Questions ?")

# Génération de toutes les slides
def generate_presentation():
    print("Génération de la présentation PowerPoint...")
    
    add_title_and_subtitle(prs)
    add_context_slide(prs)
    add_dataset_slide(prs)
    add_distribution_slide(prs)
    add_viz_slide_1(prs)
    add_viz_slide_2(prs)
    add_preprocessing_slide(prs)
    add_data_leakage_slide(prs)
    add_baseline_slide(prs)
    add_distilbert_slide(prs)
    add_evaluation_slide(prs)
    add_comparison_slide(prs)
    add_ner_intro_slide(prs)
    add_ner_data_slide(prs)
    add_routing_slide(prs)
    add_artifacts_slide(prs)
    add_limitations_slide(prs)
    add_perspectives_slide(prs)
    add_conclusion_slide(prs)
    add_thank_you_slide(prs)
    
    # Sauvegarde
    output_path = "presentation_soutenance.pptx"
    prs.save(output_path)
    print(f"✅ Présentation sauvegardée : {output_path}")
    print(f"📊 Nombre de diapositives : {len(prs.slides)}")

if __name__ == "__main__":
    generate_presentation()
