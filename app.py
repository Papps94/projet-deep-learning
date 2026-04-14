import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu

# 1. Configuration de la page
st.set_page_config(
    page_title="CNN vs ViT - Projet Deep Learning",
    layout="wide"
)

# 2. Les différentes pages
def page_accueil():
    st.title("CNN vs ViT : Comparaison d'Architectures")
    
    st.info("""
    **Contexte du Projet :** Le CNN et le ViT sont deux algorithmes d'**Intelligence Artificielle** (Deep Learning). 
    Leur mission ici est la **classification d'images** : on leur donne une image, et ils doivent deviner seuls à quelle catégorie elle appartient.
    """)
    
    st.markdown("""
    Bienvenue sur notre tableau de bord interactif. 
    L'objectif de ce projet est de comparer ces deux géants de la vision par ordinateur : l'approche classique (CNN) et la nouvelle génération (Vision Transformer).
    
    Plutôt que de les tester sur des bases de données classiques, nous avons créé nos propres images synthétiques pour comprendre **comment** ces modèles réfléchissent, ce qu'ils regardent réellement, et quelles sont leurs limites fondamentales.
    """)
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.header("Le Modèle CNN")
        st.markdown("""
        **Comment ça marche ?**
        Le CNN parcourt l'image région par région, de gauche à droite, à la manière d'une fenêtre glissante. Il ne voit jamais l'image en entier d'un seul coup.
        
        * **Sa grande force :** Il est excellent pour repérer des détails très proches les uns des autres (des textures, des rayures, des contours). Il apprend très vite, même avec un faible volume de données.
        * **Sa faiblesse :** Il a beaucoup de mal à faire le lien entre deux objets qui sont très éloignés sur une grande image, car son champ de vision est restreint.
        
        *(Détails techniques : Conv1(16), Conv2(32), MaxPool 2x2, Dense 128)*
        """)
        
    with col2:
        st.header("Le Modèle ViT")
        st.markdown("""
        **Comment ça marche ?**
        Le ViT découpe l'image en plusieurs sous-régions (des "patches") et les observe toutes simultanément. Grâce à son mécanisme d'Attention, il cherche à comprendre comment chaque région est liée aux autres, peu importe la distance qui les sépare.
        
        * **Sa grande force :** Il comprend parfaitement le contexte global d'une image et peut lier des informations très éloignées spatialement.
        * **Sa faiblesse :** Contrairement au CNN, il n'a pas d'instinct de base pour la proximité des pixels. Il doit tout déduire lui-même, ce qui exige une quantité massive de données pour commencer à être performant.
        
        *(Détails techniques : Patches 8x8, Embed 64, 4 têtes, 2 couches)*
        """)

def page_datasets():
    st.title("Méthodologie & Datasets")
    st.markdown("""
    ### L'approche Synthétique
    Nous avons généré nos propres images (32x32 pixels) afin d'isoler des caractéristiques spécifiques. Cela nous permet de tester les modèles avec deux tâches distinctes :
    * **Une tâche locale (Texture) :** Conçue pour exploiter les filtres locaux du CNN.
    * **Une tâche globale (Distance) :** Conçue pour exploiter la vision d'ensemble du ViT.
    """)
    st.divider()
    
    st.markdown("### Nos jeux de données")
    try:
        st.image("images/datasets.png", caption="À gauche : Texture (avantage CNN). À droite : Distance (avantage ViT).", use_container_width=True)
    except:
        st.warning("Image introuvable : Vérifiez que 'datasets.png' ou 'datasets.jpg' est bien dans le dossier 'images'.")

def page_resultats():
    st.title("Exploration des Résultats")
    st.markdown("### Choisissez une expérience à analyser :")
    
    experience = st.selectbox(
        "Expérience :",
        ["Exp 1 : 1k images (Volume initial)", "Exp 3 : 10k images (Volume massif)", "Exp 4 : 64x64 (Haute Résolution)", "Exp 5 : 10 classes (Robustesse)"]
    )
    st.divider()

    if experience == "Exp 1 : 1k images (Volume initial)":
        st.subheader("Résultats avec 1000 images")
        try:
            st.image("images/graphe_1k.png", use_container_width=True)
        except:
            st.info("Le graphique 'graphe_1k.png' s'affichera ici.")
        st.success("""
        **Observation :** Sur la tâche Distance (à droite), le CNN atteint 100% de précision presque instantanément. Le ViT stagne autour de 50% pendant de nombreuses époques avant d'apprendre.
        
        **Analyse :** Sur un petit volume de données (1000 images), le biais inductif du CNN lui donne un avantage écrasant. Le ViT manque d'exemples pour structurer sa compréhension visuelle de manière autonome.
        """)

    elif experience == "Exp 3 : 10k images (Volume massif)":
        st.subheader("Résultats avec 10 000 images")
        try:
            st.image("images/graphe_10k.png", use_container_width=True)
        except:
            st.info("Le graphique 'graphe_10k.png' s'affichera ici.")
        st.success("""
        **Observation :** Cette fois-ci, le comportement du ViT change drastiquement. Il suit la courbe du CNN de très près et atteint 98.5% d'efficacité rapidement.
        
        **Analyse :** Avec un volume de données dix fois supérieur, le ViT compense son absence de biais inductif. La limite observée à l'expérience 1 était strictement liée à la quantité de données, et non à une limitation de l'architecture.
        """)

    elif experience == "Exp 4 : 64x64 (Haute Résolution)":
        st.subheader("Résultats en 64x64")
        try:
            st.image("images/graphe_64.png", use_container_width=True)
        except:
            st.info("Le graphique 'graphe_64.png' s'affichera ici.")
        st.success("""
        **Observation :** Les performances s'inversent. Le ViT atteint 99.5%, tandis que le CNN peine à dépasser les 92.5%.
        
        **Analyse :** En doublant la taille de l'image (de 32x32 à 64x64), la distance entre les objets augmente. Les filtres du CNN ne sont plus assez larges pour relier les coins de l'image. Le ViT, avec son attention globale, reste insensible à cette distance.
        """)

    elif experience == "Exp 5 : 10 classes (Robustesse)":
        st.subheader("Résultats Multi-classes")
        try:
            st.image("images/graphe_10classes.png", use_container_width=True)
        except:
            st.info("Le graphique 'graphe_10classes.png' s'affichera ici.")
        st.success("""
        **Observation :** Malgré la complexité accrue (passage de 2 à 10 motifs possibles), les deux architectures parviennent à converger vers 100%.
        
        **Analyse :** Cette expérience démontre la robustesse des deux modèles. Le CNN et le ViT conservent une excellente capacité de généralisation une fois la tâche fondamentale acquise.
        """)

def page_bilan():
    st.title("Bilan Stratégique & Conclusion")
    
    st.markdown("### Le Verdict des Performances")
    col1, col2, col3 = st.columns(3)
    col1.metric(label="CNN (Petits Volumes)", value="Vainqueur", delta="Apprentissage quasi-immédiat")
    col2.metric(label="ViT (Grands Volumes)", value="Vainqueur", delta="Précision Maximale à terme")
    col3.metric(label="ViT (Grandes Images)", value="Vainqueur", delta="Vision Globale imbattable")
    
    st.divider()
    
    st.markdown("### Tableau Récapitulatif")
    data = {
        "Expérience": ["Texture (1k)", "Distance (1k)", "Distance (10k)", "Haute Résol. (64x64)", "10 Classes"],
        "Score CNN": ["100%", "100%", "100%", "92.5%", "100%"],
        "Score ViT": ["100%", "51% ➔ 100%", "98.5%", "99.5%", "100%"],
        "Analyse": ["Égalité parfaite", "Le CNN gagne du temps", "Le ViT rattrape son retard", "Le ViT surpasse le CNN", "Égalité (Robustesse validée)"]
    }
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()

    st.markdown("### Recommandations Architecturales")
    st.write("Sur la base de ces observations, voici les recommandations pour le déploiement de ces architectures :")
    
    colA, colB = st.columns(2)
    with colA:
        st.success("""
        **Privilégiez le CNN si :**
        * Le volume de données disponibles est limité.
        * L'objectif est d'identifier des détails ou des motifs très locaux.
        * Les ressources de calcul sont contraintes et nécessitent une convergence rapide.
        """)
    with colB:
        st.info("""
        **Privilégiez le ViT si :**
        * Le volume de données d'entraînement est massif.
        * La tâche requiert une compréhension globale de l'image ou la mise en relation d'objets distants.
        * La puissance de calcul (serveurs GPU) permet un apprentissage long.
        """)

# 3. Sidebar et Routage avec le nouveau menu
with st.sidebar:
    st.title("Sommaire")
    choix_page = option_menu(
        menu_title=None,
        options=["Accueil & Architectures", "Méthodologie & Datasets", "Exploration des Résultats", "Bilan Stratégique"],
        icons=["house", "database", "graph-up", "check2-square"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#FAFAFA", "font-size": "16px"},
            "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#1E2129"},
            "nav-link-selected": {"background-color": "#2E86C1", "color": "white"},
        }
    )

# 4. Logique de routage
if choix_page == "Accueil & Architectures":
    page_accueil()
elif choix_page == "Méthodologie & Datasets":
    page_datasets()
elif choix_page == "Exploration des Résultats":
    page_resultats()
elif choix_page == "Bilan Stratégique":
    page_bilan()

st.sidebar.markdown("---")
st.sidebar.info("Projet réalisé par Faye Papp ")