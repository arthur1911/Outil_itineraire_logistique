import streamlit as st
from utils.main_code import generate_graph_and_routes, load_nodes, create_folium_map, translate_mode, translate_path_type, generate_recommendation, generate_explanation
import pandas as pd
from datetime import datetime
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import plotly.express as px
from streamlit import markdown
import locale
from datetime import datetime
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
locale.setlocale(locale.LC_ALL, 'fr_FR.UTF-8')

# Configuration de Streamlit
st.set_page_config(page_title="Itinéraire Logistique", layout="centered")

# Charger les informations d'identification Google API
creds = Credentials.from_service_account_file("credentials.json", scopes=["https://www.googleapis.com/auth/spreadsheets"])

# Connectez-vous à l'API Google Sheets
service = build("sheets", "v4", credentials=creds)

# ID et plage de votre Google Sheet
spreadsheet_id = "1sBc1qsi4MABwBsMGQF-OoOj6h3SK1QPYzW6R9sX5dVk"
range_ = "'Infos'!A:D"  # Adaptez en fonction de votre structure

# Fonction pour ajouter des données à Google Sheets
def ajouter_donnees_google_sheets(nom, prenom, email, societe):
    # Ajoutez la date de soumission
    date_soumission = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    # Préparez les données
    values = [[nom, prenom, email, societe, date_soumission]]
    body = {"values": values}
    # Ajoutez-les à la Google Sheet
    service.spreadsheets().values().append(
        spreadsheetId=spreadsheet_id,
        range=range_,
        valueInputOption="RAW",
        body=body
    ).execute()

# Initialiser les variables de session
if "trajet_genere" not in st.session_state:
    st.session_state.trajet_genere = False
if "popup_remplie" not in st.session_state:
    st.session_state.popup_remplie = False
if "bouton_appuye" not in st.session_state:
    st.session_state.bouton_appuye = False

# Fonction de formatage localisé
def format_number_locale(value):
    if isinstance(value, int):
        return locale.format_string("%d", value, grouping=True)
    elif isinstance(value, float):
        return locale.format_string("%g", value, grouping=True)
    else:
        return str(value)

# Charger les données une fois lors de l'initialisation
nodes, sheets_rail, sheets_ports, sheets_river, maritime_waypoints = load_nodes()

# Titre de la page
st.title("Simulateur d'itinéraire logistique")

# Section de sélection des inputs dans la sidebar
st.sidebar.header("Paramètres")
depart = st.sidebar.text_input("Adresse de départ", value="3 rue de Stockholm, Paris")
destination = st.sidebar.text_input("Adresse de destination", value="SCALANDES, Mont de Marsan")
quantite = st.sidebar.number_input("Quantité en tonnes", min_value=1, value=1)

# Fonction de traduction des types de cargaison
def translate_cargo_type(cargo_type):
    cargo_translations = {
        "general": "Stockage à température ambiante",
        "refrigerated": "Stockage réfrigéré"
    }
    return cargo_translations.get(cargo_type, cargo_type)

# Appliquer la traduction des options de cargaison
cargo_options = ["general", "refrigerated"]
translated_cargo_options = [translate_cargo_type(option) for option in cargo_options]
selected_cargo_option = st.sidebar.selectbox("Type de cargaison", options=translated_cargo_options)
cargaison = cargo_options[translated_cargo_options.index(selected_cargo_option)]  # Convertir en valeur originale

# Bouton pour générer le trajet
if st.sidebar.button("Générer le trajet"):
    st.session_state.bouton_appuye = True

# Ajouter un point méthodologie dans la sidebar
with st.sidebar.expander("Lire les règles de calcul"):
    st.write("""
    **Règles de calcul :**
    
    - Les **émissions de CO2** sont basées sur les ratios donnés par la méthodologie 
    [GLEC](https://smartfreightcentre.org/en/our-programs/emissions-accounting/global-logistics-emissions-council/) 
    accrédités par le Smart Freight Center.
    - Les **coûts économiques** sont calculés à partir d'un rapport publié par 
    [Panteia](https://panteia.com/) pour l'[Institut néerlandais d'analyse des politiques de transport](https://english.kimnet.nl/).
    - Les **temps de trajet** sont estimés à partir des vitesses moyennes des différents modes de transport, 
    avec des délais ajoutés pour chaque changement de mode.
    - Les **distances** sont récupérées à partir de Google Maps via son [API](https://developers.google.com/maps?hl=fr) si possible, ou sont déterminées à partir 
    des distances à vol d'oiseau ajustées par un facteur tenant compte de l'irrégularité du chemin.
    - Les **itinéraires** sont calculés grâce à la [théorie des graphes](https://fr.wikipedia.org/wiki/Th%C3%A9orie_des_graphes) et l'[algorithme de Dijkstra](https://fr.wikipedia.org/wiki/Algorithme_de_Dijkstra), 
    prenant en compte 230 gares ferroviaires, 100 aéroports majeurs, 30 ports maritimes, et 20 ports fluviaux en Europe.
    """)


# Texte explicatif à afficher avant la génération du trajet
description_text = """
<span style="font-size: 1em;">
Ce <strong>simulateur</strong> permet de comparer différents <strong>itinéraires logistiques</strong> en termes de <strong>distance</strong>, <strong>coût</strong>, <strong>durée</strong> et <strong>émissions de CO₂</strong>. <br>
\nGrâce à cet outil, vous pouvez <strong>visualiser</strong> et <strong>évaluer</strong> <strong>l’impact environnemental</strong> de vos choix de transport en fonction de plusieurs options de chemins et types de cargaison.<br>
\nSélectionnez vos préférences de trajet et de chargement pour obtenir des informations précises, incluant les trajets routiers, ferroviaires, maritimes, fluviaux et aériens.<br>
\nPour des comparaisons plus avancées ou des options personnalisées, veuillez contacter notre <a href="https://www.etyo.com/accueil/contact/">support</a>.
</span>
"""

# Afficher le texte si le trajet n'a pas encore été généré
if 'graph_data' not in st.session_state or not st.session_state.graph_data:
    st.markdown(description_text, unsafe_allow_html=True)

# Initialiser le session state si nécessaire
if 'graph_data' not in st.session_state:
    st.session_state.graph_data = {}
if 'map_path' not in st.session_state:
    st.session_state.map_path = None
if 'path_type' not in st.session_state:
    st.session_state.path_type = 'road'

# Afficher le formulaire une fois que le trajet a été généré et si les données ne sont pas encore remplies
if st.session_state.bouton_appuye and not st.session_state.popup_remplie:
    st.info("Pour sécuriser vos données, veuillez rentrer vos informations personnelles.")
    with st.form("formulaire_utilisateur"):
        nom = st.text_input("Nom *")
        prenom = st.text_input("Prénom *")
        email = st.text_input("Email *")
        societe = st.text_input("Société (facultatif)")
        submitted = st.form_submit_button("Envoyer")

        if submitted:
            if nom and prenom and email:
                try:
                    ajouter_donnees_google_sheets(nom, prenom, email, societe)
                    st.session_state.popup_remplie = True
                    st.success("Merci pour vos informations ! Vous pouvez maintenant accéder au résultat du trajet.")
                except Exception as e:
                    st.error(f"Une erreur est survenue : {e}")
            else:
                st.error("Veuillez remplir tous les champs obligatoires (Nom, Prénom, Email).")

# Bouton pour générer le trajet
if st.session_state.bouton_appuye & st.session_state.popup_remplie:
    # Génération des données de chemin
    G, ecolo_path, fastest_path, cheapest_path, road_route, optimal_path, eco_is_road, fast_is_road, cheap_is_road, optimal_is_road, distance_by_mode, emissions_by_mode, price_by_mode, duration_by_mode, input_coordinates = generate_graph_and_routes(
        depart, destination, quantite, cargaison, nodes, sheets_rail, sheets_ports, sheets_river, maritime_waypoints)

    # Stocker les données dans le session state
    st.session_state.graph_data = {
        'depart': depart,
        'destination': destination,
        'qty': quantite,
        'cargaison': cargaison,
        'G': G,
        'ecolo_path': ecolo_path,
        'fastest_path': fastest_path,
        'cheapest_path': cheapest_path,
        'road_route': road_route,
        'optimal_path': optimal_path,
        'eco_is_road': eco_is_road, 
        'fast_is_road': fast_is_road,
        'cheap_is_road': cheap_is_road, 
        'optimal_is_road': optimal_is_road,
        'distance_by_mode': distance_by_mode,
        'emissions_by_mode': emissions_by_mode,
        'price_by_mode': price_by_mode,
        'duration_by_mode': duration_by_mode,
        'input_coordinates': input_coordinates
    }

    # Générer la carte initiale avec le path_type par défaut et la stocker dans session_state
    st.session_state.path_type = 'road'
    st.session_state.map_path = create_folium_map(
        st.session_state.path_type, depart, destination, ecolo_path, fastest_path, cheapest_path, road_route,
        optimal_path, eco_is_road, fast_is_road, cheap_is_road, optimal_is_road, G, nodes, maritime_waypoints, sheets_river, sheets_ports, input_coordinates
    )

    st.session_state.trajet_genere = True  # Indique que le trajet a été généré
    st.session_state.bouton_appuye = False


# Sélection du type de chemin
if st.session_state.trajet_genere:
    # Appliquer la traduction aux options du selectbox
    path_options = ["eco", "fast", "cheap", "road", "optimal"]
    translated_options = [translate_path_type(option) for option in path_options]
    selected_option = st.selectbox("Choisissez le type de chemin", options=translated_options, 
                                   index=path_options.index(st.session_state.path_type))
    path_type = path_options[translated_options.index(selected_option)]  # Convertir en valeur originale

    # Si le type de chemin a changé, mettre à jour la carte
    if path_type != st.session_state.path_type:
        st.session_state.path_type = path_type
        st.session_state.map_path = create_folium_map(
            st.session_state.path_type, st.session_state.graph_data['depart'], st.session_state.graph_data['destination'],
            st.session_state.graph_data['ecolo_path'], st.session_state.graph_data['fastest_path'], 
            st.session_state.graph_data['cheapest_path'], st.session_state.graph_data['road_route'],
            st.session_state.graph_data['optimal_path'], st.session_state.graph_data['eco_is_road'],
            st.session_state.graph_data['fast_is_road'], st.session_state.graph_data['cheap_is_road'],
            st.session_state.graph_data['optimal_is_road'], st.session_state.graph_data['G'], nodes, 
            maritime_waypoints, sheets_river, sheets_ports, st.session_state.graph_data['input_coordinates']
        )

    # Extraire les indicateurs pour le type de chemin sélectionné
    distance_selected = st.session_state.graph_data['distance_by_mode'][st.session_state.path_type]
    emissions_selected = st.session_state.graph_data['emissions_by_mode'][st.session_state.path_type]
    price_selected = st.session_state.graph_data['price_by_mode'][st.session_state.path_type]
    duration_selected = st.session_state.graph_data['duration_by_mode'][st.session_state.path_type]

    road_distance = sum(st.session_state.graph_data['distance_by_mode']["road"].values())
    road_emissions = st.session_state.graph_data['emissions_by_mode']["road"]
    road_price = st.session_state.graph_data['price_by_mode']["road"]
    road_duration = sum(st.session_state.graph_data['duration_by_mode']["road"].values())

    selected_distance = sum(distance_selected.values())
    selected_emissions = emissions_selected
    selected_price = price_selected
    selected_duration = sum(duration_selected.values())

    # Calculer les différences
    diff_distance = selected_distance - road_distance
    diff_emissions = selected_emissions - road_emissions
    diff_price = selected_price - road_price
    diff_duration = selected_duration - road_duration

    # Fonction pour formater les différences avec flèches, couleur et signe +
    def format_difference(diff_value, unit=""):
        if diff_value > 0:
            color = "red"
            arrow = "⬆️"
            sign = "+"
        else:
            color = "green"
            arrow = "⬇️"
            sign = "-"
        # Utilise format_number_locale pour formater la différence avec le format localisé
        formatted_value = format_number_locale(abs(round(diff_value)))  # `abs` pour éviter les signes répétitifs
        return f"<span style='color: {color};'>{arrow} {sign}{formatted_value}{unit}</span>"

    # Encapsuler les indicateurs dans un seul bloc de `st.markdown` pour garantir qu'ils apparaissent dans le cadre
    st.markdown("""
    <div style="
        border: 2px solid #ccc; 
        padding: 20px; 
        border-radius: 10px; 
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1); 
        margin-bottom: 20px;
        display: flex;
        justify-content: space-around;
    ">
        <div style="text-align: center;">
            <h4 style="font-size: 0.9em; color: #555;">Distance Totale (km)</h4>
            <p style="font-size: 2em; font-weight: bold; color: #333;">{distance}</p>
            <p style="font-size: 0.9em;">{distance_diff}</p>
        </div>
        <div style="text-align: center;">
            <h4 style="font-size: 0.9em; color: #555;">Emissions de CO2 (kg)</h4>
            <p style="font-size: 2em; font-weight: bold; color: #333;">{emissions}</p>
            <p style="font-size: 0.9em;">{emissions_diff}</p>
        </div>
        <div style="text-align: center;">
            <h4 style="font-size: 0.9em; color: #555;">Coût (€)</h4>
            <p style="font-size: 2em; font-weight: bold; color: #333;">{price}</p>
            <p style="font-size: 0.9em;">{price_diff}</p>
        </div>
        <div style="text-align: center;">
            <h4 style="font-size: 0.9em; color: #555;">Durée Totale (h)</h4>
            <p style="font-size: 2em; font-weight: bold; color: #333;">{duration}</p>
            <p style="font-size: 0.9em;">{duration_diff}</p>
        </div>
    </div>
    """.format(
        distance=format_number_locale(round(selected_distance)),
        distance_diff = "" if path_type == "road" else format_difference(diff_distance, " km"),
        emissions=format_number_locale(round(selected_emissions)),
        emissions_diff= "" if path_type == "road" else format_difference(diff_emissions, " kg"),
        price=format_number_locale(round(selected_price)),
        price_diff= "" if path_type == "road" else format_difference(diff_price, " €"),
        duration=format_number_locale(round(selected_duration)),
        duration_diff= "" if path_type == "road" else format_difference(diff_duration, " h")
    ), unsafe_allow_html=True)

    # Fermer le div pour le cadre
    st.markdown("</div>", unsafe_allow_html=True)

    # Préparer les données pour le camembert interactif
    labels = [translate_mode(label) for label in duration_selected.keys()]
    sizes = list(duration_selected.values())
    colors_path = {"road": "orange", "rail": "green", "sea": "blue", "air": "red", "river": "cyan"}
    colors = [colors_path.get(label, "gray") for label in labels]  # Utiliser les couleurs définies

    # Créer un camembert interactif avec Plotly
    fig = px.pie(
        names=labels,
        values=sizes,
        color=labels,
        color_discrete_map={translate_mode(k): v for k, v in colors_path.items()}  # Appliquer les couleurs aux labels traduits
    )

    # Configurer des détails d'affichage
    fig.update_traces(textinfo='percent', hoverinfo='label+percent+value')
    fig.update_layout(transition_duration=500)  # Animation de transition

    # Disposer la carte et la légende côte à côte
    col5, col6 = st.columns([5, 1])

    with col5:
        # Titre de la carte centré horizontalement
        st.markdown("<h3 style='text-align: center;'>Carte de l'itinéraire</h3>", unsafe_allow_html=True)
        st_folium(st.session_state.map_path, width=700, height=500)

    with col6:
        # Légende centrée verticalement et avec une police réduite
        st.markdown("""
        <div style="
            display: flex; 
            flex-direction: column; 
            justify-content: center; 
            align-items: center; 
            height: 500px;  /* Aligne la légende par rapport à la hauteur de la carte */
            ">
            <ul style="list-style-type: none; padding-left: 0; text-align: left; font-size: 0.7em;">
                <li style="font-size: 1em;"><span style="color: orange; font-weight: bold;">●</span> Route</li>
                <li style="font-size: 1em;"><span style="color: green; font-weight: bold;">●</span> Chemin de fer</li>
                <li style="font-size: 1em;"><span style="color: blue; font-weight: bold;">●</span> Voie maritime</li>
                <li style="font-size: 1em;"><span style="color: red; font-weight: bold;">●</span> Voie aérienne</li>
                <li style="font-size: 1em;"><span style="color: cyan; font-weight: bold;">●</span> Voie fluviale</li>
            </ul>
        </div>

        """, unsafe_allow_html=True)


    st.markdown("<h3 style='text-align: center;'>Durée du trajet par mode de transport</h3>", unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)

    # Générer et afficher la recommandation en bas de page
    explanation_text = generate_explanation(path_type)
    st.markdown(explanation_text, unsafe_allow_html=True)

    # Générer et afficher la recommandation en bas de page
    recommendation_text = generate_recommendation(st.session_state.graph_data['emissions_by_mode'], st.session_state.graph_data['price_by_mode'], st.session_state.graph_data['duration_by_mode'])
    st.markdown(recommendation_text, unsafe_allow_html=True)


    
