import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from utils import functions as f
import concurrent.futures
import time

# Chemin vers les fichiers Excel
path_rail = "data/list_train_stations_2.xlsx"
path_airports = "data/list_airports.xlsx"
path_ports = "data/list_ports.xlsx"
path_river_ports = "data/list_river_ports.xlsx"
path_maritime_route = "data/maritime_route.xlsx"
path_river_routes = "data/river_routes.xlsx"

def load_nodes():
    # Charger toutes les coordonnées en une fois (pour éviter de les chercher individuellement)
    df_coords = pd.read_excel("data/location_coordinates.xlsx")
    
    # Charger les fichiers Excel en parallèle
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_sheets_rail = executor.submit(pd.read_excel, path_rail, sheet_name=None)
        future_df_airports = executor.submit(pd.read_excel, path_airports)
        future_df_ports = executor.submit(pd.read_excel, path_ports)
        future_sheets_ports = executor.submit(pd.read_excel, path_river_ports, sheet_name=None)
        future_df_maritime_route = executor.submit(pd.read_excel, path_maritime_route)
        future_sheets_river = executor.submit(pd.read_excel, path_river_routes, sheet_name=None)

        # Récupérer les résultats
        sheets_rail = future_sheets_rail.result()
        df_airports = future_df_airports.result()
        df_ports = future_df_ports.result()
        sheets_ports = future_sheets_ports.result()
        df_maritime_route = future_df_maritime_route.result()
        sheets_river = future_sheets_river.result()
    
    # Extraire les coordonnées pour les stations ferroviaires
    stations = set()
    for sheet_name, df in sheets_rail.items():
        stations.update(df['Ville'])
    stations = list(stations)

    # Utilisation de la recherche vectorisée pour obtenir les coordonnées des stations
    station_coords = {station: f.get_coordinates_from_name(f"Train station of {station}", df_coords) for station in stations}

    # Obtenir les aéroports et leurs coordonnées
    airports = df_airports['Airports'].tolist()
    airports_coords = {airport: f.get_coordinates_from_name(airport, df_coords) for airport in airports}

    # Obtenir les ports maritimes et leurs coordonnées
    ports = df_ports['Ports'].tolist()
    ports_coords = {port: f.get_coordinates_from_name(port, df_coords) for port in ports}

    # Obtenir les ports fluviaux et leurs coordonnées
    river_ports = set()
    for sheet_name, df in sheets_ports.items():
        river_ports.update(df['Ports'])
    river_ports = np.unique(list(river_ports))
    river_ports_coords = {port: f.get_coordinates_from_name(f"Fluvial port of {port}", df_coords) for port in river_ports}

    # Traitement des données de la route maritime
    df_maritime_route[['Latitude', 'Longitude']] = df_maritime_route['Point'].str.extract(r'([0-9°\'"]+[NS])\s+([0-9°\'"]+[EW])')
    df_maritime_route['Latitude'] = df_maritime_route['Latitude'].apply(f.dms_to_decimal)
    df_maritime_route['Longitude'] = df_maritime_route['Longitude'].apply(f.dms_to_decimal)
    maritime_waypoints = df_maritime_route[["Latitude", "Longitude"]].apply(tuple, axis=1).tolist()

        # Créer un dictionnaire contenant tous les lieux
    nodes = {
        "railway_stations": [],
        "airports": [],
        "ports": [],
        "river_ports": []
    }

    # Ajouter les coordonnées des gares, aéroports, ports et ports fluviaux dans nodes
    for category, coords in zip(["railway_stations", "airports", "ports", "river_ports"],
                                [station_coords, airports_coords, ports_coords, river_ports_coords]):
        for place, (lat, lng) in coords.items():
            if lat is not None and lng is not None:
                nodes[category].append((place, lat, lng))

    return nodes, sheets_rail, sheets_ports, sheets_river, maritime_waypoints

def generate_graph_and_routes(depart, destination, qty, selected_category, nodes, sheets_rail, sheets_ports, sheets_river, maritime_waypoints):

    # Définir les vitesses moyennes (en km/h)
    speeds = {
        "road": 80,
        "rail": 60,
        "sea": 25,
        "air": 900,
        "river": 15
    }

    # Définir les délais supplémentaires (en heures)
    delays = {
        "road": 0,
        "rail": 3,
        "sea": 24,
        "air": 2,
        "river": 1
    }

    # Facteurs d'émissions (kg CO2 par tonne-km)
    emission_factors = {
        "general": {
            "road": 0.1,
            "rail": 0.03,
            "sea": 0.005,
            "air": 1,
            "river": 0.02
        },
        "refrigerated": {
            "road": 0.15,
            "rail": 0.045,
            "sea": 0.007,
            "air": 1.2,
            "river": 0.03
        }
    }

    # Coûts par tonne-km (EUR)
    price_factors = {
        "general": {
            "road": 0.1,
            "rail": 0.02,
            "sea": 0.003,
            "air": 0.19,
            "river": 0.02
        },
        "refrigerated": {
            "road": 0.12,
            "rail": 0.03,
            "sea": 0.005,
            "air": 0.22,
            "river": 0.025
        }
    }

    # Ajouter le point de départ et d'arrivée dans nodes
    start_coords = f.get_gps_coordinates(depart)
    end_coords = f.get_gps_coordinates(destination)
    nodes["others"] = [(depart, *start_coords), (destination, *end_coords)]
    print(nodes["others"])

    input_coordinates = (start_coords, end_coords)

    # Créer le graphe comme dans ton code
    start_time = time.time()
    G = f.create_graph(nodes, speeds, delays, emission_factors, price_factors, selected_category, qty, sheets_rail, sheets_ports, sheets_river, maritime_waypoints)
    end_time = time.time()
    
    # Calculer et afficher la durée
    execution_time = end_time - start_time
    print(f"Graphe initialisé en {execution_time:.2f} secondes !")
    print(G)
    
    # Calculer les chemins écolo, rapide, moins cher, optimal, etc.
    ecolo_path, eco_duration, eco_emissions, eco_price, eco_distance = f.find_ecolo_path(G, depart, destination)
    fastest_path, fast_duration, fast_emissions, fast_price, fast_distance = f.find_fastest_path(G, depart, destination)
    cheapest_path, cheap_duration, cheap_emissions, cheap_price, cheap_distance = f.find_cheapest_path(G, depart, destination)
    road_route, road_duration, road_emissions, road_price, road_distance = f.find_fastest_road_path(depart, destination, speeds, emission_factors, price_factors, selected_category, qty)
    optimal_path, optimal_duration, optimal_emissions, optimal_price, optimal_distance = f.find_optimal_path(G, depart, destination)

    optimal_is_road, cheap_is_road, fast_is_road, eco_is_road = False, False, False, False
    # Vérifier si le chemin "road only" est meilleur selon tous les critères
    if sum(road_duration.values()) < sum(optimal_duration.values()) and road_emissions < optimal_emissions and road_price < optimal_price:
        optimal_path = road_route
        optimal_distance = road_distance
        optimal_duration = road_duration
        optimal_emissions = road_emissions
        optimal_price = road_price
        optimal_is_road = True

    # Vérifier si le chemin "cheap" est meilleur selon tous les critères
    if road_price < cheap_price:
        cheapest_path = road_route
        cheap_distance = road_distance
        cheap_duration = road_duration
        cheap_emissions = road_emissions
        cheap_price = road_price
        cheap_is_road = True

    # Vérifier si le chemin "fast" est meilleur selon tous les critères
    if sum(road_duration.values()) < sum(fast_duration.values()):
        fastest_path = road_route
        fast_distance = road_distance
        fast_duration = road_duration
        fast_emissions = road_emissions
        fast_price = road_price
        fast_is_road = True
        print("La route est le chemin rapide")

    # Vérifier si le chemin "eco" est meilleur selon tous les critères
    if road_emissions < eco_emissions:
        ecolo_path = road_route
        eco_distance = road_distance
        eco_duration = road_duration
        eco_emissions = road_emissions
        eco_price = road_price
        eco_is_road = True
    

    # Calcul des distances et émissions par mode pour chaque chemin
    distance_by_mode = {
        "eco": eco_distance,
        "fast": fast_distance,
        "cheap": cheap_distance,
        "road": road_distance,
        "optimal": optimal_distance
    }

    duration_by_mode = {
        "eco": eco_duration,
        "fast": fast_duration,
        "cheap": cheap_duration,
        "road": road_duration,
        "optimal": optimal_duration
    }

    emissions_by_mode = {
        "eco": round(eco_emissions, 2),
        "fast": round(fast_emissions, 2),
        "cheap": round(cheap_emissions, 2),
        "road": round(road_emissions, 2),
        "optimal": round(optimal_emissions, 2)
    }

    price_by_mode = {
        "eco": eco_price,
        "fast": fast_price,
        "cheap": cheap_price,
        "road": road_price,  
        "optimal": optimal_price
    }

    return G, ecolo_path, fastest_path, cheapest_path, road_route, optimal_path, eco_is_road, fast_is_road, cheap_is_road, optimal_is_road, distance_by_mode, emissions_by_mode, price_by_mode, duration_by_mode, input_coordinates

    
    
# Cette fonction crée la carte Folium avec le chemin choisi
def create_folium_map(path_type, start, end, ecolo_path, fastest_path, cheapest_path, road_route, optimal_path, eco_is_road, fast_is_road, cheap_is_road, optimal_is_road, G, nodes, maritime_waypoints, sheets_river, sheets_ports, input_coordinates):
    # Afficher le chemin correspondant
    if path_type == 'eco':
        # Chemin écologique
        map_route = f.add_path_to_map(ecolo_path, start, end, G, nodes, maritime_waypoints, sheets_river, sheets_ports, input_coordinates, eco_is_road)
    elif path_type == 'fast':
        # Chemin rapide
        map_route = f.add_path_to_map(fastest_path, start, end, G, nodes, maritime_waypoints, sheets_river, sheets_ports, input_coordinates, fast_is_road)
    elif path_type == 'cheap':
        # Chemin le moins cher
        map_route = f.add_path_to_map(cheapest_path, start, end, G, nodes, maritime_waypoints, sheets_river, sheets_ports, input_coordinates, cheap_is_road)
    elif path_type == 'road':
        map_route = f.add_path_to_map(road_route, start, end, G, nodes, maritime_waypoints, sheets_river, sheets_ports, input_coordinates, path_is_road = True)
    elif path_type == 'optimal':
        # Chemin optimal
        map_route = f.add_path_to_map(optimal_path, start, end, G, nodes, maritime_waypoints, sheets_river, sheets_ports, input_coordinates, optimal_is_road)

    return map_route

# Fonction de traduction des modes de transport
def translate_mode(mode):
    translations = {
        "road": "Route",
        "rail": "Chemin de fer",
        "sea": "Voie maritime",
        "air": "Voie aérienne",
        "river": "Voie fluviale"
    }
    return translations.get(mode, mode)

# Fonction de traduction des types de chemin
def translate_path_type(path_type):
    path_translations = {
        "eco": "Itinéraire minimisant les émissions de CO2",
        "fast": "Itinéraire le plus rapide",
        "cheap": "Itinéraire le plus économique",
        "road": "Itinéraire route",
        "optimal": "Itinéraire optimal"
    }
    return path_translations.get(path_type, path_type)

# ### Choisissez votre itinéraire logistique idéal 🚛🚂✈️
def generate_explanation(path_type):
    if path_type == "eco":
        return f"""
        ### Choisissez votre itinéraire logistique idéal 🚛🚂✈️
        **Itinéraire écologique** 🌱  
        Priorisez l'environnement avec un itinéraire réduisant au maximum les émissions de CO2.  
        Les modes comme le fret maritime ou fluvial sont privilégiés, grâce à leur faible impact par tonne transportée.  
        Idéal pour réduire votre empreinte carbone, mais prévoyez un délai plus long dû aux distances et changements de mode.
        """
    if path_type == "fast":
        return f"""
        ### Choisissez votre itinéraire logistique idéal 🚛🚂✈️
        **Itinéraire rapide** ⚡  
        Gagnez du temps avec le trajet le plus rapide !  
        Lorsque possible, l'avion est recommandé pour sa vitesse, bien qu'il soit plus coûteux et émetteur (200 fois plus que le bâteau !).  
        Sinon, le camion reste une alternative rapide et flexible pour une livraison efficace.
        """
    if path_type == "cheap":
        return f"""
        ### Choisissez votre itinéraire logistique idéal 🚛🚂✈️
        **Itinéraire économique** 💰  
        Maîtrisez vos coûts grâce au trajet le moins cher.  
        Souvent, cet itinéraire coïncide avec l'option écologique, les modes les moins émetteurs étant aussi les plus économiques, comme le fret maritime.
        """
    if path_type == "road":
        return f"""
        ### Choisissez votre itinéraire logistique idéal 🚛🚂✈️
        **Itinéraire routier** 🚛  
        Suivez l'itinéraire classique proposé par Google Maps.  
        Parfait pour la flexibilité et les courtes distances, il reste toutefois énergivore : transporter un conteneur en camion consomme bien plus qu'en train ou en bateau.
        """
    if path_type == "optimal":
        return f"""
        ### Choisissez votre itinéraire logistique idéal 🚛🚂✈️
        **Itinéraire optimal** 🎯  
        Trouvez le meilleur équilibre entre coût, vitesse et impact environnemental.  
        Cet itinéraire privilégie souvent le train, une solution idéale combinant rapidité, prix abordable et faible émission de CO2.
        """

# ### Recommandation personnalisée d'itinéraire 🌍📦
def generate_recommendation(emissions_by_mode, price_by_mode, duration_by_mode):
    emissions_road = emissions_by_mode["road"]
    price_road = price_by_mode["road"]
    duration_road = sum(duration_by_mode["road"].values())

    emissions_eco = emissions_by_mode["eco"]
    price_eco = price_by_mode["eco"]
    duration_eco = sum(duration_by_mode["eco"].values())

    emissions_optimal = emissions_by_mode["optimal"]
    price_optimal = price_by_mode["optimal"]
    duration_optimal = sum(duration_by_mode["optimal"].values())

    # Cas 1 : Préférence pour l'itinéraire routier
    if (
        emissions_eco/emissions_road > 0.7 and
        price_eco/price_road > 0.7
    ):
        return f"""
        ### Préconisation ETYO 
        **Restez sur l'itinéraire routier** 🚛  
        Les alternatives n'apportent pas de bénéfices significatifs en termes de coûts ou d'impact écologique.  
        Optez pour la route, une solution simple et flexible adaptée à vos besoins.
        """

    # Cas 2 : Préférence pour l'itinéraire optimal
    if (
        emissions_eco/emissions_road < 0.7 and
        price_eco/price_road < 0.7 and
        duration_eco/duration_eco > 2
    ):
        return f"""
        ### Préconisation ETYO 
        **Choisissez l'itinéraire optimal** 🎯  
        Combinez gains écologiques et économiques tout en limitant les pertes de temps.  
        Grâce à la multimodalité, profitez d'une solution équilibrée et durable.
        """

    # Cas 3 : Préférence pour l'itinéraire écologique
    if (
        emissions_eco/emissions_road < 0.7 and
        price_eco/price_road < 0.7 and
        duration_eco/duration_eco < 2
    ):
        return f"""
        ### Préconisation ETYO 
        **Adoptez l'itinéraire écologique** 🌱  
        Réduisez vos coûts tout en minimisant vos émissions de CO2.  
        Un choix responsable sans trop sacrifier votre temps de trajet !
        """




