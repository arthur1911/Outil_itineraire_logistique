import googlemaps
from geopy.distance import geodesic
import networkx as nx
import numpy as np
import folium
import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from scipy.spatial import KDTree
import time
import math

# API Key Google Maps
gmaps = googlemaps.Client(key='AIzaSyCbI43yqEFGUioCA-ySgr538q7xIBuOtn8')

# Function to get GPS coordinates
def get_gps_coordinates(place):
    try:
        geocode_result = gmaps.geocode(place)
        if geocode_result:
            return geocode_result[0]['geometry']['location']['lat'], geocode_result[0]['geometry']['location']['lng']
        return None, None
    except googlemaps.exceptions.HTTPError as e:
        print(f"HTTP Error for place {place}: {e}")
        return None, None
    except Exception as e:
        print(f"Error for place {place}: {e}")
        return None, None
    
    
# Fonction pour obtenir les coordonnées d'un lieu
def get_coordinates_from_name(location_name, df_coords):
    coords = df_coords[df_coords['Location'] == location_name]
    if not coords.empty:
        lat = coords['Latitude'].values[0]
        lng = coords['Longitude'].values[0]
        return (lat, lng)
    else:
        return None


# Fonction pour convertir les coordonnées en format DMS en décimal
def dms_to_decimal(dms_str):
    dms_str = dms_str.strip()
    degrees, minutes, seconds, direction = re.split('[°\'"]', dms_str)[:4]
    decimal = float(degrees) + float(minutes) / 60 + float(seconds) / 3600
    if direction in ['S', 'W']:
        decimal *= -1
    return decimal

# Function to calculate distance between two points
def calculate_distance(lat1, lng1, lat2, lng2):
    return geodesic((lat1, lng1), (lat2, lng2)).kilometers

def haversine(lat1, lng1, lat2, lng2):
    # Convertir les coordonnées en radians
    lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])

    # Différences entre les coordonnées
    dlat = lat2 - lat1
    dlng = lng2 - lng1

    # Formule de Haversine
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Rayon de la Terre en kilomètres (moyenne)
    R = 6371.0
    return R * c


# Fonction pour trouver le point le plus proche sur la voie maritime
def find_closest_maritime_point(lat, lng, maritime_waypoints):
    closest_point = None
    min_distance = float('inf')
    for waypoint in maritime_waypoints:
        distance = haversine(lat, lng, waypoint[0], waypoint[1])
        if distance < min_distance:
            min_distance = distance
            closest_point = waypoint
    return closest_point, min_distance

# Fonction pour trouver le point le plus proche sur la voie maritime
def find_closest_river_point(lat, lng, sheets_river):
    closest_point = None
    min_distance = float('inf')
    for sheet_name, df_river_route in sheets_river.items():
        # Séparer les colonnes de latitude et de longitude
        df_river_route[['Latitude', 'Longitude']] = df_river_route['Point'].str.extract(r'([0-9°\'"]+[NS])\s+([0-9°\'"]+[EW])')

        # Convertir les coordonnées en décimal
        df_river_route['Latitude'] = df_river_route['Latitude'].apply(dms_to_decimal)
        df_river_route['Longitude'] = df_river_route['Longitude'].apply(dms_to_decimal)

        # Convertir en liste de tuples (latitude, longitude)
        river_waypoints = df_river_route[["Latitude", "Longitude"]].apply(tuple, axis=1).tolist()

        for waypoint in river_waypoints:
            distance = haversine(lat, lng, waypoint[0], waypoint[1])
            if distance < min_distance:
                min_distance = distance
                closest_point = waypoint
        return closest_point, min_distance

# Fonction pour vérifier si deux gares sont dans le même onglet
def are_stations_in_same_sheet(station1, station2, sheets):
    for sheet_name, df in sheets.items():
        if station1 in df.values and station2 in df.values:
            return True
    return False

# Function to get driving distance
def get_driving_distance(start, end):
    directions = gmaps.directions(start, end, mode="driving")
    if directions:
        distance = directions[0]['legs'][0]['distance']['value'] / 1000  # Convert to kilometers
        return distance
    return None

# Function to get driving route
def get_driving_route(start, end):
    directions = gmaps.directions(start, end, mode="driving", avoid="ferries")
    if directions:
        steps = directions[0]['legs'][0]['steps']
        route = [(step['start_location']['lat'], step['start_location']['lng']) for step in steps]
        route.append((steps[-1]['end_location']['lat'], steps[-1]['end_location']['lng']))
        return route
    return None

# Function to get train route
def get_train_route(start, end):
    directions = gmaps.directions(start, end, mode="transit", transit_mode="train")
    if directions:
        steps = directions[0]['legs'][0]['steps']
        route = [(step['start_location']['lat'], step['start_location']['lng']) for step in steps]
        route.append((steps[-1]['end_location']['lat'], steps[-1]['end_location']['lng']))
        return route
    return None

# Function to calculate total emissions
def calculate_total_emissions(path, G):
    total_emissions = 0
    for i in range(len(path) - 1):
        total_emissions += G[path[i]][path[i+1]]['emission']
    return int(total_emissions)

def get_maritime_distance(lat1, lng1, lat2, lng2, waypoints):
    start_closest, start_dist = find_closest_maritime_point(lat1, lng1, waypoints)
    end_closest, end_dist = find_closest_maritime_point(lat2, lng2, waypoints)
    start_idx, end_idx = waypoints.index(start_closest), waypoints.index(end_closest)
    if start_idx < end_idx:
        maritime_dist = sum(haversine(waypoints[i][0], waypoints[i][1], waypoints[i + 1][0], waypoints[i + 1][1]) for i in range(start_idx, end_idx))
    else:
        maritime_dist = sum(haversine(waypoints[i][0], waypoints[i][1], waypoints[i - 1][0], waypoints[i - 1][1]) for i in range(start_idx, end_idx, -1))
    return start_dist + maritime_dist + end_dist

# Fonction pour obtenir la distance fluviale
def get_river_distance(lat1, lng1, lat2, lng2, port1, port2, sheets_ports, sheets_river):
    for sheet_name, df_ports in sheets_ports.items():
        if port1 in df_ports['Ports'].tolist() and port2 in df_ports['Ports'].tolist():
            df_river_route = sheets_river[sheet_name]
            df_river_route[['Latitude', 'Longitude']] = df_river_route['Point'].str.extract(r'([0-9°\'"]+[NS])\s+([0-9°\'"]+[EW])')
            df_river_route['Latitude'] = df_river_route['Latitude'].apply(dms_to_decimal)
            df_river_route['Longitude'] = df_river_route['Longitude'].apply(dms_to_decimal)
            river_waypoints = df_river_route[["Latitude", "Longitude"]].apply(tuple, axis=1).tolist()
            start_closest, start_dist = find_closest_maritime_point(lat1, lng1, river_waypoints)
            end_closest, end_dist = find_closest_maritime_point(lat2, lng2, river_waypoints)
            start_idx, end_idx = river_waypoints.index(start_closest), river_waypoints.index(end_closest)
            if start_idx < end_idx:
                river_dist = sum(haversine(river_waypoints[i][0], river_waypoints[i][1], river_waypoints[i + 1][0], river_waypoints[i + 1][1]) for i in range(start_idx, end_idx))
            else:
                river_dist = sum(haversine(river_waypoints[i][0], river_waypoints[i][1], river_waypoints[i - 1][0], river_waypoints[i - 1][1]) for i in range(start_idx, end_idx, -1))
            return start_dist + river_dist + end_dist
    return float('inf')  # Return a large distance if ports are not in the same sheet

def create_graph(nodes, speeds, delays, emission_factors, price_factors, selected_category, qty, sheets_rail, sheets_ports, sheets_river, maritime_waypoints):
    G = nx.Graph()

    # Ajouter les noeuds au graphe
    for category, places_list in nodes.items():
        for place, lat, lng in places_list:
            G.add_node(place, category=category, lat=lat, lng=lng)

    # Calculer la distance entre le point de départ et le point d'arrivée
    start_point = nodes["others"][0]
    end_point = nodes["others"][1]
    start_lat, start_lng = start_point[1], start_point[2]
    end_lat, end_lng = end_point[1], end_point[2]
    
    direct_distance = haversine(start_lat, start_lng, end_lat, end_lng)

    # Calcul des pires facteurs pour la normalisation
    worst_speed = min(speeds.values())
    worst_price_factor = max(price_factors[selected_category].values())
    worst_emission_factor = max(emission_factors[selected_category].values())
    worst_duration = direct_distance / worst_speed + max(delays.values())
    worst_price = direct_distance * qty * worst_price_factor
    worst_emission = direct_distance * qty * worst_emission_factor

    # Distances maximum pour relier les modes à d'autres
    max_distance = 500

    def add_edge(node1, node2, precomputed_distance=None):
        category1, category2 = node1[1]['category'], node2[1]['category']
        lat1, lng1, lat2, lng2 = node1[1]['lat'], node1[1]['lng'], node2[1]['lat'], node2[1]['lng']

        # Utiliser la distance pré-calculée ou la calculer si nécessaire
        distance = precomputed_distance if precomputed_distance is not None else haversine(lat1, lng1, lat2, lng2)

        # Vérifier que la distance n'excède pas max_distance si les catégories sont différentes
        if category1 != category2 and distance > max_distance:
            return

        mode = None
        factor = 1.0

        # Gestion des différentes catégories
        if category1 == category2:
            if category1 == 'railway_stations' and are_stations_in_same_sheet(node1[0], node2[0], sheets_rail):
                mode, factor = 'rail', 1.2
            elif category1 == 'airports':
                mode, factor = 'air', 1.0
            elif category1 == 'ports':
                mode, distance = 'sea', get_maritime_distance(lat1, lng1, lat2, lng2, maritime_waypoints)
            elif category1 == 'river_ports' and are_stations_in_same_sheet(node1[0], node2[0], sheets_ports):
                mode, distance = 'river', get_river_distance(lat1, lng1, lat2, lng2, node1[0], node2[0], sheets_ports, sheets_river)
        else:
            mode, factor = 'road', 1.3

        if mode and distance is not None:
            if mode in ['rail', 'air', 'road']:
                distance = distance * factor

            # Calculer les métriques pour l'arête
            duration = distance / speeds[mode] + delays[mode]
            price = distance * qty * price_factors[selected_category][mode]
            emission = distance * qty * emission_factors[selected_category][mode]
            combined_factors = 0.33 * duration / worst_duration + 0.33 * price / worst_price + 0.33 * emission / worst_emission

            # Ajouter l'arête au graphe
            G.add_edge(node1[0], node2[0], mode=mode, distance_km=distance, duration=duration, emission=emission, price=price, combined_factors=combined_factors)

    # Pré-calculer les distances entre tous les nœuds
    node_pairs = list(combinations(G.nodes(data=True), 2))
    precomputed_distances = {}
    for node1, node2 in node_pairs:
        lat1, lng1 = node1[1]['lat'], node1[1]['lng']
        lat2, lng2 = node2[1]['lat'], node2[1]['lng']
        precomputed_distances[(node1[0], node2[0])] = haversine(lat1, lng1, lat2, lng2)

    # Boucle pour traiter toutes les paires de nœuds avec les distances pré-calculées
    for node1, node2 in node_pairs:
        add_edge(node1, node2, precomputed_distances[(node1[0], node2[0])])
    return G


def find_worst_path(G, start, end):
    # Inverser les poids des émissions en les multipliant par -1
    for u, v, data in G.edges(data=True):
        data['negative_emission'] = 1/(data['emission'] + 1e-12)
    
    # Trouver le chemin avec le plus grand poids négatif (c'est-à-dire les émissions maximales)
    worst_path = nx.shortest_path(G, source=start, target=end, weight='negative_emission')
    total_emissions = calculate_total_emissions(worst_path, G)
    
    # Restaurer les poids originaux si nécessaire
    for u, v, data in G.edges(data=True):
        del data['negative_emission']
    
    return worst_path, total_emissions

# Fonction pour calculer la durée totale
def calculate_total_duration(path, G):
    total_duration = 0
    for i in range(len(path) - 1):
        total_duration += G[path[i]][path[i+1]]['duration']
    return total_duration

# Fonction pour calculer la distance totale par mode
def calculate_total_duration_by_mode(path, G):
    total_duration_by_mode = {}
    
    for i in range(len(path) - 1):
        edge = G[path[i]][path[i+1]]
        mode = edge['mode']
        duration = edge['duration']
        
        if mode in total_duration_by_mode:
            total_duration_by_mode[mode] += round(duration, 1)
        else:
            total_duration_by_mode[mode] = round(duration, 1)
    
    return total_duration_by_mode

# Fonction pour calculer le prix total
def calculate_total_price(path, G):
    total_price = 0
    for i in range(len(path) - 1):
        total_price += G[path[i]][path[i+1]]['price']
    return total_price

# Fonction pour calculer la distance totale par mode
def calculate_total_distance_by_mode(path, G):
    total_distance_by_mode = {}
    
    for i in range(len(path) - 1):
        edge = G[path[i]][path[i+1]]
        mode = edge['mode']
        distance = edge['distance_km']
        
        if mode in total_distance_by_mode:
            total_distance_by_mode[mode] += int(distance)
        else:
            total_distance_by_mode[mode] = int(distance)
    
    return total_distance_by_mode

# Trouver le chemin le plus rapide basé sur la durée
def find_fastest_path(G, start, end):
    fastest_path = nx.shortest_path(G, source=start, target=end, weight='duration')
    total_distance = calculate_total_distance_by_mode(fastest_path, G)
    total_duration = calculate_total_duration_by_mode(fastest_path, G)
    total_emissions = calculate_total_emissions(fastest_path, G)
    total_price = calculate_total_price(fastest_path, G)
    return fastest_path, total_duration, total_emissions, total_price, total_distance

# Trouver le chemin le plus rapide basé sur la durée en utilisant uniquement la route
def find_fastest_road_path(start, end, speeds, emission_factors, price_factors, selected_category, qty):
    distance = get_driving_distance(start, end)
    total_distance = {}
    total_distance["road"] = int(distance)
    route = get_driving_route(start, end)
    total_duration = {}
    total_duration["road"] = round(distance/speeds['road'], 1)
    total_emissions = int(distance * qty * emission_factors[selected_category]['road'])
    total_price = distance * qty * price_factors[selected_category]['road']
    return route, total_duration, total_emissions, total_price, total_distance

# Trouver le chemin le plus rapide basé sur la durée
def find_cheapest_path(G, start, end):
    cheapest_path = nx.shortest_path(G, source=start, target=end, weight='price')
    total_distance = calculate_total_distance_by_mode(cheapest_path, G)
    total_duration = calculate_total_duration_by_mode(cheapest_path, G)
    total_emissions = calculate_total_emissions(cheapest_path, G)
    total_price = calculate_total_price(cheapest_path, G)
    return cheapest_path, total_duration, total_emissions, total_price, total_distance

# Trouver le chemin le plus écolo
def find_ecolo_path(G, start, end):
    ecolo_path = nx.shortest_path(G, source=start, target=end, weight='emission')
    total_distance = calculate_total_distance_by_mode(ecolo_path, G)
    total_duration = calculate_total_duration_by_mode(ecolo_path, G)
    total_emissions = calculate_total_emissions(ecolo_path, G)
    total_price = calculate_total_price(ecolo_path, G)
    return ecolo_path, total_duration, total_emissions, total_price, total_distance

# Trouver le chemin avec le meilleur rapport écologie-temps-prix
def find_optimal_path(G, start, end):
    optimal_path = nx.shortest_path(G, source=start, target=end, weight='combined_factors')
    total_distance = calculate_total_distance_by_mode(optimal_path, G)
    total_duration = calculate_total_duration_by_mode(optimal_path, G)
    total_emissions = calculate_total_emissions(optimal_path, G)
    total_price = calculate_total_price(optimal_path, G)
    return optimal_path, total_duration, total_emissions, total_price, total_distance

# Find the shortest path based on emissions without any duration
def find_optimal_path_v1(G, start, end):
    optimal_path = nx.shortest_path(G, source=start, target=end, weight='emission')
    total_emissions = calculate_total_emissions(optimal_path, G)
    return optimal_path, total_emissions

# Fonction pour obtenir la route maritime entre deux ports
def get_maritime_route(maritime_waypoints, start_lat, start_lng, end_lat, end_lng):
    start_closest_point, _ = find_closest_maritime_point(start_lat, start_lng, maritime_waypoints)
    end_closest_point, _ = find_closest_maritime_point(end_lat, end_lng, maritime_waypoints)
    
    start_index = maritime_waypoints.index(start_closest_point)
    end_index = maritime_waypoints.index(end_closest_point)
    
    if start_index < end_index:
        route = maritime_waypoints[start_index:end_index+1]
    else:
        route = maritime_waypoints[end_index:start_index+1][::-1]
    
    return [(start_lat, start_lng)] + route + [(end_lat, end_lng)]

# Fonction pour obtenir le nom de la feuille contenant les deux ports
def get_sheet_name(port1, port2, sheets_ports):
    for sheet_name, df_ports in sheets_ports.items():
        ports_list = df_ports['Ports'].tolist()
        if port1 in ports_list and port2 in ports_list:
            return sheet_name
    return None

# Fonction pour obtenir la route fluviale entre deux ports
def get_river_route(sheets_river, sheets_ports, start_port, end_port, start_node, end_node):
    # Trouver le nom de la feuille contenant les deux ports
    sheet_name = get_sheet_name(start_port, end_port, sheets_ports)
    if sheet_name is None:
        return []

    df_river_route = sheets_river[sheet_name]

    # Séparer les colonnes de latitude et de longitude
    df_river_route[['Latitude', 'Longitude']] = df_river_route['Point'].str.extract(r'([0-9°\'"]+[NS])\s+([0-9°\'"]+[EW])')

    # Convertir les coordonnées en décimal
    df_river_route['Latitude'] = df_river_route['Latitude'].apply(dms_to_decimal)
    df_river_route['Longitude'] = df_river_route['Longitude'].apply(dms_to_decimal)

    # Convertir en liste de tuples (latitude, longitude)
    river_waypoints = df_river_route[["Latitude", "Longitude"]].apply(tuple, axis=1).tolist()

    start_lat, start_lng = start_node['lat'], start_node['lng']
    end_lat, end_lng = end_node['lat'], end_node['lng']

    start_closest_point, _ = find_closest_maritime_point(start_lat, start_lng, river_waypoints)
    end_closest_point, _ = find_closest_maritime_point(end_lat, end_lng, river_waypoints)

    start_index = river_waypoints.index(start_closest_point)
    end_index = river_waypoints.index(end_closest_point)

    if start_index < end_index:
        route = river_waypoints[start_index:end_index+1]
    else:
        route = river_waypoints[end_index:start_index+1][::-1]

    if not route:
        raise ValueError("La route fluviale calculée est vide. Vérifiez les points de départ et d'arrivée.")

    return [(start_lat, start_lng)] + route + [(end_lat, end_lng)]

def add_path_to_map(path, start, end ,G, nodes, maritime_waypoints, sheets_river, sheets_ports, input_coordinates, path_is_road):
    # Créer une carte centrée sur l'Europe
    map_route = folium.Map(location=[54.5260, 15.2551], zoom_start=4)
    colors_path = {"road": "orange", "rail": "green", "sea": "blue", "air": "red", "river": "cyan"}

    folium.Marker([input_coordinates[0][0], input_coordinates[0][1]], popup=start, icon=folium.Icon()).add_to(map_route)
    folium.Marker([input_coordinates[1][0], input_coordinates[1][1]], popup=end, icon=folium.Icon()).add_to(map_route)

    for category in ['airports', 'ports', 'railway_stations', 'river_ports']:
        for place, lat, lng in nodes[category]:
            if place in path:
                folium.Marker([lat, lng], popup=place, icon=folium.Icon()).add_to(map_route)

    if path_is_road:
        folium.PolyLine(path, color='orange', weight=2.5, opacity=1, dash_array=None).add_to(map_route)

    else:
        for i in range(len(path) - 1):
            mode = G[path[i]][path[i + 1]]['mode']
            
            if mode == 'road':
                route = get_driving_route(path[i], path[i+1])
                folium.PolyLine(route, color=colors_path[mode]).add_to(map_route)   
            elif mode == 'sea':
                start_node = G.nodes[path[i]]
                end_node = G.nodes[path[i + 1]]
                start_lat, start_lng = start_node['lat'], start_node['lng']
                end_lat, end_lng = end_node['lat'], end_node['lng']
                maritime_route = get_maritime_route(maritime_waypoints, start_lat, start_lng, end_lat, end_lng)
                folium.PolyLine(maritime_route, color=colors_path[mode]).add_to(map_route) 
            elif mode == 'river':
                start_node = G.nodes[path[i]]
                end_node = G.nodes[path[i + 1]]
                # Obtenir les noms des ports fluviaux à partir de nodes
                start_port = next((place for place, lat, lng in nodes['river_ports'] if lat == start_node['lat'] and lng == start_node['lng']), None)
                end_port = next((place for place, lat, lng in nodes['river_ports'] if lat == end_node['lat'] and lng == end_node['lng']), None)

                maritime_route = get_river_route(sheets_river, sheets_ports, start_port, end_port, start_node, end_node)
                folium.PolyLine(maritime_route, color=colors_path[mode]).add_to(map_route)
            else:
                start_node = G.nodes[path[i]]
                end_node = G.nodes[path[i + 1]]
                folium.PolyLine(
                    [(start_node['lat'], start_node['lng']), (end_node['lat'], end_node['lng'])], color=colors_path[mode]).add_to(map_route)
                
    return map_route

