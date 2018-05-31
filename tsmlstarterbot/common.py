import math

# Max number of planets.
PLANET_MAX_NUM = 28

# Radius we use to look for friends/enemies
RADIUS = 20

# # These are the features we compute per planet
# FEATURE_NAMES = [
#     "health",
#     "available_docking_spots",
#     "remaining_production",
#     "signed_current_production",
#     "gravity",
#     "closest_friendly_ship_distance",
#     "closest_enemy_ship_distance",
#     "ownership",
#     "distance_from_center",
#     "weighted_average_distance_from_friendly_ships",
#     "is_active"]

# These are the features we compute per ship
SHIP_FEATURE_NAMES = [
    "is_docked",
    "is_docking",
    "health",
    "position_x",
    "position_y",
    "closest_friendly_ship_distance_x",
    "closest_enemy_ship_distance_x",
    "closest_friendly_ship_distance_y",
    "closest_enemy_ship_distance_y",
    "closest_friendly_docked_ship_distance_x",
    "closest_enemy_docked_ship_distance_x",
    "closest_friendly_docked_ship_distance_y",
    "closest_enemy_docked_ship_distance_y",
    "number_of_enemies_within_radius",
    "number_of_friends_within_radius",
    "ours_total_ship_count",
    "enemy_total_ship_count",
    "closest_friendly_planet_distance_x",
    "closest_enemy_planet_distance_x",
    "closest_neutral_planet_distance_x",
    "closest_friendly_planet_distance_y",
    "closest_enemy_planet_distance_y",
    "closest_neutral_planet_distance_y",
    "closest_friendly_planet_docking_spots",
    "closest_enemy_planet_docking_spots",
    "closest_neutral_planet_docking_spots",
    "closest_friendly_planet_health",
    "closest_enemy_planet_health",
    "closest_neutral_planet_health",
    "closest_friendly_planet_production",
    "closest_enemy_planet_production",
    "closest_neutral_planet_production"]

SHIP_ACTION_NAMES = [
    "move",
    "dock",
    "noaction",
    "undock"]

SHIP_MOVE = [
    "angle",
    "velocity"]

# # Number of initial features per planet we have
# PER_PLANET_FEATURES = len(FEATURE_NAMES)
# Number of initial features per ship we have
PER_SHIP_FEATURES = len(SHIP_FEATURE_NAMES)
# Number of initial actions per ship we have
PER_SHIP_ACTIONS = len(SHIP_ACTION_NAMES)
# Number of initial features per move:
PER_MOVE_FEATURES = len(SHIP_MOVE)

def distance2(x1, y1, x2, y2):
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


def distance(x1, y1, x2, y2):
    return math.sqrt(distance2(x1, y1, x2, y2))
