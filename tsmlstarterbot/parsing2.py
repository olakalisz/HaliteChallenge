import numpy as np
import pandas as pd

from tsmlstarterbot.common import *


def angle(x, y):
    radians = math.atan2(y, x)
    if radians < 0:
        radians = radians + 2 * math.pi
    return round(radians / math.pi * 180)


def find_winner(data):
    for player, stats in data['stats'].items():
        if stats['rank'] == 1:
            return player
    return -1


def angle_dist(a1, a2):
    return (a1 - a2 + 360) % 360


def format_data_for_training(data):
    """
    Create numpy array with ship features ready to feed to the neural net.
    :param data: parsed features
    :return: numpy array of shape (number of frames * number of ships, PER_SHIP_FEATURES)
             numpy array of shape (number of frames * number of ships, SHIP_ACTION_NAMES)
    """
    training_input = []
    training_output = []
    for d in data:
        features, expected_output = d

        features_vector = np.array(features)
        output_vector = np.array(expected_output)

        training_input.append(features_vector)
        training_output.append(output_vector)

    return np.array(training_input), np.array(training_output)


def serialize_data(data, dump_features_location):
    """
    Serialize all the features into .h5 file.

    :param data: data to serialize
    :param dump_features_location: path to .h5 file where the features should be saved
    """
    training_data_for_pandas = {
        (game_id, frame_id, planet_id): planet_features
        for game_id, frame in enumerate(data)
        for frame_id, planets in enumerate(frame)
        for planet_id, planet_features in planets[0].items()}

    training_data_to_store = pd.DataFrame.from_dict(training_data_for_pandas, orient="index")
    training_data_to_store.columns = FEATURE_NAMES
    index_names = ["game", "frame", "planet"]
    training_data_to_store.index = pd.MultiIndex.from_tuples(training_data_to_store.index, names=index_names)
    training_data_to_store.to_hdf(dump_features_location, "training_data")



def parse(all_games_json_data, bot_to_imitate=None, dump_features_location=None):
    print("Parsing data...")

    parsed_games = 0

    training_data_action = []
    training_data_move = []

    # Choosing the bot with the highest number of games won
    players_games_count = {}
    for json_data in all_games_json_data:
        w = find_winner(json_data)
        p = json_data['player_names'][int(w)]
        if p not in players_games_count:
            players_games_count[p] = 0
        players_games_count[p] += 1

    bot_to_imitate = max(players_games_count, key=players_games_count.get)

    for json_data in all_games_json_data:

        frames = json_data['frames']
        moves = json_data['moves']
        planets = json_data['planets']

        # For each game see if bot_to_imitate played in it
        if bot_to_imitate not in set(json_data['player_names']):
            continue
        # We train on all the games of the bot regardless whether it won or not.
        bot_to_imitate_id = str(json_data['player_names'].index(bot_to_imitate))

        parsed_games = parsed_games + 1
        game_training_data_action = []
        game_training_data_move = []

        # Ignore the last frame, no decision to be made there
        for idx in range(len(frames) - 1):

            current_moves = moves[idx]
            current_frame = frames[idx]

            if bot_to_imitate_id not in current_frame['ships'] or len(current_frame['ships'][bot_to_imitate_id]) == 0:
                continue
            current_ships = current_frame['ships']
            
            # Compute features
            for player_id, ships in current_ships.items():
                for ship_id, ship_data in ships.items():
                    is_bot_to_imitate = 1 if player_id == bot_to_imitate_id else 0

                    # Get ship features only if the player is the bot of interest
                    if not is_bot_to_imitate:
                        continue

                    # print(ship_data)
                    is_docked = 1 if ship_data['docking']['status'] == 'docked' else 0
                    is_docking = 1 if ship_data['docking']['status'] == 'docking' else 0
                    health = ship_data['health']
                    position_x = ship_data['x']
                    position_y = ship_data['y']

                    # Features with info about other ships
                    closest_friendly_ship_distance_x = 10000
                    closest_enemy_ship_distance_x = 10000
                    closest_friendly_ship_distance_y = 10000
                    closest_enemy_ship_distance_y = 10000

                    closest_friendly_docked_ship_distance_x = 10000
                    closest_enemy_docked_ship_distance_x = 10000
                    closest_friendly_docked_ship_distance_y = 10000
                    closest_enemy_docked_ship_distance_y = 10000

                    number_of_enemies_within_radius = 0
                    number_of_friends_within_radius = 0
                    ours_total_ship_count = 1
                    enemy_total_ship_count = 0

                    # Helper variables
                    friendly_distance = 10000
                    enemy_distance = 10000
                    friendly_distance_docked = 10000
                    enemy_distance_docked = 10000

                    for other_player_id, other_ships in current_ships.items():
                        for other_ship_id, other_ship_data in other_ships.items():

                            # Only interested in ships other than the one we are collecting the data for
                            if other_ship_id == ship_id:
                                continue

                            friendly = 1 if other_player_id == bot_to_imitate_id else 0

                            if friendly:
                                if friendly_distance > distance(other_ship_data['x'], other_ship_data['y'], position_x, position_y):
                                    friendly_distance = distance(other_ship_data['x'], other_ship_data['y'], position_x, position_y)
                                    closest_friendly_ship_distance_x = abs(position_x - other_ship_data['x'])
                                    closest_friendly_ship_distance_y = abs(position_y - other_ship_data['y'])

                                if other_ship_data['docking']['status'] == 'docked':
                                    if friendly_distance_docked > distance(other_ship_data['x'], other_ship_data['y'], position_x, position_y):
                                        friendly_distance_docked = distance(other_ship_data['x'], other_ship_data['y'], position_x, position_y)
                                        closest_friendly_docked_ship_distance_x = abs(position_x - other_ship_data['x'])
                                        closest_friendly_docked_ship_distance_y = abs(position_y - other_ship_data['y'])

                                if distance(other_ship_data['x'], other_ship_data['y'], position_x, position_y) < RADIUS:
                                    number_of_friends_within_radius = number_of_friends_within_radius + 1

                                ours_total_ship_count = ours_total_ship_count + 1

                            # else the ship belongs to enemy
                            else:
                                if enemy_distance > distance(other_ship_data['x'], other_ship_data['y'], position_x, position_y):
                                    enemy_distance = distance(other_ship_data['x'], other_ship_data['y'], position_x, position_y)
                                    closest_enemy_ship_distance_x = abs(position_x - other_ship_data['x'])
                                    closest_enemy_ship_distance_y = abs(position_y - other_ship_data['y'])

                                if other_ship_data['docking']['status'] == 'docked':
                                    if enemy_distance_docked > distance(other_ship_data['x'], other_ship_data['y'], position_x, position_y):
                                        enemy_distance_docked = distance(other_ship_data['x'], other_ship_data['y'], position_x, position_y)
                                        closest_enemy_docked_ship_distance_x = abs(position_x - other_ship_data['x'])
                                        closest_enemy_docked_ship_distance_y = abs(position_y - other_ship_data['y'])

                                if distance(other_ship_data['x'], other_ship_data['y'], position_x, position_y) < RADIUS:
                                    number_of_enemies_within_radius = number_of_enemies_within_radius + 1

                                enemy_total_ship_count = enemy_total_ship_count + 1

                    # Features with info about planets
                    closest_friendly_planet_distance_x = 10000
                    closest_enemy_planet_distance_x = 10000
                    closest_neutral_planet_distance_x = 10000
                    closest_friendly_planet_distance_y = 10000
                    closest_enemy_planet_distance_y = 10000
                    closest_neutral_planet_distance_y = 10000

                    closest_friendly_planet_docking_spots = 0
                    closest_enemy_planet_docking_spots = 0
                    closest_neutral_planet_docking_spots = 0

                    closest_friendly_planet_health = 0
                    closest_enemy_planet_health = 0
                    closest_neutral_planet_health = 0

                    closest_friendly_planet_production = 0
                    closest_enemy_planet_production = 0
                    closest_neutral_planet_production = 0

                    # Helper variables
                    closest_friendly_planet = 10000
                    closest_enemy_planet = 10000
                    closest_neutral_planet = 10000

                    for planet_id in range(len(planets)):
                        if str(planet_id) not in current_frame['planets']:
                            continue

                        planet_data = planets[planet_id]
                        owner = current_frame['planets'][str(planet_id)]['owner']
                        if owner is None:
                            if closest_neutral_planet > distance(planet_data['x'], planet_data['y'], position_x, position_y):
                                closest_neutral_planet = distance(planet_data['x'], planet_data['y'], position_x, position_y)
                                closest_neutral_planet_distance_x = abs(position_x - (planet_data['x'] + planet_data['r']))
                                closest_neutral_planet_distance_y = abs(position_y - (planet_data['y'] + planet_data['r']))
                                closest_neutral_planet_docking_spots = planet_data['docking_spots']
                                closest_neutral_planet_health = planet_data['health']
                                closest_neutral_planet_production = planet_data['production']

                        if str(owner) == bot_to_imitate_id:
                            if closest_friendly_planet > distance(planet_data['x'], planet_data['y'], position_x, position_y):
                                closest_friendly_planet = distance(planet_data['x'], planet_data['y'], position_x, position_y)
                                closest_friendly_planet_distance_x = abs(position_x - (planet_data['x'] + planet_data['r']))
                                closest_friendly_planet_distance_y = abs(position_y - (planet_data['y'] + planet_data['r']))
                                closest_friendly_planet_docking_spots = planet_data['docking_spots']
                                closest_friendly_planet_health = planet_data['health']
                                closest_friendly_planet_production = planet_data['production']

                        if str(owner) != bot_to_imitate_id and owner is not None:
                            if closest_enemy_planet > distance(planet_data['x'], planet_data['y'], position_x, position_y):
                                closest_enemy_planet = distance(planet_data['x'], planet_data['y'], position_x, position_y)
                                closest_enemy_planet_distance_x = abs(position_x - (planet_data['x'] + planet_data['r']))
                                closest_enemy_planet_distance_y = abs(position_y - (planet_data['y'] + planet_data['r']))
                                closest_enemy_planet_docking_spots = planet_data['docking_spots']
                                closest_enemy_planet_health = planet_data['health']
                                closest_enemy_planet_production = planet_data['production']

                    state = [
                        is_docked,
                        is_docking,
                        health,
                        position_x,
                        position_y,
                        closest_friendly_ship_distance_x,
                        closest_enemy_ship_distance_x,
                        closest_friendly_ship_distance_y,
                        closest_enemy_ship_distance_y,
                        closest_friendly_docked_ship_distance_x,
                        closest_enemy_docked_ship_distance_x,
                        closest_friendly_docked_ship_distance_y,
                        closest_enemy_docked_ship_distance_y,
                        number_of_enemies_within_radius,
                        number_of_friends_within_radius,
                        ours_total_ship_count,
                        enemy_total_ship_count,
                        closest_friendly_planet_distance_x,
                        closest_enemy_planet_distance_x,
                        closest_neutral_planet_distance_x,
                        closest_friendly_planet_distance_y,
                        closest_enemy_planet_distance_y,
                        closest_neutral_planet_distance_y,
                        closest_friendly_planet_docking_spots,
                        closest_enemy_planet_docking_spots,
                        closest_neutral_planet_docking_spots,
                        closest_friendly_planet_health,
                        closest_enemy_planet_health,
                        closest_neutral_planet_health,
                        closest_friendly_planet_production,
                        closest_enemy_planet_production,
                        closest_neutral_planet_production]

                    # Find the action
                    noaction = 0
                    if ship_id not in current_moves[bot_to_imitate_id][0]:
                        noaction = 1
                        move = 0
                        dock = 0
                        undock = 0
                        angle = 0
                        velocity = 0
                    else:
                        move_data = current_moves[bot_to_imitate_id][0][ship_id]
                        move = 1 if move_data['type'] == 'thrust' else 0
                        dock = 1 if move_data['type'] == 'dock' else 0
                        undock = 1 if move_data['type'] == 'undock' else 0
                        angle = 0
                        velocity = 0
                    if move:
                        angle = move_data['angle']
                        velocity = move_data['magnitude']

                    action = [
                        move,
                        dock,
                        noaction,
                        undock,
                        ]

                    move_action = [
                        angle,
                        velocity
                    ]
            game_training_data_action.append((state, action))        
            if move == 1:
                game_training_data_move.append((state, move_action))
        training_data_action.append(game_training_data_action)
        training_data_move.append(game_training_data_move)

    flat_training_data_action = [item for sublist in training_data_action for item in sublist]
    flat_training_data_move = [item for sublist in training_data_move for item in sublist]

    if dump_features_location is not None:
        serialize_data(training_data, dump_features_location)

    print("Action data parsed, parsed {} games, total frames: {}".format(parsed_games, len(flat_training_data_action)))
    print("Move data parsed, parsed {} games, total frames: {}".format(parsed_games, len(flat_training_data_move)))
    # print(flat_training_data)

    return format_data_for_training(flat_training_data_action), format_data_for_training(flat_training_data_move)
