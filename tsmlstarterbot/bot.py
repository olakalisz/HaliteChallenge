import heapq
import numpy as np
import os
import time
import logging

import hlt
from tsmlstarterbot.common import *
from tsmlstarterbot.neural_net import NeuralNet
from tsmlstarterbot.neural_net_move import NeuralNetMove


class Bot:
    def __init__(self, location, name):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        action_model_location = os.path.join(current_directory, os.path.pardir, "models", location)
        move_model_location = os.path.join(current_directory, os.path.pardir, "models", "move_model.ckpt")
        self._name = name
        self._neural_net = NeuralNet(cached_model=action_model_location)
        self._neural_net_move = NeuralNetMove(cached_model=move_model_location)

        # Run prediction on random data to make sure that code path is executed at least once before the game starts
        random_input_data = np.random.rand(PER_SHIP_FEATURES)
        predictions = self._neural_net.predict(random_input_data)
        assert len(predictions) == PER_SHIP_ACTIONS

        predictions2 = self._neural_net_move.predict(random_input_data)
        assert len(predictions2) == PER_MOVE_FEATURES

    def play(self):
        """
        Play a game using stdin/stdout.
        """

        # Initialize the game.
        game = hlt.Game(self._name)

        while True:
            # Update the game map.
            game_map = game.update_map()
            start_time = time.time()

            # Produce features for each ship owned by player.
            features = self.produce_ship_feature_dict(game_map)

            # Find predictions which move should be applied to each ship
            predictions = {}
            for ship in features:
                predictions[ship] = self._neural_net.predict(features[ship])
            #print(predictions)

            # # Use simple greedy algorithm to assign closest ships to each planet according to predictions.
            # ships_to_planets_assignment = self.produce_ships_to_planets_assignment(game_map, predictions)

            # Produce halite instruction for each ship.
            instructions = self.produce_instructions(game_map, predictions, features)

            # Send the command.
            game.send_command_queue(instructions)

    def produce_ship_feature_dict(self, game_map):
        """
        For each ship owned by us create a set of features that we will feed to the neural net. We always return a dict
        of ships, we use the same predefined constants as in processing data from replays.

        :param game_map: game map
        :return: dict, with ships as keys and features per ship as values
        """

        # Initialize the dictionary of ships
        feature_dict = {}

        # Find the ID of our player
        my_id = game_map.get_me()

        # Get data about ships owned by our bot in the given game state
        for ship in my_id.all_ships():
            # Basic features per ship
            is_docked = 1 if ship.docking_status == ship.DockingStatus.DOCKED else 0
            is_docking = 1 if ship.docking_status == ship.DockingStatus.DOCKING else 0
            health = ship.health
            position_x = ship.x
            position_y = ship.y

            # Predefine the constant values in the same manner as for replays processing:
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

            for player in game_map.all_players():
                for other_ship in player.all_ships():

                    # Only interested in ships other than the one we are collecting the data for
                    if other_ship.id == ship.id:
                        continue

                    friendly = 1 if player == my_id else 0

                    if friendly:
                        if friendly_distance > distance(other_ship.x, other_ship.y, position_x, position_y):
                            friendly_distance = distance(other_ship.x, other_ship.y, position_x, position_y)
                            closest_friendly_ship_distance_x = abs(position_x - other_ship.x)
                            closest_friendly_ship_distance_y = abs(position_y - other_ship.y)

                        if other_ship.docking_status == other_ship.DockingStatus.DOCKED:
                            if friendly_distance_docked > distance(other_ship.x, other_ship.y, position_x, position_y):
                                friendly_distance_docked = distance(other_ship.x, other_ship.y, position_x, position_y)
                                closest_friendly_docked_ship_distance_x = abs(position_x - other_ship.x)
                                closest_friendly_docked_ship_distance_y = abs(position_y - other_ship.y)

                        if distance(other_ship.x, other_ship.y, position_x, position_y) < RADIUS:
                            number_of_friends_within_radius = number_of_friends_within_radius + 1

                        ours_total_ship_count = ours_total_ship_count + 1

                    # else the ship belongs to enemy
                    else:
                        if enemy_distance > distance(other_ship.x, other_ship.y, position_x, position_y):
                            enemy_distance = distance(other_ship.x, other_ship.y, position_x, position_y)
                            closest_enemy_ship_distance_x = abs(position_x - other_ship.x)
                            closest_enemy_ship_distance_y = abs(position_y - other_ship.y)

                        if other_ship.docking_status == other_ship.DockingStatus.DOCKED:
                            if enemy_distance_docked > distance(other_ship.x, other_ship.y, position_x, position_y):
                                enemy_distance_docked = distance(other_ship.x, other_ship.y, position_x, position_y)
                                closest_enemy_docked_ship_distance_x = abs(position_x - other_ship.x)
                                closest_enemy_docked_ship_distance_y = abs(position_y - other_ship.y)

                        if distance(other_ship.x, other_ship.y, position_x, position_y) < RADIUS:
                            number_of_enemies_within_radius = number_of_enemies_within_radius + 1

                        enemy_total_ship_count = enemy_total_ship_count + 1

            for planet in game_map.all_planets():
                # Compute "ownership" feature - 0 if planet is not occupied, 1 if occupied by us,
                # -1 if occupied by enemy.
                if planet.owner == my_id:
                    ownership = 1
                elif planet.owner is None:
                    ownership = 0
                else:  # owned by enemy
                    ownership = -1

                if ownership == 0:
                    if closest_neutral_planet > distance(planet.x, planet.y, position_x, position_y):
                        closest_neutral_planet = distance(planet.x, planet.y, position_x, position_y)
                        closest_neutral_planet_distance_x = abs(position_x - (planet.x + planet.radius))
                        closest_neutral_planet_distance_y = abs(position_y - (planet.y + planet.radius))
                        closest_neutral_planet_docking_spots = planet.num_docking_spots
                        closest_neutral_planet_health = planet.health
                        closest_neutral_planet_production = planet.current_production

                if ownership == 1:
                    if closest_friendly_planet > distance(planet.x, planet.y, position_x, position_y):
                        closest_friendly_planet = distance(planet.x, planet.y, position_x, position_y)
                        closest_friendly_planet_distance_x = abs(position_x - (planet.x + planet.radius))
                        closest_friendly_planet_distance_y = abs(position_y - (planet.y + planet.radius))
                        closest_friendly_planet_docking_spots = planet.num_docking_spots
                        closest_friendly_planet_health = planet.health
                        closest_friendly_planet_production = planet.current_production

                if ownership == -1:
                    if closest_enemy_planet > distance(planet.x, planet.y, position_x, position_y):
                        closest_enemy_planet = distance(planet.x, planet.y, position_x, position_y)
                        closest_enemy_planet_distance_x = abs(position_x - (planet.x + planet.radius))
                        closest_enemy_planet_distance_y = abs(position_y - (planet.y + planet.radius))
                        closest_enemy_planet_docking_spots = planet.num_docking_spots
                        closest_enemy_planet_health = planet.health
                        closest_enemy_planet_production = planet.current_production

            # Add ship features to dictionary
            feature_dict[ship] = [
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

        return feature_dict

    def produce_instructions(self, game_map, predictions, features):
        """
        Given dictionary of predictions per ship produce instructions for every ship.
        We apply the move from the predictions to every ship

        :param game_map: game map
        :param predictions: dictionary of predictions, ships as keys, predictions per ship as values
        :return: list of instructions to send to the Halite engine
        """
        command_queue = []

        # Create moves for each ship
        for ship in predictions:
            prediction = predictions[ship]
            logging.info(prediction)
            # Define action based on prediction
            move = prediction[0]
            dock = prediction[1]
            noaction = prediction[2]
            undock = prediction[3]
            # angle = prediction[1]
            # velocity = prediction[2]

            move_prediction = self._neural_net_move.predict(features[ship])
            logging.info(move_prediction)

            # Apply move action
            if move > dock and move > undock:
                move_prediction = self._neural_net_move.predict(features[ship])
                angle = move_prediction[0]
                velocity = move_prediction[1]
                command_queue.append(ship.thrust(velocity, angle))

            # Apply dock action
            if dock > move and dock > undock:
                position_x = ship.x
                position_y = ship.y
                # Find the closest planet
                distance = 10000
                planet_to_dock = None
                for planet in game_map.all_planets():
                    if distance > distance(planet.x, planet.y, position_x, position_y):
                        distance = distance(planet.x, planet.y, position_x, position_y)
                        planet_to_dock = planet

                command_queue.append(ship.dock(planet_to_dock))

            # Apply undock action
            if undock > move and undock > dock:
                command_queue.append(ship.undock())

            
            # position_x = ship.x
            # position_y = ship.y

            # # Find the closest planet
            # distance = 10000
            # planet_to_dock = None
            # for planet in game_map.all_planets():
            #     if distance > distance(planet.x, planet.y, position_x, position_y):
            #         distance = distance(planet.x, planet.y, position_x, position_y)
            #         planet_to_dock = planet
            # # Dock to the closest planet if possible
            # if ship.can_dock(planet_to_dock):
            #     command_queue.append(ship.dock(planet_to_dock))
            # else:
            #     # Move according to values learned from the network, if possible
            #     if ship.docking_status == UNDOCKED:
            #         command_queue.append(ship.thrust(velocity, angle))


        return command_queue