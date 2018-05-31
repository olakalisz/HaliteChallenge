import argparse
import json
import os.path
import zipfile

import numpy as np
import pandas as pd
from tsmlstarterbot.parsing2 import parse

from tsmlstarterbot.neural_net import NeuralNet
from tsmlstarterbot.neural_net_move import NeuralNetMove


def fetch_data_dir(directory, limit):
    """
    Loads up to limit games into Python dictionaries from uncompressed replay files.
    """
    replay_files = sorted([f for f in os.listdir(directory) if
                           os.path.isfile(os.path.join(directory, f)) and f.startswith("replay-")])

    if len(replay_files) == 0:
        raise Exception("Didn't find any game replays. Please call make games.")

    print("Found {} games.".format(len(replay_files)))
    print("Trying to load up to {} games ...".format(limit))

    loaded_games = 0

    all_data = []
    for r in replay_files:
        full_path = os.path.join(directory, r)
        with open(full_path) as game:
            game_data = game.read()
            game_json_data = json.loads(game_data)
            all_data.append(game_json_data)
        loaded_games = loaded_games + 1

        if loaded_games >= limit:
            break

    print("{} games loaded.".format(loaded_games))

    return all_data

def fetch_data_zip(zipfilename, limit):
    """
    Loads up to limit games into Python dictionaries from a zipfile containing uncompressed replay files.
    """
    all_jsons = []
    with zipfile.ZipFile(zipfilename) as z:
        print("Found {} games.".format(len(z.filelist)))
        print("Trying to load up to {} games ...".format(limit))
        for i in z.filelist[:limit]:
            with z.open(i) as f:
                lines = f.readlines()
                assert len(lines) == 1
                d = json.loads(lines[0].decode())
                all_jsons.append(d)
    print("{} games loaded.".format(len(all_jsons)))
    return all_jsons

def main():
    parser = argparse.ArgumentParser(description="Halite II training")
    parser.add_argument("--model_name", help="Name of the model")
    parser.add_argument("--minibatch_size", type=int, help="Size of the minibatch", default=100)
    parser.add_argument("--steps", type=int, help="Number of steps in the training", default=100)
    parser.add_argument("--data", help="Data directory or zip file containing uncompressed games")
    parser.add_argument("--cache", help="Location of the model we should continue to train")
    parser.add_argument("--games_limit", type=int, help="Train on up to games_limit games", default=1000)
    parser.add_argument("--seed", type=int, help="Random seed to make the training deterministic")
    parser.add_argument("--bot_to_imitate", help="Name of the bot whose strategy we want to learn")
    parser.add_argument("--dump_features_location", help="Location of hdf file where the features should be stored")

    args = parser.parse_args()

    # Make deterministic if needed
    if args.seed is not None:
        np.random.seed(args.seed)
    nn_action = NeuralNet(cached_model=args.cache, seed=args.seed)
    nn_move = NeuralNetMove(cached_model=args.cache, seed=args.seed)

    if args.data.endswith('.zip'):
        raw_data = fetch_data_zip(args.data, args.games_limit)
    else:
        raw_data = fetch_data_dir(args.data, args.games_limit)

    data_action, data_move = parse(raw_data, args.bot_to_imitate, args.dump_features_location)
    data_input_action, data_output_action = data_action
    data_input_move, data_output_move = data_move
    data_size_action = len(data_input_action)
    data_size_move = len(data_input_move)

    training_input_action, training_output_action = data_input_action[:int(0.85 * data_size_action)], data_output_action[:int(0.85 * data_size_action)]
    print("Training input action training_input_action[0]: {}".format(training_input_action[0]))
    validation_input_action, validation_output_action = data_input_action[int(0.85 * data_size_action):], data_output_action[int(0.85 * data_size_action):]

    training_data_size_action = len(training_input_action)

    # randomly permute the data
    permutation = np.random.permutation(training_data_size_action)
    training_input_action, training_output_action = training_input_action[permutation], training_output_action[permutation]


    training_input_move, training_output_move = data_input_move[:int(0.85 * data_size_move)], data_output_move[:int(0.85 * data_size_move)]
    print("Training input action training_input_move[0]: {}".format(training_input_move[0]))
    print("Training output action training_input_move[0]: {}".format(training_output_move[0]))
    validation_input_move, validation_output_move = data_input_move[int(0.85 * data_size_move):], data_output_move[int(0.85 * data_size_move):]

    training_data_size_move = len(training_input_move)

    # randomly permute the data
    permutation = np.random.permutation(training_data_size_move)
    training_input_move, training_output_move = training_input_move[permutation], training_output_move[permutation]

    print("Initial, cross validation loss action: {}".format(nn_action.compute_loss(validation_input_action, validation_output_action)))
    print("Initial, cross validation loss move: {}".format(nn_move.compute_loss(validation_input_move, validation_output_move)))

    curves_action = []
    curves_move = []

    for s in range(args.steps):
        start1 = (s * args.minibatch_size) % training_data_size_action
        end1 = start1 + args.minibatch_size
        training_loss_action = nn_action.fit(training_input_action[start1:end1], training_output_action[start1:end1])
        if s % 25 == 0 or s == args.steps - 1:
            validation_loss_action = nn_action.compute_loss(validation_input_action, validation_output_action)
            print("Step: {}, action cross validation loss: {}, training_loss: {}".format(s, validation_loss_action, training_loss_action))
            curves_action.append((s, training_loss_action, validation_loss_action))

        start2 = (s * args.minibatch_size) % training_data_size_move
        end2 = start2 + args.minibatch_size
        training_loss_move = nn_move.fit(training_input_move[start2:end2], training_output_move[start2:end2])
        if s % 25 == 0 or s == args.steps - 1:
            validation_loss_move = nn_move.compute_loss(validation_input_move, validation_output_move)
            print("Step: {}, move cross validation loss: {}, training_loss: {}".format(s, validation_loss_move, training_loss_move))
            curves_move.append((s, training_loss_move, validation_loss_move))

    cf1 = pd.DataFrame(curves_action, columns=['step', 'training_loss', 'cv_loss'])
    fig1 = cf1.plot(x='step', y=['training_loss', 'cv_loss']).get_figure()

    cf2 = pd.DataFrame(curves_move, columns=['step', 'training_loss', 'cv_loss'])
    fig2 = cf2.plot(x='step', y=['training_loss', 'cv_loss']).get_figure()

    # Save the trained model, so it can be used by the bot
    current_directory = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_directory, os.path.pardir, "models", args.model_name + ".ckpt")
    print("Training finished, serializing action model to {}".format(model_path))
    nn_action.save(model_path)
    print("Model serialized - action")

    # Save the trained model, so it can be used by the bot
    model_path_move = os.path.join(current_directory, os.path.pardir, "models", "move_model" + ".ckpt")
    print("Training finished, serializing move model to {}".format(model_path))
    nn_move.save(model_path_move)
    print("Model serialized - move")

    curve_path = os.path.join(current_directory, os.path.pardir, "models", args.model_name + "_training_plot.png")
    curve_path_move = os.path.join(current_directory, os.path.pardir, "models", "move_model" + "_training_plot.png")
    fig1.savefig(curve_path)
    fig2.savefig(curve_path_move)

if __name__ == "__main__":
    main()
