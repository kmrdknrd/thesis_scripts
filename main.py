## Created by Konrad Mikalauskas on 2023-05-16, 17:00

import numpy as np
import pandas as pd
import scipy.stats as stats
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import math
import shutil
import tempfile
import pickle
import json
import re

from numpy import dot
from numpy.linalg import norm
from pathlib import Path
from itertools import product
from sklearn.metrics import pairwise_distances


def gpt_output_cleaner(gpt_output_path):
    """Reads in GPT output and returns a dictionary of item alternative numbers and item texts.

    Args:
        gpt_output_path (str): Path to GPT output file.

    Returns:
        dict: Dictionary of item alternative numbers (keys) and item texts (values).
    """

    # Read in the GPT output
    with open(gpt_output_path, 'r') as f:
        gpt_output = f.readlines()
    
    # Remove non-item lines
    gpt_output = [line.strip() for line in gpt_output if line[0].isdigit()]

    # Keep only item numbers (1a, 1b, 1c, etc.)
    item_alt_nums = [re.findall(r'\d+[a-z]', item)[0] for item in gpt_output]

    # Keep only item texts
    gpt_output = [re.findall(r'"(.*?)"', item)[0] for item in gpt_output]

    # Convert list to dictionary
    gpt_output = {item_alt_nums[i]: gpt_output[i] for i in range(0, len(gpt_output))}

    return gpt_output


def item_pairer(gpt_output, n_alts = 5):
    """Takes in a dictionary of item alternative numbers and item texts and returns a list of all possible item pairs.

    Args:
        gpt_output (dict): Dictionary of item alternative numbers (keys) and item texts (values).
        n_alts (int, optional): Number of alternatives per item. Defaults to 5.

    Returns:
        list: List of all possible item pairs.
    """

    # Initialize empty list
    pair_list = []

    # Get item alternative numbers
    item_alt_nums = list(gpt_output.keys())

    # Split item_alt_nums into 'n_items' lists of 'n_alts' items each
    item_alt_nums = [item_alt_nums[i:i + n_alts] for i in range(0, len(item_alt_nums), n_alts)]

    # Get all unique item_alt pairs
    for item_higher in range(len(item_alt_nums)):
        for item_lower in range(item_higher + 1, len(item_alt_nums)):
            for alt_higher in item_alt_nums[item_higher]:
                for alt_lower in item_alt_nums[item_lower]:
                    pair_list.append(alt_higher + ' + ' + alt_lower)
    
    return pair_list


def pair_similarity_computer(gpt_output, pair_list):
    """
    Takes in a dictionary of item alternative numbers and item texts, and a list of all possible item pairs
    and returns a dictionary of item pairs and their cosine similarities.
    """

    # Load GUSE model
    try:
        model(input)
    except NameError as ne:
        print(ne)
        model_url = 'https://tfhub.dev/google/universal-sentence-encoder/4'
        try:
            print("Downloading model...")
            model = hub.load(model_url)
        except OSError as ose:
            print(ose)
            print("Directory already exists. Deleting and re-downloading model...")
            temp_dir = Path(tempfile.gettempdir()) / "tfhub_modules"
            shutil.rmtree(temp_dir)
            model = hub.load(model_url)
        finally:
            print("Model downloaded.")
    except ValueError as ve:
        pass

    # Define function to embed text
    def embed_text(input):
        return model(input)
    
    # Initialize empty list
    pair_similarities = {}

    # Get USE embeddings and compute cosine similarities
    for pair in pair_list:
        item_1_num, item_2_num = pair.split(' + ')
        pair_embeddings = embed_text([gpt_output[item_1_num],
                                      gpt_output[item_2_num]])

        # Compute item-item cosine similarity
        pair_similarities[pair] = dot(pair_embeddings[0], pair_embeddings[1]) / (norm(pair_embeddings[0]) * norm(pair_embeddings[0]))
    
    return pair_similarities


def combo_finder(gpt_output, pair_similarities, n_alts = 5):
    """Takes in a dictionary of item alternative numbers and item texts, a dictionary of item pairs and their cosine similarities,
    and returns a dictionary of all possible item combinations and their median similarities.
    
    Args:
        gpt_output (dict): Dictionary of item alternative numbers (keys) and item texts (values).
        pair_similarities (dict): Dictionary of item pairs (keys) and their cosine similarities (values).
        n_alts (int, optional): Number of alternatives per item. Defaults to 5.
    
    Returns:
        dict: Dictionary of all possible item combinations (keys) and their median similarities (values)."""

    # Get item alternative numbers
    item_alt_nums = list(gpt_output.keys())

    # Split item_alt_nums into 'n_items' lists of 'n_alts' items each
    item_alt_nums = [item_alt_nums[i:i + n_alts] for i in range(0, len(item_alt_nums), n_alts)]

    # Get all possible item combinations
    all_item_combos = list(product(*item_alt_nums))

    # Get number of pairs in each combination and initialize empty array to store median similarities
    n_pairs = math.comb(len(all_item_combos[0]), 2)
    median_similarities = np.empty(len(all_item_combos))

    # Get median similarities for all pairs in each combination
    for idx, combo in enumerate(all_item_combos):
        item_item_similarities = np.empty((0, n_pairs))
        for item_higher in range(len(combo)):
            for item_lower in range(item_higher + 1, len(combo)):
                item_item_similarities = np.append(item_item_similarities,
                                                   pair_similarities[combo[item_higher] + ' + ' + combo[item_lower]])
        print(combo)
        median_similarities[idx] = np.median(item_item_similarities)
    
    # Create dictionary of item combinations and their median similarities
    combo_sem_sims = {combo: median_similarities[idx] for idx, combo in enumerate(all_item_combos)}

    return combo_sem_sims


def semantic_separator(gpt_output_path, n_alts = 5, write = True, output_path = None):
    """Takes in a path to a GPT output file and returns a dictionary of item alternative combinations
    and their median semantic similarities.
    
    Args:
        gpt_output_path (str): Path to a GPT output file.
        n_alts (int, optional): Number of alternatives per item. Defaults to 5.
    
    Returns:
        dict: Dictionary of item alternative combinations and their median semantic similarities.
    """

    # Clean GPT output
    gpt_output = gpt_output_cleaner(gpt_output_path)

    # Get all possible item pairs
    pair_list = item_pairer(gpt_output, n_alts)

    # Get cosine similarities for all item pairs
    pair_similarities = pair_similarity_computer(gpt_output, pair_list)

    # Get all possible item combinations and their median similarities
    combo_sem_sims = combo_finder(gpt_output, pair_similarities, n_alts)

    # Write to CSV
    if write:
        # Get scale name and output path
        if output_path is None:
            scale_name = re.search(r"/([^/]+)$", str(gpt_output_path)).group(1)
            output_path = Path.cwd() / "processed"
        else:
            output_path = Path(output_path)
        
        # Create output directory if it doesn't exist
        if not output_path.exists():
            output_path.mkdir()
        
        # Create output txt path
        output_path = output_path / scale_name

        # Get item combination with lowest median similarity
        combo_lowest = min(combo_sem_sims, key = combo_sem_sims.get)
        output_items = [gpt_output[item] for item in combo_lowest]

        # Write to txt
        with open(output_path, 'w') as f:
            f.write('\n'.join(output_items))

    return combo_sem_sims


# Get paths to all loose GPT output files
loose_dir = Path.cwd() / "loose_alts"
loose_paths = [file_path for file_path in loose_dir.glob("*.txt")]

for path in loose_paths:
    # Get semantic similarities of all item combinations
    scale_alt_sem_sims = semantic_separator(path)

    # Write to CSV
    dict_name = re.search(r"/([^/.]+)\.[^.]+$", str(path)).group(1)
    output_path = Path.cwd() / "processed" / (dict_name + ".csv")
    pd.Series(np.round(np.fromiter(scale_alt_sem_sims.values(), dtype=float), 8),name = "median_sem_sim").\
        to_csv(output_path, index = False)

    # output_path = Path.cwd() / "processed" / (dict_name + ".json")
    # with open(output_path, 'w') as f:
    #     json_obj = json.dumps({str(key): value for key, value in scale_alt_sem_sims.items()},
    #                           indent = 4)
    #     f.write(json_obj)

    # Write to pickle
    output_path = Path.cwd() / "processed" / (dict_name + ".pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(pd.Series(np.round(np.fromiter(scale_alt_sem_sims.values(), dtype=float), 4), name = "median_sem_sim"), f)

