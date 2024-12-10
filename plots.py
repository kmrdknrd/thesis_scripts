## Created by Konrad Mikalauskas on 2023-03-12

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json

from pathlib import Path
from sklearn.metrics import pairwise_distances
from scipy.stats import median_abs_deviation as mad

from load_guse import load_model

# scale_alt_sem_sims_json_path = Path("processed/Barchard-PE_IPIP-9.json")

def similarity(model, captions, metric = 'cosine', plot = False, title = "Main", labels = None):
    """
    Plots a similarity matrix of the caption embeddings using a given embedding model.
    
    Parameters:
        model (tensorflow_hub.KerasLayer): The embedding model to use.
        captions (list): A list of captions to embed and compare.
        metric (str): The metric to use for calculating the similarity matrix.
        plot (bool): Whether to plot the plot or not.
        title (str): The title of the plot.
        labels (list): A list of labels for the captions.
    
    Returns:
        cos_sim_mat (numpy.ndarray): The similarity matrix.
        caption_embedding (numpy.ndarray): The caption embeddings.
    """

    caption_embedding = model(captions)
    cos_sim_mat = 1 - pairwise_distances(caption_embedding, metric=metric)

    if plot == True:
        # Make plot
        fig, ax = plt.subplots()
        im = ax.imshow(cos_sim_mat)

        # Create default sentence labels
        if labels == None:
            labels = ['S' + str(i) for i in range(1, len(captions)+1)]
        
        # Add ticks and labels
        ax.set_title("Semantic Similarity Matrix ({0})".format(title), fontsize=15)
        cbar = plt.colorbar(im, ax = ax)
        cbar.ax.set_ylabel('Semantic Similarity', fontsize=15)
        ax.set_xticks(range(len(captions)), labels, rotation=60, fontsize=8)
        ax.set_yticks(range(len(captions)), labels, rotation='horizontal', fontsize=8)

        # Add cell values as text
        for x in range(len(captions)):
            for y in range(len(captions)):
                text = ax.text(y, x, round(cos_sim_mat[x, y], 2),
                                ha="center", va="center", color="w")
        
        fig.tight_layout() # make everything nicely fit in window
        plt.show() # show

    return cos_sim_mat, caption_embedding

def semantic_overlap_range_plot(scale_alt_sem_sims_json_path, og_ssm_median = None, sp_ssm_median = None, lp_ssm_median = None):
    """
    Plots the semantic similarity of the original scale alternatives, the alternatives from the full scale, and the alternatives from the full scale with the original scale alternatives removed.
    
    Parameters:
        scale_alt_sem_sims_json_path (str): The path to the JSON file containing the semantic similarities of the loose scale alternatives.
        og_ssm_median (float): The median semantic similarity of the original scale items.
        sp_ssm_median (float): The median semantic similarity of the strictly paraphrased original scale items.
        lp_ssm_median (float): The median semantic similarity of the loosely paraphrased original scale items.

    Returns:
        None
    """


    # Read in the scale alternatives semantic similarities
    with open(scale_alt_sem_sims_json_path, 'r') as f:
        scale_alt_sem_sims = json.load(f)

    # Dictionary keys to numpy array
    scale_alt_sem_sims_arr = np.fromiter(scale_alt_sem_sims.values(), dtype = float)

    # Calculate the median of the semantic similarities
    median_sem_sims = np.median(scale_alt_sem_sims_arr)

    # Plot the distribution of semantic similarities
    plt.hist(scale_alt_sem_sims_arr, bins = 40, color = 'blue', edgecolor = 'black', density = True)
    plt.title("Semantic Similarities of Scale Alternatives")
    plt.xlabel("Semantic Similarity")
    plt.ylabel("Density")
    plt.axvline(x = median_sem_sims, color = 'red', label = 'Median')

    if og_ssm_median != None:
        plt.axvline(x = og_ssm_median, color = 'green', label = 'Original SemSim')

    if sp_ssm_median != None:
        plt.axvline(x = sp_ssm_median, color = 'magenta', label = 'Strict-Par. SemSim')

    if lp_ssm_median != None:
        plt.axvline(x = lp_ssm_median, color = 'yellow', label = 'Loose-Par. SemSim')

    plt.legend()
    plt.savefig(f"plots/{scale_dir.name}_range_plot.pdf", dpi = 300, format = "pdf")

def cronbach_alpha(corr_mat):
    """
    Calculates the Cronbach's alpha of a correlation matrix.

    Parameters:
        corr_mat (numpy.ndarray): The correlation matrix.

    Returns:
        alpha (float): The Cronbach's alpha of the correlation matrix.
    """

    N = len(corr_mat)
    mean_r = np.mean(corr_mat[np.triu_indices(len(corr_mat),1)])
    alpha = (N * mean_r) / (1 + (N - 1) * mean_r)

    return alpha

def plot_ssm(ax, data, title, labels, min_val, max_val):
    ax.imshow(data, cmap='viridis', vmin=min_val, vmax=max_val, interpolation='nearest')
    ax.set_title(title)
    ax.set_xticks(range(len(labels)), labels, rotation=45, fontsize=8)
    ax.set_yticks(range(len(labels)), labels, rotation='horizontal', fontsize=8)

# Load the GUSE model
try:
    model(["hello", "goodbye"])
except:
    model = load_model()

# Read in the scale alternatives semantic similarities
final_scales_dir = Path("final_scales")

# Create DataFrame for storing various scale statistics
scale_stats_df = pd.DataFrame(columns = ["Scale", "Version", "Median Semantic Similarity", "MAD", "Cronbach's Alpha"])

# Get SSMs for the five different scales, and the three different versions of each scale + plot semantic range plots for each scale
for scale_dir in final_scales_dir.iterdir():
    if scale_dir.is_dir():
        print(scale_dir)
        # Read in the three different versions per scale
        for scale_file in scale_dir.iterdir():
            print(scale_file)
            if "_OG" in scale_file.name:
                og_scale = open(scale_file, 'r').read().splitlines()
            elif "_SP" in scale_file.name:
                sp_scale = open(scale_file, 'r').read().splitlines()
            elif "_LP" in scale_file.name:
                lp_scale = open(scale_file, 'r').read().splitlines()
            elif scale_file.suffix == ".json":
                scale_alt_sem_sims_json_path = scale_file

        # Compute the semantic similarity matrices of the three versions
        og_ssm, og_embeddings = similarity(model, og_scale, plot = False, title = scale_dir.name + " Original")
        sp_ssm, sp_embeddings = similarity(model, sp_scale, plot = False, title= scale_dir.name + " Strict")
        lp_ssm, lp_embeddings = similarity(model, lp_scale, plot = False, title= scale_dir.name + " Loose")

        # Write the semantic similarity matrices to CSV files
        ssm_dir = Path("ssms")
        ssm_dir.mkdir(exist_ok = True)
        pd.DataFrame(og_ssm).to_csv(ssm_dir / f"{scale_dir.name}_OG_ssm.csv", index = False, header = False)
        pd.DataFrame(sp_ssm).to_csv(ssm_dir / f"{scale_dir.name}_SP_ssm.csv", index = False, header = False)
        pd.DataFrame(lp_ssm).to_csv(ssm_dir / f"{scale_dir.name}_LP_ssm.csv", index = False, header = False)

        # Calculate the "semantic Cronbach's alpha" of the SSMs
        og_ssm_alpha = cronbach_alpha(og_ssm)
        sp_ssm_alpha = cronbach_alpha(sp_ssm)
        lp_ssm_alpha = cronbach_alpha(lp_ssm)


        ## Plotting
        # Plot the three SSMs together
        fig, axes = plt.subplots(1, 3, figsize = (15, 5))
        fig.suptitle(f"Semantic Similarity Matrices ({scale_dir.name})", fontsize=15)

        # Get min/max values for consistent coloring
        min_val = min(og_ssm.min(), sp_ssm.min(), lp_ssm.min())
        max_val = max(og_ssm.max(), sp_ssm.max(), lp_ssm.max())

        # Create labels
        labels = ['Item ' + str(i) for i in range(1, len(og_ssm)+1)]

        # Plotting the heatmaps
        data_sources = [(og_ssm, "Original Items"), (sp_ssm, "Strict Paraphrases"), (lp_ssm, "Loose Paraphrases")]
        for ax, (data, title) in zip(axes, data_sources):
            plot_ssm(ax, data, title, labels, min_val, max_val)

        # Add colorbar
        fig.subplots_adjust(right = 0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(axes[-1].images[0], cax=cbar_ax) # axes[-1] is the last axis

        plt.savefig(f"plots/{scale_dir.name}_SSMs.pdf", dpi = 300, format = "pdf")
        # plt.show()
        ## End of plotting


        # Get the upper triangular part of the semantic similarity matrices
        og_ssm_upper_tri = og_ssm[np.triu_indices(len(og_ssm),1)]
        sp_ssm_upper_tri = sp_ssm[np.triu_indices(len(sp_ssm),1)]
        lp_ssm_upper_tri = lp_ssm[np.triu_indices(len(lp_ssm),1)]

        # Calculate the medians of the SSMs
        og_ssm_median = np.median(og_ssm_upper_tri)
        sp_ssm_median = np.median(sp_ssm_upper_tri)
        lp_ssm_median = np.median(lp_ssm_upper_tri)

        # Calculate the median absolute deviations of the SSMs
        og_ssm_mad = mad(og_ssm_upper_tri)
        sp_ssm_mad = mad(sp_ssm_upper_tri)
        lp_ssm_mad = mad(lp_ssm_upper_tri)

        # Fill in the scale statistics DataFrame
        scale_stats_df = scale_stats_df.append({"Scale": scale_dir.name,
                                                "Version": "Original",
                                                "Median Semantic Similarity": og_ssm_median,
                                                "MAD": og_ssm_mad,
                                                "Cronbach's Alpha": og_ssm_alpha},
                                                ignore_index = True)
        scale_stats_df = scale_stats_df.append({"Scale": scale_dir.name,
                                                "Version": "Strict",
                                                "Median Semantic Similarity": sp_ssm_median,
                                                "MAD": sp_ssm_mad,
                                                "Cronbach's Alpha": sp_ssm_alpha},
                                                ignore_index = True)
        scale_stats_df = scale_stats_df.append({"Scale": scale_dir.name,
                                                "Version": "Loose",
                                                "Median Semantic Similarity": lp_ssm_median,
                                                "MAD": lp_ssm_mad,
                                                "Cronbach's Alpha": lp_ssm_alpha},
                                                ignore_index = True)

        # Plot semantic range plot
        plt.close('all')
        semantic_overlap_range_plot(scale_alt_sem_sims_json_path, og_ssm_median, sp_ssm_median, lp_ssm_median)
        plt.close('all')

# Save the scale statistics DataFrame
scale_stats_df.to_csv("scale_stats.csv", index = False)

# Plot the median semantic similarity of each scale
plt.close('all')
fig, ax = plt.subplots(figsize = (10, 5))
ax = sns.barplot(x = "Scale", y = "Median Semantic Similarity", hue = "Version", data = scale_stats_df)
ax.set_title("Median Semantic Similarity of Scales")
ax.set_xlabel("Scale")
ax.set_ylabel("Median Semantic Similarity")
plt.tight_layout()
plt.savefig("plots/median_semantic_similarity.pdf", dpi = 300, format = "pdf")
plt.show()

# Plot the cronbach's alpha of each scale
plt.close('all')
fig, ax = plt.subplots(figsize = (10, 5))
ax = sns.barplot(x = "Scale", y = "Cronbach's Alpha", hue = "Version", data = scale_stats_df)
ax.set_title("Semantic Cronbach's Alpha of Scales")
ax.set_xlabel("Scale")
ax.set_ylabel("Semantic Cronbach's Alpha")
plt.tight_layout()
plt.savefig("plots/cronbachs_alpha.pdf", dpi = 300, format = "pdf")
plt.show()