# stolen from Gustaw Opielka on 2023/03/03
# edited by Konrad Mikalauskas on 2023/03/12

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

from load_guse import load_model

def similarity(model, captions, metric = 'cosine', show = False, title = "Main", labels = None):
    caption_embedding = model(captions)
    cos_sim_mat = 1 - pairwise_distances(caption_embedding, metric=metric)

    if show == True:
        # make plot
        fig, ax = plt.subplots()
        im = ax.imshow(cos_sim_mat)

        # create default sentence labels
        if labels == None:
            labels = ['S' + str(i) for i in range(1, len(captions)+1)]
        
        # add ticks and labels
        ax.set_title("Semantic Similarity Matrix ({0})".format(title), fontsize=15)
        cbar = plt.colorbar(im, ax = ax)
        cbar.ax.set_ylabel('Semantic Similarity', fontsize=15)
        ax.set_xticks(range(len(captions)), labels, rotation=60, fontsize=8)
        ax.set_yticks(range(len(captions)), labels, rotation='horizontal', fontsize=8)

        # add cell values as text
        for x in range(len(captions)):
            for y in range(len(captions)):
                text = ax.text(y, x, round(cos_sim_mat[x, y], 2),
                                ha="center", va="center", color="w")
        
        fig.tight_layout() # make everything nicely fit in window
        plt.show() # show

    return cos_sim_mat, caption_embedding

def cronbach_alpha(corr_mat):
    N = len(corr_mat)
    mean_r = np.mean(corr_mat[np.triu_indices(len(corr_mat),1)])
    alpha = (N * mean_r) / (1 + (N - 1) * mean_r)

    return alpha

# load model
model = load_model()

## items
item_measures = ['Anxiety', 'Depression', 'Cog./Beh. Control', 'Pos. Affect #1', 'Pos. Affect #2']
item_start = "How much of the time during the past month have you "

# original items
items_og = ["been a very anxious person?",
            "felt downhearted and blue?",
            "felt so down in the dumps nothing could cheer you up?",
            "been a happy person?",
            "felt calm and peaceful?"]
# alternative form items
items_af = ["felt tense or high strung?",
            "been in low or very low spirits?",
            "felt you had nothing to look forward to?",
            "generally enjoyed the things you do?",
            "felt cheerful, lighthearted?"]
# chatGPT loosely paraphrased items
items_lp = ["experienced feelings of excessive worry or nervousness?",
            "felt disheartened or discouraged?",
            "experienced persistent feelings of sadness that nothing seems to alleviate?",
            "felt content and satisfied with your life?",
            "felt relaxed or at ease in your daily activities?"]

items_og = [item_start + item for item in items_og]
items_af = [item_start + item for item in items_af]
items_lp = [item_start + item for item in items_lp]

og_ssm = similarity(model, items_og, 'cosine', True, "OG", item_measures)[0]
af_ssm = similarity(model, items_af, 'cosine', True, "AF", item_measures)[0]
lp_ssm = similarity(model, items_lp, 'cosine', True, "LP", item_measures)[0]

og_ssm_upper_tri = og_ssm[np.triu_indices(len(og_ssm),1)]
af_ssm_upper_tri = af_ssm[np.triu_indices(len(af_ssm),1)]
lp_ssm_upper_tri = lp_ssm[np.triu_indices(len(lp_ssm),1)]

og_ssm_alpha = cronbach_alpha(og_ssm)
af_ssm_alpha = cronbach_alpha(af_ssm)
lp_ssm_alpha = cronbach_alpha(lp_ssm)


