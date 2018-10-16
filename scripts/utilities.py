import numpy as np
import pickle
from pathlib import Path

def build_pairs(df, save=False):
    """
    Builds pairs from the fanfics in a DataFrame.

    Args:
    df - pandas.DataFrame - A DataFrame with fanfic metadata.
    save - bool/str - if string, the filename where the pairs should be pickled to. If False, no pickling happens.
    """
    pairs = set()

    for index, row in df.iterrows():

        rel_condition = False
        if row.relationship:      # handle NaNs
            rels = row.relationship.split(', ')
            rel_condition = df.relationship.str.startswith(rels[0])

        similars = df[rel_condition & (df.work_id != row.work_id) & (df.rating == row.rating) 
                      & (df.published == row.published)]
        if not similars.empty:
            for i, r in similars.iterrows():
                if row.work_id > r.work_id:      # a way to avoid having the same pair more than once
                    pairs.add((r.work_id, row.work_id))
                else:
                    pairs.add((row.work_id, r.work_id))

    if save:
        output_dir = Path(save)
        with open(save, 'wb') as s:
            pickle.dump(pairs, s)

    return pairs


def convert_to_pairwise(vectors, ids, id_label_dict, pairs, other_vectors=None, min_diff=1):
    """
    Transforms the data so that binary classification is possible.
    
    Args:
    vectors - np.array - Feature vectors.
    ids - np.array - The corresponding fanfic IDs
    pairs - set - Time and topic controlled pairs, as extracted by build_pairs()
    other_vectors - np.array - Other feature vectors. The order of the instances is assumed to be the same as vectors.
    min_diff - int - Pairs with score difference < min_diff are skipped. Default is 1, which skips pairs with score difference = 0.
    
    Returns:
    X - np.array - The new feature vectors, each representing a pair of instances.
    y - np.array - The new labels -> -1/+1
    diff - np.array - The score differences for each pair in X.
    """
    
    if other_vectors is not None:
        id_vector_dict = dict(zip(ids, np.concatenate((vectors, other_vectors), axis=1)))
    else:
        id_vector_dict = dict(zip(ids, vectors))
        
    k = 0
    X, y = [], []

    for fic1, fic2 in pairs:
        
        if abs(id_label_dict[fic1] - id_label_dict[fic2]) < min_diff:   # skip pairs with too small score difference
            continue
            
        X.append(id_vector_dict[fic1] - id_vector_dict[fic2])
        y.append(np.sign(id_label_dict[fic1] - id_label_dict[fic2]))
        
        # in order to output balanced classes - labels take the form [-1, 1, -1, 1 ...]
        if y[-1] != (-1) ** k:
            y[-1] *= -1
            X[-1] *= -1
            
        k += 1

    return map(np.asanyarray, (X, y))


def reader(directory, fandom_abbr):
    """
    Reads all files needed for the ML experiment.

    Args:
    directory - str - Directory of the saved files.
    fandom_abbr - str - mcu/vld/mha

    Returns:
    loaded - list - The five files. Order: vectors, ids, labels, pairs, bow_tags.
    """
    d = Path(directory)
    files = ['{}_vectors.p', '{}_ids.p', '{}_labels.p', 'tc_{}_pairs.p', '{}_bow_tags.p']
    loaded = []
    
    for f in files:
        with (d / f.format(fandom_abbr)).open('rb') as v:
            loaded.append(pickle.load(v))

    return loaded