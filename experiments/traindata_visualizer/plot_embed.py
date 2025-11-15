import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import umap.umap_ as umap


def update_annot(ind):
    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = [files[n] for n in ind["ind"]][0]
    annot.set_text(text)
    annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
    annot.get_bbox_patch().set_alpha(0.4)


def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()


if __name__ == "__main__":

    with open('embed.pkl', 'rb') as fp:
        data = pickle.load(fp)

    files = data['files']
    files = [f.replace('_dehydrated.bin', '') for f in files]
    files = [re.sub('_\d{4}\-\d{2}', '', f) for f in files]

    passages = data['passages']
    embeddings = data['embeddings']
    print('files: ' + str(files))
    print('embeddings.shape: ' + str(embeddings.shape))

    color_map = dict()
    c = []
    fileset = set()
    for f in files:
        if f not in color_map:
            color_map[f] = len(color_map)

        c.append(color_map[f])

    print('color_map: ' + str(color_map))

    # c = (np.array(c) / np.max(c) * 255).astype(np.int32).tolist()
    # print('c: ' + str(c))

    umap_obj = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42)
    embedding = umap_obj.fit_transform(embeddings)

    fig, ax = plt.subplots()

    norm = plt.Normalize(1, 4)
    cmap = plt.cm.RdYlGn
    # cmap = plt.cm.hsv

    sc = ax.scatter(embedding[:, 0], embedding[:, 1], s=10, c=c, cmap=cmap, norm=norm, alpha=0.5, label=files)
    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.show()
