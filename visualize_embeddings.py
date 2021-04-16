import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

def generate_embedding_plot(model, preprocessed):
    # getting embeddings from the embedding layer of our model, by name
    embeddings = model.in_embed.weight.to('cpu').data.numpy()

    viz_words = 380
    tsne = TSNE()
    embed_tsne = tsne.fit_transform(embeddings[:viz_words, :])

    fig, ax = plt.subplots(figsize=(16, 16))
    for idx in range(viz_words):
        plt.scatter(*embed_tsne[idx, :], color='steelblue')
        plt.annotate(preprocessed['int_to_vocab'][idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)