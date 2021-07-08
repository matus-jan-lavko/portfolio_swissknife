import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

#set global style
plt.style.use('bmh')

def plot_returns(df, r_benchmark, ax = None, *args, **kwargs):
    if ax is None:
        ax = plt.gca()
        fig = plt.gcf()
        fig.set_figheight(10)
        fig.set_figwidth(13)

    ax.set_title('Cumulative Returns')

    labels = df.columns.to_list()
    labels += ['Benchmark']
    dates = df.index.to_list()

    ax.plot(dates, df, *args, **kwargs)
    ax.plot(dates, r_benchmark, alpha=0.8, c='gray', linestyle='--')
    ax.legend(labels)
    fig = plt.gcf()
    fig.tight_layout()
    return ax

def plot_weights(weights_dict: dict, models: list,
                 num_rows = 2, ax = None, *args, **kwargs):

    fig, axes = plt.subplots(num_rows, 2)
    fig.set_figheight(4*num_rows)
    fig.set_figwidth(13)

    #don't show ew
    if 'EW' in models:
        models.remove('EW')

    for idx, mod in enumerate(models):
        _plot_stacked_weights(weights_dict[mod], mod, ax = axes.ravel()[idx])

        #turn of the last axs
        if (idx+1) == len(models) and (idx+1) % 2 != 0:
            axes.ravel()[idx+1].set_visible(False)

    labels = weights_dict[models[0]].columns.to_list()
    fig.legend(labels[::-1], loc = 'center left', bbox_to_anchor=(1, 0.5))
    fig = plt.gcf()
    fig.tight_layout()

def _plot_stacked_weights(df, model: str, ax = None, *args, **kwargs):
    if ax is None:
        ax = plt.gca()
        fig = plt.gcf()

    ax.set_title(f"{model} weights structure")
    labels = df.columns.tolist()

    colormap = cm.get_cmap('tab20')
    colormap = colormap(np.linspace(0, 1, 20))

    cycle = plt.cycler("color", colormap)
    ax.set_prop_cycle(cycle)
    X = df.index.tolist()

    ax.stackplot(X, np.vstack(df.values.T), labels=labels, alpha=0.7, edgecolor="black")

    ax.set_ylim(0, 1)
    ax.set_xlim(0, X[-1])

    ticks_loc = ax.get_yticks().tolist()
    ax.set_yticks(ax.get_yticks().tolist())
    ax.set_yticklabels(["{:3.2%}".format(x) for x in ticks_loc])
    ax.grid(linestyle=":")
    n = int(np.ceil(len(labels) / 4))
    fig = plt.gcf()
    fig.tight_layout()
    return ax