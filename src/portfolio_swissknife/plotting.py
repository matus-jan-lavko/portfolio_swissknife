import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime as dt

#set global style
plt.style.use('bmh')

def plot_rolling_beta(df):
    '''
    Plots the rolling betas.

    :param df: dataframe of betas: pd.DataFrame
    :return: None
    '''
    num_plots = 0
    fig, axes = plt.subplots(figsize=(15,12))
    for col in df:
        num_plots += 1
        plt.subplot(3, 3, num_plots)
        for v in df:
            plt.plot(df.index, df[v], marker='', linewidth=0.8, alpha=0.3)
            plt.axhline(0, color='gray', alpha=0.3, linewidth=0.5)

        plt.ylim(-1.8, 1.8)
        plt.plot(df.index, df[col], color='firebrick', linewidth=2.4, alpha=1, label=col)
        plt.xticks(rotation=50)
        plt.title(col)
        plt.tight_layout()
    fig = plt.gcf()
    fig.tight_layout()

def plot_returns(df, r_benchmark, ax = None, title = None, *args, **kwargs):
    '''
    Plots returns together with a benchmark.

    :param df: table of returns: pd.DataFrame
    :param r_benchmark: series of benchmark returns: pd.Series, pd.DataFrame
    :param ax: axis to be plotted on: plt.axis
    :param title: name to be displayed: str
    :return: plt.axis
    '''
    if ax is None:
        ax = plt.gca()
        fig = plt.gcf()
        fig.set_figheight(10)
        fig.set_figwidth(13)

    text = 'Cumulative Returns'

    if title:
        text = text  + ' ' + title

    ax.set_title(text)

    labels = df.columns.to_list()
    labels += ['Benchmark']
    dates = df.index.to_list()

    ax.plot(dates, df)
    ax.plot(dates, r_benchmark, alpha=0.8, c='gray', linestyle='--')
    ax.legend(labels)

    fig = plt.gcf()
    fig.tight_layout()
    return ax

def plot_weights(weights_dict: dict, models: list,
                 num_rows = 2, ax = None, *args, **kwargs):
    '''
    Plots the weights plots of a fixed number of assets and their development according to a weighing model.

    :param weights_dict: dictionary of weights of all models to be plotted: dict
    :param models: names of models to be plotted: list
    :param num_rows: number of rows of plots: int
    :param ax: axis to be plotted on: plt.axis
    :return: None
    '''

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
    '''
    Helper function for plotting stacked weights plots.

    :param df: dataframe of weights: pd.DataFrame
    :param model: name of the model: str
    :param ax: axis to be plotted on: plt.axis
    :return: plt.axis
    '''
    if ax is None:
        ax = plt.gca()
        fig = plt.gcf()

    ax.set_title(f"{model} weights structure")
    labels = df.columns.tolist()

    colormap = cm.get_cmap('tab20')
    colormap = colormap(np.linspace(0, 1, 20))

    cycle = plt.cycler("color", colormap)
    ax.set_prop_cycle(cycle)
    X = df.index

    ax.stackplot(X, np.vstack(df.values.T), labels=labels, alpha=0.7, edgecolor="black")

    ax.set_ylim(0, 1)
    ax.set_xlim(X[0],X[-1])

    ticks_loc = ax.get_yticks().tolist()
    ax.set_yticks(ax.get_yticks().tolist())
    ax.set_yticklabels(["{:3.2%}".format(x) for x in ticks_loc])
    ax.grid(linestyle=":")
    n = int(np.ceil(len(labels) / 4))
    fig = plt.gcf()
    fig.tight_layout()
    return ax

def plot_implied_volatility(df, s_0, op_type = None):
    if op_type == 'call':
        df1 = df.loc[df['call'] == 1]
    elif op_type == 'put':
        df1 = df.loc[df['call'] == 0]
    else:
        df1 = df

    fig = plt.figure(1, figsize=(15, 10))
    cols = 2
    rows = 4 // cols
    rows += 4 % cols
    position = range(1, 4 + 1)


    ax = fig.add_subplot(rows, cols, position[0])
    ax.set_xlim(min(df1['strike']), max(df1['strike']))
    ax.set_ylim(min(df1['impliedVolatility']), max(df1['impliedVolatility']))

    ax.set_title('Short < 30d')
    count = 0
    for name, gr in df1.groupby('expirationDate'):
        tenor = (name - dt.today()).days
        grp = gr.sort_values(by='strike')
        if tenor > 30 and count < 1:
            count += 1
            ax = fig.add_subplot(rows, cols, position[count])
            ax.set_title('Medium < 90d')
            ax.set_xlim(min(df1['strike']), max(df1['strike']))
            ax.set_ylim(min(df1['impliedVolatility']), max(df1['impliedVolatility']))
            legend = []

        if tenor > 91 and count < 2:
            count += 1
            ax = fig.add_subplot(rows, cols, position[count])
            ax.set_title('Long < 252d')
            ax.set_xlim(min(df1['strike']), max(df1['strike']))
            ax.set_ylim(min(df1['impliedVolatility']), max(df1['impliedVolatility']))

            legend = []
        if tenor > 252 and count < 3:
            count += 1
            ax = fig.add_subplot(rows, cols, position[count])
            ax.set_title('Very long > 252d')
            ax.set_xlim(min(df1['strike']), max(df1['strike']))
            ax.set_ylim(min(df1['impliedVolatility']), max(df1['impliedVolatility']))

            legend = []

        ax.scatter(grp['strike'], grp['impliedVolatility'], s=15)
        ax.plot(grp['strike'], grp['impliedVolatility'], linewidth=0.5, label='_nolegend_')

        legend.append(name.strftime('%d-%b-%Y'))
        ax.axvline(s_0[0], color='black', linewidth=1.4, alpha=0.7, linestyle='--', label='_nolegend_')
        ax.legend(legend)

