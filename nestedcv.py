from sklearn.model_selection import KFold
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import matplotlib.ticker as ticker
from plotly import offline
import plotly.graph_objs as go
from sklearn.metrics.scorer import SCORERS


def param_grid_fn(param_grid=None):
    items = sorted(param_grid.items())
    keys, values = zip(*items)

    for v in product(*values):
        yield dict(zip(keys, v))


def feature_imp_finder(features=None, importance=None):
    d = dict(list(map(lambda x: x.split('_') if '_' in x else [x, '0'], features)))
    d = dict(map(lambda x: (x[0], int(x[1]) + 1), d.items()))
    c = 0
    for k, v in d.items():
        slice_ = importance[c: c+v]
        c += v
        d[k] = np.mean(slice_).round(2)
    return pd.DataFrame([d])

# f_importance = algorithm.feature_importances_.tolist()
# features = dataset.columns
#
# features, f_importance = feature_imp_finder(features=features,
#                                             importance=f_importance)


def func(col):
    if max(col) < 0.1:
        col *= 2
    return col


def plotly_cv_curve(x=None, y=None, std=None):
    y_lower = y - std
    y_upper = y + std

    # ax = sns.lineplot(x="timepoint", y="signal", data='')

    upper_bound = go.Scatter(
        name='Upper Bound',
        x=x,
        y=y_upper,
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty')

    trace = go.Scatter(
        name='Calculation',
        x=x,
        y=y,
        mode='lines',
        line=dict(color='rgb(31, 119, 180)'),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty')

    lower_bound = go.Scatter(
        name='Lower Bound',
        x=x,
        y=y_lower,
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines')
    data = [lower_bound, trace, upper_bound]

    layout = go.Layout(
        yaxis=dict(title='CV', range=[0.4, 1.0]),
        xaxis=dict(title='Grid parameters'),
        title='Continuous, variable value error bars.<br>Notice the hover text!',
        showlegend=False)
    fig = go.Figure(data=data, layout=layout)
    offline.plot(fig, filename='name.html')


def shuffle(x=None, y=None):
    arr = np.random.permutation(y.shape[0])
    if isinstance(x, pd.DataFrame) and isinstance(y, pd.Series):
        return x.loc[arr, :].reset_index(drop=True), y[arr].reset_index(drop=True)
    return x[arr], y[arr]


def repeated_grid_search_cv(nexp=None, estimator=None, v=None,
                            x=None, y=None, param_grid=None, scoring=None):
    scores = []
    feature_importance = []

    for _ in range(nexp):
        kf = KFold(n_splits=v, shuffle=True, random_state=_)
        for train_idx, test_idx in kf.split(x):
            if isinstance(x, pd.DataFrame):
                x_train, y_train = x.values[train_idx], y.values[train_idx]
                x_test, y_test = x.values[test_idx, :], y.values[test_idx]
            else:
                x_train, y_train = x[train_idx], y[train_idx]
                x_test, y_test = x[test_idx], y[test_idx]

            tem_scores = []
            for param in param_grid_fn(param_grid):
                model = estimator.__class__(**param).fit(x_train, y_train)
                score = SCORERS[scoring](model, x_test, y_test)
                tem_scores.append(score)
            scores.append(tem_scores)

    for param in param_grid_fn(param_grid):
        model = estimator.__class__(**param).fit(x, y)
        f_importance = model.feature_importances_.tolist()
        f_imp_df = feature_imp_finder(features=x.columns, importance=f_importance)
        feature_importance.append(f_imp_df)
    dict_ = {'accuracy': 'Accuracy', 'roc_auc': 'ROC_Auc', 'average_precision': 'PR_Auc'}

    importance_df = pd.concat(feature_importance)
    importance_df = importance_df.reset_index(drop=True)
    importance_df = importance_df.set_index(importance_df.index + 1)
    importance_df = importance_df.apply(func, axis=0)
    ax_1 = importance_df[['Credit amount', 'Credit history', 'Housing', 'Age']].plot(subplots=True, figsize=(6, 6), marker='o', fontsize=14)
    ax_1[0].xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax_1[-1].set_xlabel("Grid parameters", size=15)
    ax_1[0].set_ylabel("Importance", size=14)
    ax_1[1].set_ylabel("Importance", size=14)
    ax_1[2].set_ylabel("Importance", size=14)
    ax_1[3].set_ylabel("Importance", size=14)
    fig_1 = ax_1[0].get_figure()
    fig_1.tight_layout()
    fig_1.savefig(f"graphs/features_imp_{dict_[scoring]}.png")
    plt.close(fig_1)

    scores = np.array(scores)
    df = pd.DataFrame(scores, columns=list(range(scores.shape[1])))
    df = df.unstack().reset_index().rename(columns={0: 'Accuracy', 'level_0': 'Grid parameters'})
    df['Grid parameters'] = df['Grid parameters'] + 1
    sns.set(font='sans-serif')
    sns.set(font_scale=1.4)
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
    ax_2 = sns.lineplot(x="Grid parameters", y="Accuracy", data=df, ci="sd", marker="o")
    ax_2.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax_2.set_xlabel("Grid space", fontsize=19)
    ax_2.set_ylabel(dict_[scoring], fontsize=19)
    plt.xticks(rotation=30)
    # ax.axes.set_title("Title", fontsize=50)
    fig_2 = ax_2.get_figure()
    fig_2.tight_layout()
    fig_2.savefig(f"graphs/grid_search_{dict_[scoring]}.png")
    plt.close(fig_2)
    return None
    # best_params = params[scores.mean(axis=0).argmax()]
    # return best_params, scores.mean(axis=0), scores.std(axis=0)


def repeated_nested_cv(estimator=None, param_grid=None, v2=None, v1=None,
                       x=None, y=None, nexp1=None, nexp2=None, scoring=None):

    outer_score = []
    # means = []
    for _ in range(nexp2):
        kf = KFold(n_splits=v2, shuffle=True, random_state=_)
        for train_idx, test_idx in kf.split(x):
            if isinstance(x, pd.DataFrame):
                x_train, y_train = x.loc[train_idx, :].reset_index(drop=True), y.loc[train_idx].reset_index(drop=True)
                x_test, y_test = x.loc[test_idx, :].reset_index(drop=True), y.loc[test_idx].reset_index(drop=True)
            else:
                x_train, y_train = x[train_idx], y[train_idx]
                x_test, y_test = x[test_idx], y[test_idx]

            best_params, best_mean, __ = repeated_grid_search_cv(nexp=nexp1, estimator=estimator,
                                                                 param_grid=param_grid, x=x_train,
                                                                 y=y_train, v=v1, scoring=scoring)
            # means.append(best_mean)
            model = estimator.__class__(**best_params).fit(x_train, y_train)
            score = SCORERS[scoring](model, x_test, y_test)
            outer_score.append(score)

    # means = np.array(means).mean(axis=0)
    # print(max(means))
    # print('---------')
    # print(np.mean(outer_score))
    return np.mean(outer_score)


if __name__ == '__main__':
    x = pd.read_csv('german_encoded.csv')
    y = x.pop('class')
    x, y = shuffle(x, y)
    nexp1 = 20
    nexp2 = 1
    v1 = 10
    v2 = 5
    estimator = RandomForestClassifier()
    param_grid = {'n_estimators': [50, 100, 300],
                  'max_features': ['sqrt', 'log2', None],
                  'min_samples_leaf': [1, 50, 100]}

    p = ['roc_auc', 'average_precision']
    for i in p:
        repeated_grid_search_cv(nexp=nexp1, v=v1, param_grid=param_grid,
                                estimator=estimator, x=x, y=y,  scoring=i)
    # plotly_cv_curve(x=list(range(len(std))), y=mean, std=std)
    # adding f_imp for 5 features over tha param_grid space

    # repeated_nested_cv(estimator=estimator, param_grid=param_grid, scoring='accuracy',
    #                    v2=v2, v1=v1, x=x, y=y, nexp1=nexp1, nexp2=nexp2)

