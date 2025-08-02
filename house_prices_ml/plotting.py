import matplotlib.pyplot as plt
import seaborn as sns


def countplot_with_order(data, **kwargs):
    order = data["value"].value_counts().index
    sns.countplot(data=data, x="value", order=order, **kwargs)


def plot_num_features(df, num_cols):
    df = df[num_cols].copy()
    df = df.melt(var_name="feature", value_name="value")

    g = sns.FacetGrid(df, col="feature", col_wrap=3, sharex=False, sharey=False)
    g.map(sns.histplot, "value", kde=True)

    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


def plot_cat_features(df, cat_cols):
    df = df[cat_cols].copy()
    df = df.melt(var_name="feature", value_name="value")

    g = sns.FacetGrid(df, col="feature", col_wrap=3, sharex=False, sharey=False)
    g.map_dataframe(countplot_with_order)

    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


def plot_num_vs_target(df, num_cols, target):
    df = df[num_cols + [target]].copy()
    df = df.melt(id_vars=target, value_vars=num_cols, var_name="feature", value_name="value")

    g = sns.FacetGrid(df, col="feature", col_wrap=3, sharex=False, sharey=False)
    g.map(sns.regplot, "value", target, scatter_kws={"alpha": 0.4}, line_kws={"color": "red"})

    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()


def plot_cat_vs_target(df, cat_cols, target):
    df = df[cat_cols + [target]].copy()
    df = df.fillna("Missing")
    df = df.melt(id_vars=target, value_vars=cat_cols, var_name="feature", value_name="value")

    g = sns.FacetGrid(df, col="feature", col_wrap=4, sharey=True, sharex=False)
    g.map(sns.boxplot, "value", target, order=None)

    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()
