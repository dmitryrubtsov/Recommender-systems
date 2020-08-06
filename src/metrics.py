import pandas as pd
import numpy as np
import swifter


def money_precision_at_k(y_pred: pd.Series, y_true: pd.Series, item_price, k=5):
    y_pred = y_pred.swifter.progress_bar(False).apply(pd.Series)
    user_filter = ~(y_true.swifter.progress_bar(False).apply(len) < k)

    y_pred = y_pred.loc[user_filter]
    y_true = y_true.loc[user_filter]

    prices_recommended = y_pred.swifter.progress_bar(False).applymap(lambda item: item_price.price.get(item))
    flags = y_pred.loc[:, :k - 1].swifter.progress_bar(False) \
        .apply(lambda row: np.isin(np.array(row), y_true.get(row.name)), axis=1) \
        .swifter.progress_bar(False).apply(pd.Series)

    metric = (
        (flags * prices_recommended.loc[:, :k - 1]).sum(axis=1) / prices_recommended.loc[:, :k - 1].sum(axis=1)
    ).mean()
    return metric


