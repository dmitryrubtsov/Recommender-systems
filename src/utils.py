import pandas as pd
import swifter
import numpy as np


def optimizing_df(df, silent=False, width_line=100):
    assert isinstance(df, pd.DataFrame), 'This is not a dataframe'

    if not silent:
        start_memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        print('Start of dataframe memory optimization'.center(width_line, '*'))
        print(f'Memory usage by dataframe: {start_memory_usage:.02f} MB')

    df_dtype = pd.DataFrame(df.dtypes, columns=['dtype'], index=df.columns)

    df_dtype['min'] = df.select_dtypes(['int', 'float']).min()
    df_dtype['max'] = df.select_dtypes(['int', 'float']).max()
    df_dtype['is_int'] = ~(df.select_dtypes(['int', 'float']).fillna(0).astype(int) - df.select_dtypes(['int', 'float']).fillna(0)).sum().astype('bool_')

    df_dtype.loc[(df_dtype['is_int'] == True), 'dtype'] = 'int64'
    df_dtype.loc[(df_dtype['is_int'] == True) & (df_dtype['min'] >= np.iinfo('int32').min) & (df_dtype['max'] <= np.iinfo('int32').max), 'dtype'] = 'int32'
    df_dtype.loc[(df_dtype['is_int'] == True) & (df_dtype['min'] >= np.iinfo('int16').min) & (df_dtype['max'] <= np.iinfo('int16').max), 'dtype'] = 'int16'
    df_dtype.loc[(df_dtype['is_int'] == True) & (df_dtype['min'] >= np.iinfo('int8').min) & (df_dtype['max'] <= np.iinfo('int8').max), 'dtype'] = 'int8'

    df_dtype.loc[(df_dtype['is_int'] == True) & (df_dtype['min'] >= np.iinfo('uint64').min) ,'dtype'] = 'uint64'
    df_dtype.loc[(df_dtype['is_int'] == True) & (df_dtype['min'] >= np.iinfo('uint32').min) & (df_dtype['max'] <= np.iinfo('uint32').max), 'dtype'] = 'uint32'
    df_dtype.loc[(df_dtype['is_int'] == True) & (df_dtype['min'] >= np.iinfo('uint16').min) & (df_dtype['max'] <= np.iinfo('uint16').max), 'dtype'] = 'uint16'
    df_dtype.loc[(df_dtype['is_int'] == True) & (df_dtype['min'] >= np.iinfo('uint8').min) & (df_dtype['max'] <= np.iinfo('uint8').max), 'dtype'] = 'uint8'

    df_dtype.loc[(df_dtype['is_int'] == True) & (df_dtype['min'] == 0) & (df_dtype['max'] == 1),'dtype'] = 'bool_'

    df_dtype.loc[(df_dtype['is_int'] == False), 'dtype'] = 'float64'
    df_dtype.loc[(df_dtype['is_int'] == False) & (df_dtype['min'] >= np.finfo('float32').min) & (df_dtype['max'] <= np.finfo('float32').max), 'dtype'] = 'float32'

    for col in df.select_dtypes('object').columns:
        num_unique_values = df[col].nunique()
        num_total_values = df[col].count()
        if num_unique_values / num_total_values < 0.5:
            df_dtype.loc[col, 'dtype'] = 'category'

    dtypes = df_dtype['dtype'].to_dict()

    df = df.astype(dtypes)

    if not silent:
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        print('MEMORY USAGE AFTER COMPLETION:'.center(width_line, '_'))
        print(f'Memory usage of properties dataframe is : {memory_usage:.02f} MB')
        print(f'This is {100 * memory_usage / start_memory_usage:.02f} % of the initial size')
    return df


def postfilter_items(recommendations, item_info=None, user_history=None, n_rec=5, n_new=2, n_exp=1, price_lte=7):
    recommendations = recommendations.swifter.progress_bar(False).apply(pd.Series)

    mask_unique = recommendations.swifter.progress_bar(False) \
        .apply(lambda row: ~pd.Series(row).duplicated(), axis=1)

    mask_sub_commodity = recommendations.fillna(item_info.index.max() + 1) \
        .swifter.progress_bar(False) \
        .applymap(lambda item: item_info.SUB_COMMODITY_DESC.get(item)) \
        .swifter.progress_bar(False) \
        .apply(lambda row: ~pd.Series(row).duplicated(), axis=1)

    mask = mask_unique & mask_sub_commodity

    recommendations = recommendations.where(mask) \
        .swifter.progress_bar(False) \
        .apply(lambda row: np.array(row), axis=1) \
        .swifter.progress_bar(False) \
        .apply(lambda item: item[~np.isnan(item)]) \
        .swifter.progress_bar(False) \
        .apply(pd.Series)
    if user_history is not None:
        rec_new = recommendations.fillna(user_history.index.max() + 1) \
            .swifter.progress_bar(False) \
            .apply(lambda row: ~np.isin(np.array(row), user_history.get(row.name)), axis=1) \
            .swifter.progress_bar(False).apply(pd.Series)
        rec_new_filter = rec_new.loc[rec_new.loc[:, :n_rec - 1].sum(axis=1) < n_new]
        mask_new = rec_new_filter.swifter.progress_bar(False) \
            .apply(lambda row: postfilter_for_item(row, n=n_new, n_rec=n_rec), axis=1).swifter \
            .progress_bar(False).apply(pd.Series)
        recommendations.loc[mask_new.index] = recommendations.loc[mask_new.index].where(mask_new.apply(pd.Series))
        recommendations = recommendations.swifter.progress_bar(False) \
            .apply(lambda row: np.array(row), axis=1) \
            .swifter.progress_bar(False).apply(lambda item: item[~np.isnan(item)]) \
            .swifter.progress_bar(False).apply(pd.Series)
    if item_info is not None:
        rec_exp = recommendations.fillna(item_info.index.max() + 1) \
            .swifter.progress_bar(False).applymap(lambda item: item_info.price.get(item))
        rec_exp_filter = rec_exp.loc[(rec_exp.loc[:, :n_rec - 1] >= price_lte).sum(axis=1) < n_exp]
        mask_exp = rec_exp_filter.swifter.progress_bar(False) \
            .apply(lambda row: postfilter_for_item(row, n=n_exp, n_rec=n_rec), axis=1) \
            .swifter.progress_bar(False).apply(pd.Series)
        recommendations.loc[mask_exp.index] = recommendations.loc[mask_exp.index] \
            .where(mask_exp.apply(pd.Series))
        recommendations = recommendations.swifter.progress_bar(False) \
            .apply(lambda row: np.array(row, dtype='uint'), axis=1) \
            .swifter.progress_bar(False).apply(lambda item: item[~np.isnan(item)]) \
            .swifter.progress_bar(False).apply(pd.Series)

    recommendations = recommendations.loc[:, :n_rec - 1].swifter.progress_bar(False) \
        .apply(lambda row: np.array(row), axis=1)
    return recommendations


def postfilter_for_item(items_mask, n=2, n_rec=5):
    mask = np.ones(len(items_mask), dtype=bool)
    n = n - items_mask[:n_rec].sum()
    n_low = n
    for index, item in enumerate(np.flip(items_mask[:n_rec])):
        if not item:
            mask[n_rec - index - 1] = False
            n_low -= 1
        if not n_low:
            break
    n_high = n
    for index, item in enumerate(items_mask[n_rec:]):
        if not item:
            mask[n_rec + index] = False
        else:
            n_high -= 1
        if not n_high:
            break
    return mask
