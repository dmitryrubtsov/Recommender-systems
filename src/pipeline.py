import pandas as pd
import numpy as np
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender
from implicit.nearest_neighbours import bm25_weight
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelector(TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame), 'This is not a dataframe'
        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError(f'DataFrame does not contain the following columns: {cols_error}')


class PrefilterItems(TransformerMixin, BaseEstimator):
    def __init__(
            self, take_n_popular=5000, item_features=None,
            filter_item_id=-99, n_last_week=52,
    ):

        self.take_n_popular = take_n_popular
        self.item_features = item_features
        self.filter_item_id = filter_item_id
        self.n_last_week = n_last_week

    def _reset(self):
        if hasattr(self, 'is_fit_'):
            del self.is_fit_

    def fit(self, X, items=None):
        self._reset
        return self

    def transform(self, X, items=None):
        if not hasattr(self, 'is_fit_'):
            assert isinstance(X, pd.DataFrame), 'This is not a dataframe'
            # Уберем самые популярные товары (их и так купят)
            popularity = X.groupby('item_id')['user_id'].nunique().reset_index() / X['user_id'].nunique()
            popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)

            top_popular = popularity[popularity['share_unique_users'] > 0.2].item_id.tolist()
            X = X[~X['item_id'].isin(top_popular)]
            # Уберем самые НЕ популярные товары (их и так НЕ купят)
            top_notpopular = popularity[popularity['share_unique_users'] < 0.02].item_id.tolist()
            X = X[~X['item_id'].isin(top_notpopular)]
            # Уберем товары, которые не продавались за последние 12 месяцев
            last_time = X.week_no.max() - self.n_last_week
            X = X.loc[X.item_id.isin(X.loc[X.week_no > last_time, 'item_id'])]
            # Уберем не интересные для рекоммендаций категории (department)
            if self.item_features is not None:
                department_size = self.item_features.groupby('DEPARTMENT')['PRODUCT_ID'] \
                    .nunique().sort_values(ascending=False).rename('n_items')
                rare_departments = department_size[department_size > 150].index.tolist()
                items_in_rare_departments = self.item_features.loc[self.item_features['DEPARTMENT']
                                                                   .isin(rare_departments)]['PRODUCT_ID'].unique().tolist()
                X = X.loc[X.item_id.isin(items_in_rare_departments)]
            # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
            X = X[X['price'] > 2]
            # Уберем слишком дорогие товары
            X = X[X['price'] < 50]
            # Возьмем топ по популярности
            popularity = X.groupby('item_id')['quantity'].sum().reset_index()
            popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

            top = popularity.sort_values('n_sold', ascending=False)[: self.take_n_popular].item_id.tolist()
            # Заведем фиктивный item_id (если юзер покупал товары из топ-n, то он "купил" такой товар)
            X.loc[~X['item_id'].isin(top), 'item_id'] = self.filter_item_id
            self.is_fit_ = True

        return X


class RandomEstimator(TransformerMixin, BaseEstimator):
    def __init__(
        self, n_rec=5, n_rec_pre=100, n_new=2, n_exp=1, price_lte=7,
        filter_item_id=-99, filter=True, filter_post=True,
        postfilter_func=None, random_state=42
    ):

        self.n_rec = n_rec
        self.n_rec_pre = n_rec_pre
        self.n_new = n_new
        self.n_exp = n_exp
        self.price_lte = price_lte
        self.filter_item_id = filter_item_id
        self.filter = filter
        self.filter_post = filter_post
        self.postfilter_func = postfilter_func
        self.random_state = random_state

    def _reset(self):
        if hasattr(self, 'items'):
            del self.items
        if hasattr(self, 'item_info'):
            del self.item_info
        if hasattr(self, 'user_history'):
            del self.user_history

    def fit(self, X, items=None):
        self._reset()
        self.items = X.item_id.unique()
        self.item_info = X.groupby('item_id').agg({'price': 'max', 'SUB_COMMODITY_DESC': 'first'})
        self.user_history = pd.DataFrame(X.groupby('user_id').item_id.unique().rename('history'))

        if items is not None:
            self.items = items
        else:
            self.items = X.item_id.unique()
        if self.filter:
            self.items = self.items[np.where(self.items != self.filter_item_id)]
        return self

    def transform(self, X):
        X = X['user_id'].drop_duplicates()
        return X

    def predict(self, X):
        X = self.transform(X)

        if self.filter_post:
            n_rec = self.n_rec_pre
        else:
            n_rec = self.n_rec

        rec = X.swifter.progress_bar(False).apply(lambda x: self._random_recommendation(n_rec))
        rec.index = X.values

        if self.postfilter_func is not None and self.filter_post:
            rec = self.postfilter_func(
                rec,
                item_info=self.item_info,
                user_history=self.user_history,
                n_rec=self.n_rec,
                n_new=self.n_new,
                n_exp=self.n_exp,
                price_lte=self.price_lte,
            )

        assert (rec.swifter.progress_bar(False).apply(len) == self.n_rec).all(), f'The number of recommendations is not equal {self.n_rec}.'

        return rec

    def _random_recommendation(self, n_rec):
        np.random.seed(self.random_state)
        recs = np.random.choice(self.items, size=n_rec, replace=False, )
        return recs


class AlsEstimator(TransformerMixin, BaseEstimator):
    def __init__(
            self, recommendations='als', n_rec=5, n_rec_pre=100, n_new=2,
            n_exp=1, price_lte=7, filter_item_id=-99, filter=True, filter_post=True,
            postfilter_func=None, factors=50, regularization=0.01,
            iterations=10, matrix_values='quantity', matrix_aggfunc='count',
            weighting=True, use_native=True, use_gpu=False
    ):

        self.n_rec = n_rec
        self.n_rec_pre = n_rec_pre
        self.n_new = n_new
        self.n_exp = n_exp
        self.price_lte = price_lte
        self.filter_item_id = filter_item_id
        self.filter = filter
        self.filter_post = filter_post
        self.postfilter_func = postfilter_func

        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.matrix_values = matrix_values
        self.matrix_aggfunc = matrix_aggfunc
        self.recommendations = recommendations
        self.weighting = True

        self.use_native = use_native
        self.use_gpu = use_gpu

    def _reset(self):
        if hasattr(self, 'item_info'):
            del self.item_info
        if hasattr(self, 'user_history'):
            del self.user_history
        if hasattr(self, 'top_purchases'):
            del self.top_purchases
        if hasattr(self, 'overall_top_purchases'):
            del self.overall_top_purchases
        if hasattr(self, 'user_item_matrix'):
            del self.user_item_matrix
        if hasattr(self, 'id_to_itemid'):
            del self.id_to_itemid
        if hasattr(self, 'id_to_userid'):
            del self.id_to_userid
        if hasattr(self, 'itemid_to_id'):
            del self.itemid_to_id
        if hasattr(self, 'userid_to_id'):
            del self.userid_to_id
        if hasattr(self, '_fit'):
            del self._fit

    @staticmethod
    def _prepare_matrix(data: pd.DataFrame, values: str, aggfunc: str):
        """Готовит user-item матрицу"""
        user_item_matrix = pd.pivot_table(data,
                                          index='user_id', columns='item_id',
                                          values=values,
                                          aggfunc=aggfunc,
                                          fill_value=0
                                          )

        user_item_matrix = user_item_matrix.astype(float)

        return user_item_matrix

    @staticmethod
    def _prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    def fit(self, X, y=None):
        self._reset()
        self.item_info = X.groupby('item_id').agg({'price': 'max', 'SUB_COMMODITY_DESC': 'first'})
        self.user_history = pd.DataFrame(X.groupby('user_id').item_id.unique().rename('history'))

        self.top_purchases = X.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != self.filter_item_id]

        # Топ покупок по всему датасету
        self.overall_top_purchases = X.groupby('item_id')['quantity'].count().reset_index()
        self.overall_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.overall_top_purchases = self.overall_top_purchases[self.overall_top_purchases['item_id'] != self.filter_item_id]
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()

        self.user_item_matrix = self._prepare_matrix(X, self.matrix_values, self.matrix_aggfunc)

        self.id_to_itemid, self.id_to_userid, \
            self.itemid_to_id, self.userid_to_id = self._prepare_dicts(self.user_item_matrix)

        if self.weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            dtype=np.float32,
            use_native=self.use_native,
            use_gpu=self.use_gpu,
        )

        self.model.fit(csr_matrix(self.user_item_matrix).T.tocsr())

        self.model_own_recommender = ItemItemRecommender(K=1)
        self.model_own_recommender.fit(csr_matrix(self.user_item_matrix).T.tocsr())

        self._fit = True

    def transform(self, X):
        if self._fit:
            X = X['user_id'].drop_duplicates()
            X.index = X.values
        return X

    def _update_dict(self, user_id):
        """Если появился новыю user / item, то нужно обновить словари"""

        if user_id not in self.userid_to_id.keys():

            max_id = max(list(self.userid_to_id.values()))
            max_id += 1

            self.userid_to_id.update({user_id: max_id})
            self.id_to_userid.update({max_id: user_id})

    def _get_similar_item(self, item_id):
        """Находит товар, похожий на item_id"""
        recs = self.model.similar_items(self.itemid_to_id[item_id], N=2)  # Товар похож на себя -> рекомендуем 2 товара
        top_rec = recs[1][0]  # И берем второй (не товар из аргумента метода)
        return self.id_to_itemid[top_rec]

    def _extend_with_top_popular(self, recommendations):
        """Если кол-во рекоммендаций < N, то дополняем их топ-популярными"""

        if self.filter_post:
            n_rec = self.n_rec_pre
        else:
            n_rec = self.n_rec

        if len(recommendations) < n_rec:
            recommendations.extend(self.overall_top_purchases[:n_rec])
            recommendations = recommendations[:n_rec]

        return recommendations

    def _get_recommendations(self, user, model, n_rec):
        """Рекомендации через стардартные библиотеки implicit"""

        self._update_dict(user_id=user)
        try:
            res = [self.id_to_itemid[rec[0]] for rec in model.recommend(
                userid=self.userid_to_id[user],
                user_items=csr_matrix(self.user_item_matrix).tocsr(),
                N=n_rec,
                filter_already_liked_items=False,
                filter_items=[self.itemid_to_id[self.filter_item_id]],
                recalculate_user=True
            )]
        except:
            res = list()
        finally:
            res = self._extend_with_top_popular(res)

            assert len(res) == n_rec, 'Количество рекомендаций != {}'.format(n_rec)
            return res

    def get_als_recommendations(self, user):
        """Рекомендации через стардартные библиотеки implicit"""
        if self.filter_post:
            n_rec = self.n_rec_pre
        else:
            n_rec = self.n_rec

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.model, n_rec)

    def get_own_recommendations(self, user):
        """Рекомендуем товары среди тех, которые юзер уже купил"""

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.model_own_recommender)

    def get_similar_items_recommendations(self, user):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
        if self.filter_post:
            n_rec = self.n_rec_pre
        else:
            n_rec = self.n_rec

        top_users_purchases = self.top_purchases[self.top_purchases['user_id'] == user].head(n_rec)

        res = top_users_purchases['item_id'].apply(lambda x: self._get_similar_item(x)).tolist()
        res = self._extend_with_top_popular(res)

        assert len(res) == n_rec, 'Количество рекомендаций != {}'.format(n_rec)
        return res

    def get_similar_users_recommendations(self, user):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        if self.filter_post:
            n_rec = self.n_rec_pre
        else:
            n_rec = self.n_rec
        res = []

        # Находим топ-N похожих пользователей
        similar_users = self.model.similar_users(self.userid_to_id[user], N=n_rec + 1)
        similar_users = [rec[0] for rec in similar_users]
        similar_users = similar_users[1:]   # удалим юзера из запроса

        for user in similar_users:
            user_rec = self._get_recommendations(user, model=self.model_own_recommender, n_rec=1)
            res.extend(user_rec)

        res = self._extend_with_top_popular(res)

        assert len(res) == n_rec, 'Количество рекомендаций != {}'.format(n_rec)
        return res

    def predict(self, X):
        X = self.transform(X)
        recommender = getattr(self, f'get_{self.recommendations}_recommendations')

        rec = X.swifter.progress_bar(False).apply(lambda item: recommender(user=item))
        if self.postfilter_func is not None and self.filter_post:
            rec = self.postfilter_func(
                rec,
                item_info=self.item_info,
                user_history=self.user_history,
                n_rec=self.n_rec,
                n_new=self.n_new,
                n_exp=self.n_exp,
                price_lte=self.price_lte,
            )

        assert (rec.swifter.progress_bar(False).apply(len) == self.n_rec).all(), f'The number of recommendations is not equal {self.n_rec}.'

        return rec
