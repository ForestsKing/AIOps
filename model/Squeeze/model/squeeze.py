from functools import lru_cache

import numpy as np
from scipy.spatial.distance import cityblock

from model.Squeeze.utils.attribute_combination import AttributeCombination as AC


class Squeeze:
    def __init__(self, data, param):
        self.param = param
        self._root_cause = []

        # 过滤掉真实值与预测值全为0的数据
        self.data = data

        # 获取 attributes, set(): 列名可能重复; sorted():因为AC里sorted()了,这里如果不,顺序会不一致
        self.attribute_names = list(sorted(set(self.data.columns) - {'real', 'predict'}))

        # dataframe -> {column: value} -> AC
        self.ac_array = np.array(
            [AC(**record) for record in self.data[self.attribute_names].to_dict(orient='records')])

    @property
    @lru_cache()
    def root_cause(self):
        return self._root_cause

    @lru_cache()
    def get_cuboid_ac_array(self, cuboid):
        return np.array(list(map(lambda x: x.mask(cuboid), self.ac_array)))

    @lru_cache()
    def get_indexed_data(self, cuboid):
        return self.data.set_index(list(cuboid))

    def run(self):
        cuboid = tuple(['device', 'node'])
        cluster = [i for i in range(len(self.data))]
        # 将 cuboid 作为 dataframe 的索引
        data_cuboid_indexed = self.get_indexed_data(cuboid)

        # 对于簇中元素屏蔽掉 cuboid 之外的其他 attribute
        abnormal_cuboid_ac_arr = self.get_cuboid_ac_array(cuboid)[cluster]

        # 对于簇中数据获取 cuboid 里的 attribute values 的种类和个数
        elements, num_elements = np.unique(abnormal_cuboid_ac_arr, return_counts=True)

        # 对于所有数据获取cuboid里的attribute values的种类和个数
        num_ele_descents = np.array(list(
            np.count_nonzero(
                _.index_dataframe(data_cuboid_indexed),
            ) for _ in elements
        ))

        # 获得descent score来表示后代叶子属性组合在异常簇中的比例
        descent_score = num_elements / num_ele_descents

        # 按照descent score降序排列
        idx = np.argsort(descent_score)[::-1]
        elements = elements[idx]

        # 获得每个分区的GPS分数
        rc_scores = np.asarray(
            list(map(lambda x: self._root_cause_score(cluster=cluster, cuboid=cuboid, element=x), elements)))

        # 返回得分最高的分区和得分
        idx = np.argsort(rc_scores)[::-1]
        elements = elements[idx]

        rc = elements[:5]
        self._root_cause.append(rc.tolist())

    def get_derived_dataframe(self, ac_set, cuboid, subset_indices):
        # subset: 正常元素 + 该簇中的异常元素
        # idx: 满足当前elements切片的元素
        subset = np.zeros(len(self.data), dtype=np.bool)
        subset[subset_indices] = True
        idx = AC.batch_index_dataframe(ac_set, self.get_indexed_data(cuboid))

        data = self.data[idx & subset]
        complement_data = self.data[(~idx) & subset]
        return data, complement_data

    def _root_cause_score(self, cluster, cuboid, element):
        # data_p: [正常元素 + 该簇中的异常元素] & 满足当前 elements 的切片 S1
        # data_n: [正常元素 + 该簇中的异常元素] & 不满足当前 elements 的切片 S2
        data_p, data_n = self.get_derived_dataframe([element], cuboid=cuboid,
                                                    subset_indices=np.sort(np.concatenate([cluster])))

        _v1, _v2 = data_p.real.values, data_n.real.values
        _f1, _f2 = data_p.predict.values, data_n.predict.values

        _pv, _pf = np.sum(data_p.real.values), np.sum(data_p.predict.values)
        _a1, _a2 = data_p.predict.values * (_pv / _pf), data_n.predict.values

        # L1范数即绝对值之和
        _ps = 1 - ((np.mean(cityblock(_v1, _a1)) + np.mean(cityblock(_v2, _f2)))
                   / (np.mean(cityblock(_v1, _f1)) + np.mean(cityblock(_v2, _f2))))

        return _ps
