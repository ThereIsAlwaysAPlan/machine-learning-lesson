# coding = 'utf-8'
import numpy as np
import pandas as pd
import time
import tm
from functools import wraps


def time_decrator(func):
    """
    函数执行计时装饰器
    """
    @wraps(func)
    def inner(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print("执行函数[{}]共耗时：{}秒".format(func.__name__, end-start))
        return result
    return inner


@time_decrator
def target_mean_v1(data, y_name, x_name):
    result = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        groupby_result = data[data.index != i].groupby(
            [x_name], as_index=False).agg(['mean', 'count'])
        result[i] = groupby_result.loc[groupby_result.index ==
                                       data.loc[i, x_name], (y_name, 'mean')]
    return result


@time_decrator
def target_mean_v2(data, y_name, x_name):
    result = np.zeros(data.shape[0])
    value_dict = dict()
    count_dict = dict()
    for i in range(data.shape[0]):
        if data.loc[i, x_name] not in value_dict.keys():
            value_dict[data.loc[i, x_name]] = data.loc[i, y_name]
            count_dict[data.loc[i, x_name]] = 1
        else:
            value_dict[data.loc[i, x_name]] += data.loc[i, y_name]
            count_dict[data.loc[i, x_name]] += 1
    for i in range(data.shape[0]):
        result[i] = (value_dict[data.loc[i, x_name]] - data.loc[i,
                     y_name]) / (count_dict[data.loc[i, x_name]] - 1)
    return result


def init_para():
    """
    初始化数据集
    """
    y = np.random.randint(2, size=(5000, 1))
    x = np.random.randint(10, size=(5000, 1))
    data = pd.DataFrame(np.concatenate([y, x], axis=1), columns=['y', 'x'])
    return data


def main():
    data = init_para()
    print(data)
    result_1 = target_mean_v1(data, 'y', 'x')
    result_2 = target_mean_v2(data, 'y', 'x')
    target_mean_v3 = time_decrator(tm.target_mean_v3)
    result_3 = target_mean_v3(data, 'y', 'x')
    target_mean_v4 = time_decrator(tm.target_mean_v4)
    result_4 = target_mean_v4(data, 'y', 'x')
    if np.linalg.norm(result_1[:] - result_2[:]) or \
        np.linalg.norm(result_2[:] - result_3[:]) or \
            np.linalg.norm(result_3[:] - result_4[:]):
        print('there are diff in results')
    else:
        print('there are no diff in results')


if __name__ == '__main__':
    main()
