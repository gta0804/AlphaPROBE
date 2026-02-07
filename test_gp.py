import os
import argparse
import qlib
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
instruments = 'csi500'

import json
from collections import Counter
from alphagen.data.expression import *
from alphagen.models.alpha_pool import AlphaPool
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from alphagen_generic.features import *
from gan.utils.data import get_data_by_year


def pred_pool(capacity,data,cache):
    from alphagen_qlib.calculator import QLibStockDataCalculator
    pool = AlphaPool(capacity=capacity,
                    stock_data=data,
                    target=target,
                    ic_lower_bound=None)
    exprs = []
    for key in dict(Counter(cache).most_common(capacity)):
        exprs.append(eval(key))
    pool.force_load_exprs(exprs)
    pool._optimize(alpha=5e-3, lr=5e-1, n_iter=2000)

    exprs = pool.exprs[:pool.size]
    weights = pool.weights[:pool.size]
    calculator_test = QLibStockDataCalculator(data, target)
    ensemble_value = calculator_test.make_ensemble_alpha(exprs, weights)
    return ensemble_value


def get_data_by_year(
    train_start = 2010,train_end=2019,valid_year=2020,test_year =2021,
    instruments=None, target=None,freq=None,qlib_path=None,
                    ):
    
    from gan.utils import load_pickle,save_pickle
    # from gan.utils.qlib import get_data_my
    get_data_my = StockData

    train_dates=(f"2010-01-01", f"2020-12-31")
    val_dates=(f"2021-01-01", f"2022-06-30")
    test_dates=(f"2022-07-01", f"2025-06-30")

    train_start,train_end = train_dates
    valid_start,valid_end = val_dates
    valid_head_start = f"{valid_year-2}-01-01"
    test_start,test_end = test_dates
    test_head_start = f"{test_year-2}-07-01"

    name = instruments + '_pkl_' + str(target).replace('/','_').replace(' ','') + '_' + freq
    name = f"{name}_{train_start}_{train_end}_{valid_start}_{valid_end}_{test_start}_{test_end}"
    try:

        data = load_pickle(f'pkl/{name}/data.pkl')
        data_valid = load_pickle(f'pkl/{name}/data_valid.pkl')
        data_valid_withhead = load_pickle(f'pkl/{name}/data_valid_withhead.pkl')
        data_test = load_pickle(f'pkl/{name}/data_test.pkl')
        data_test_withhead = load_pickle(f'pkl/{name}/data_test_withhead.pkl')

    except:
        print('Data not exist, load from qlib')
        print(f"qlib_path: {qlib_path}")
        data = get_data_my(instruments, train_start, train_end,raw =False,qlib_path = qlib_path,freq=freq)
        data_valid = get_data_my(instruments, valid_start, valid_end,raw = False,qlib_path = qlib_path,freq=freq)
        data_valid_withhead = get_data_my(instruments,valid_head_start, valid_end,raw = False,qlib_path = qlib_path,freq=freq)
        data_test = get_data_my(instruments, test_start, test_end,raw = False,qlib_path = qlib_path,freq=freq)
        data_test_withhead = get_data_my(instruments, test_head_start, test_end,raw = False,qlib_path = qlib_path,freq=freq)

        os.makedirs(f"pkl/{name}",exist_ok=True)
        save_pickle(data,f'pkl/{name}/data.pkl')
        save_pickle(data_valid,f'pkl/{name}/data_valid.pkl')
        save_pickle(data_valid_withhead,f'pkl/{name}/data_valid_withhead.pkl')
        save_pickle(data_test,f'pkl/{name}/data_test.pkl')
        save_pickle(data_test_withhead,f'pkl/{name}/data_test_withhead.pkl')
    
    try:
        data_all = load_pickle(f'pkl/{name}/data_all.pkl')
    except:
        data_all = get_data_my(instruments, train_start, test_end,raw = False,qlib_path = qlib_path,freq=freq)
        save_pickle(data_all,f'pkl/{name}/data_all.pkl')
    return data_all,data,data_valid,data_valid_withhead,data_test,data_test_withhead,name


for seed in range(1):
    for train_end in range(2020,2021):
        for num in [20]:
            save_dir = f'out_gp/{instruments}_{train_end}_day_{seed}' 
            print(save_dir)
            
            returned = get_data_by_year(
                train_start = 2010,train_end=train_end,valid_year=train_end+1,test_year =train_end+2,
                instruments=instruments, target=target,freq='day',
                qlib_path = 'PATH/TO/.qlib/qlib_data/cn_data'
            )
            data_all,data,data_valid,data_valid_withhead,data_test,data_test_withhead,name = returned
            cache = json.load(open(f'{save_dir}/2.json'))['cache']

            features = ['open_', 'close', 'high', 'low', 'volume', 'vwap']
            constants = [f'Constant({v})' for v in [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]]
            terminals = features + constants

            pred = pred_pool(num,data_all,cache)
            pred = pred[-data_test.n_days:]
            torch.save(pred.detach().cpu(),f"{save_dir}/pred_{num}.pt")


import pandas as pd
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr, batch_ret, batch_sharpe_ratio, batch_max_drawdown
import torch
import os
import numpy as np

def chunk_batch_spearmanr(x, y, chunk_size=100):
    n_days = len(x)
    spearmanr_list= []
    for i in range(0, n_days, chunk_size):
        spearmanr_list.append(batch_spearmanr(x[i:i+chunk_size], y[i:i+chunk_size]))
    spearmanr_list = torch.cat(spearmanr_list, dim=0)
    return spearmanr_list

def get_tensor_metrics(x, y, y_ret, risk_free_rate=0.0):
    # Ensure tensors are 2D (days, stocks)
    if x.dim() > 2: x = x.squeeze(-1)
    if y.dim() > 2: y = y.squeeze(-1)

    ic_s = batch_pearsonr(x, y)
    ric_s = chunk_batch_spearmanr(x, y, chunk_size=400)
    ret_s = batch_ret(x, y_ret) 

    ic_s = torch.nan_to_num(ic_s, nan=0.)
    ric_s = torch.nan_to_num(ric_s, nan=0.)
    ret_s = torch.nan_to_num(ret_s, nan=0.)
    ic_s_mean = ic_s.mean().item()
    ic_s_std = ic_s.std().item() if ic_s.std().item() > 1e-6 else 1.0
    ric_s_mean = ric_s.mean().item()
    ric_s_std = ric_s.std().item() if ric_s.std().item() > 1e-6 else 1.0
    ret_s_mean = ret_s.mean().item()
    ret_s_std = ret_s.std().item() if ret_s.std().item() > 1e-6 else 1.0
    
    # Calculate Sharpe Ratio and Maximum Drawdown for ret series
    ret_sharpe = batch_sharpe_ratio(ret_s, risk_free_rate).item()
    ret_mdd = batch_max_drawdown(ret_s).item()
    result = dict(
        ic=ic_s_mean,
        ic_std=ic_s_std,
        icir=ic_s_mean / ic_s_std,
        ric=ric_s_mean,
        ric_std=ric_s_std,
        ricir=ric_s_mean / ric_s_std,
        ret=ret_s_mean * len(ret_s) / 3,
        ret_std=ret_s_std,
        retir=ret_s_mean / ret_s_std,
        ret_sharpe=ret_sharpe,
        ret_mdd=ret_mdd,
    )
    return result, ret_s


result = []
instruments = 'csi500'
for num in [20]:
    for seed in range(1):
    
        cur_seed_ic = []
        cur_seed_ric = []
        for train_end in range(2020,2021):
                #'/path/to/save/results'
                save_dir = f'out_gp/{instruments}_{train_end}_day_{seed}' 

                returned = get_data_by_year(
                    train_start = 2010,train_end=train_end,valid_year=train_end+1,test_year =train_end+2,
                    instruments=instruments, target=target,freq='day',
                    qlib_path = 'PATH/TO/.qlib/qlib_data/cn_data'
                )
                data_all,data,data_valid,data_valid_withhead,data_test,data_test_withhead,name = returned
                pred = torch.load(f"{save_dir}/pred_{num}.pt")
                
                tgt = target.evaluate(data_all)[-data_test.n_days:,:].to("cpu")
                target_ret = Ref(Close, -1) / Close - 1
                tgt_ret = target_ret.evaluate(data_all)[-data_test.n_days:,:].to("cpu")
                res, ret_s = get_tensor_metrics(torch.tensor(pred), torch.tensor(tgt), torch.tensor(tgt_ret))
                print(pd.DataFrame(res,index=["Test"]))
                np.save(f"{save_dir}/ret_s.npy", ret_s)