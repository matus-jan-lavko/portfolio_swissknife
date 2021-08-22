import sys, time
import math
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay
import numpy as np
import pandas as pd
import multiprocessing as mp
import datetime as dt
import functools

class MultiProcessingFunctions:
    """ This static functions in this class enable multi-processing"""

    def __init__(self):
        pass

    @staticmethod
    def lin_parts(num_atoms, num_threads):
        """ This function partitions a list of atoms in subsets (molecules) of equal size.
		An atom is a set of indivisible set of tasks.
		"""

        # partition of atoms with a single loop
        parts = np.linspace(0, num_atoms, min(num_threads, num_atoms) + 1)
        parts = np.ceil(parts).astype(int)
        return parts

    @staticmethod
    def nested_parts(num_atoms, num_threads, upper_triangle=False):
        """ This function enables parallelization of nested loops.
		"""
        # partition of atoms with an inner loop
        parts = []
        num_threads_ = min(num_threads, num_atoms)

        for num in range(num_threads_):
            part = 1 + 4 * (parts[-1] ** 2 + parts[-1] + num_atoms * (num_atoms + 1.) / num_threads_)
            part = (-1 + part ** .5) / 2.
            parts.append(part)

        parts = np.round(parts).astype(int)

        if upper_triangle:  # the first rows are heaviest
            parts = np.cumsum(np.diff(parts)[::-1])
            parts = np.append(np.array([0]), parts)
        return parts

    @staticmethod
    def mp_pandas_obj(func, pd_obj, num_threads=24, mp_batches=1, lin_mols=True, **kargs):

        """
		:param func: (string) function to be parallelized
		:param pd_obj: (vector) Element 0, is name of argument used to pass the molecule;
						Element 1, is the list of atoms to be grouped into a molecule
		:param num_threads: (int) number of threads
		:param mp_batches: (int) number of batches
		:param lin_mols: (bool) Tells if the method should use linear or nested partitioning
		:param kargs: (var args)
		:return: (data frame) of results
		"""

        if lin_mols:
            parts = MultiProcessingFunctions.lin_parts(len(pd_obj[1]), num_threads * mp_batches)
        else:
            parts = MultiProcessingFunctions.nested_parts(len(pd_obj[1]), num_threads * mp_batches)

        jobs = []
        for i in range(1, len(parts)):
            job = {pd_obj[0]: pd_obj[1][parts[i - 1]:parts[i]], 'func': func}
            job.update(kargs)
            jobs.append(job)

        if num_threads == 1:
            out = MultiProcessingFunctions.process_jobs_(jobs)
        else:
            out = MultiProcessingFunctions.process_jobs(jobs, num_threads=num_threads)

        if isinstance(out[0], pd.DataFrame):
            df0 = pd.DataFrame()
        elif isinstance(out[0], pd.Series):
            df0 = pd.Series()
        else:
            return out

        for i in out:
            df0 = df0.append(i)

        df0 = df0.sort_index()
        return df0

    @staticmethod
    def process_jobs_(jobs):
        """ Run jobs sequentially, for debugging """
        out = []
        for job in jobs:
            out_ = MultiProcessingFunctions.expand_call(job)
            out.append(out_)
        return out

    @staticmethod
    def expand_call(kargs):
        """ Expand the arguments of a callback function, kargs['func'] """
        func = kargs['func']
        del kargs['func']
        out = func(**kargs)
        return out

    @staticmethod
    def report_progress(job_num, num_jobs, time0, task):
        # Report progress as asynch jobs are completed

        msg = [float(job_num) / num_jobs, (time.time() - time0) / 60.]
        msg.append(msg[1] * (1 / msg[0] - 1))
        time_stamp = str(dt.datetime.fromtimestamp(time.time()))

        msg = time_stamp + ' ' + str(round(msg[0] * 100, 2)) + '% ' + task + ' done after ' + \
              str(round(msg[1], 2)) + ' minutes. Remaining ' + str(round(msg[2], 2)) + ' minutes.'

        if job_num < num_jobs:
            sys.stderr.write(msg + '\r')
        else:
            sys.stderr.write(msg + '\n')

        return

    @staticmethod
    def process_jobs(jobs, task=None, num_threads=24):
        """ Run in parallel. jobs must contain a 'func' callback, for expand_call"""

        if task is None:
            task = jobs[0]['func'].__name__

        pool = mp.Pool(processes=num_threads)
        # outputs, out, time0 = pool.imap_unordered(MultiProcessingFunctions.expand_call,jobs),[],time.time()
        outputs = pool.imap_unordered(MultiProcessingFunctions.expand_call, jobs)
        out = []
        time0 = time.time()

        # Process asyn output, report progress
        for i, out_ in enumerate(outputs, 1):
            out.append(out_)
            MultiProcessingFunctions.report_progress(i, len(jobs), time0, task)

        pool.close()
        pool.join()  # this is needed to prevent memory leaks
        return out


def get_dollar_bars(ohlcv_dict, threshold):
    dollar_bars = []
    vol_temp = 0
    high_temp, low_temp = 0, math.inf

    for i in range(len(ohlcv_dict)):
        c1, h1, l1, o1, t1, v1 = [ohlcv_dict[i][k] for k in ['close', 'high', 'low',
                                                             'open', 'time', 'volume']]
        mid_price = (o1 + c1) / 2
        dollar_volume = v1 * mid_price
        high_temp, low_temp = max(high_temp, h1), min(low_temp, l1)

        if dollar_volume + vol_temp >= threshold:
            bar_adj = t1 + 1 * BDay()
            dollar_bars += [{'time': bar_adj,
                             'open': o1,
                             'high': high_temp,
                             'low': low_temp,
                             'close': c1}]
            vol_temp = 0
            high_temp, low_temp = 0, math.inf
        else:
            vol_temp += dollar_volume
    return dollar_bars


def get_daily_vol(close, lookback=100):
    """
    :param close: (data frame) Closing prices
    :param lookback: (int) lookback period to compute volatility
    :return: (series) of daily volatility value
    """

    df0 = close.index.searchsorted(close.index - BDay())
    df0 = df0[df0 > 0]
    df0 = (pd.Series(close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0]:]))

    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1  # daily returns
    df0 = df0.ewm(span=lookback).std()
    return df0


def get_t_events(raw_price, threshold):
    """
    :param raw_price: (series) of close prices.
    :param threshold: (float) when the abs(change) is larger than the threshold, the
    function captures it as an event.
    :return: (datetime index vector) vector of datetimes when the events occurred. This is used later to sample.
    """
    print('Applying Symmetric CUSUM filter.')

    t_events = []
    s_pos = 0
    s_neg = 0

    # log returns
    diff = np.log(raw_price).diff().dropna()

    # Get event time stamps for the entire series
    for i in diff.index[1:]:
        pos = float(s_pos + diff.loc[i])
        neg = float(s_neg + diff.loc[i])
        s_pos = max(0.0, pos)
        s_neg = min(0.0, neg)

        if s_neg < -threshold:
            s_neg = 0
            t_events.append(i)

        elif s_pos > threshold:
            s_pos = 0
            t_events.append(i)

    event_timestamps = pd.DatetimeIndex(t_events)
    return event_timestamps


def add_vertical_barrier(t_events, close, num_days=1):
    """
    :param t_events: (series) series of events (symmetric CUSUM filter)
    :param close: (series) close prices
    :param num_days: (int) maximum number of days a trade can be active
    :return: (series) timestamps of vertical barriers
    """
    t1 = close.index.searchsorted(t_events + num_days*BDay())
    t1 = t1[t1 < close.shape[0]]
    t1 = pd.Series(close.index[t1], index=t_events[:t1.shape[0]])  # NaNs at end
    return t1


def apply_pt_sl_on_t1(close, events, pt_sl, molecule):
    """
    :param close: (series) close prices
    :param events: (series) of indices that signify "events"
    :param pt_sl: (array) element 0, indicates the profit taking level;
                          element 1 is stop loss level
    :param molecule: (an array) a set of datetime index values for processing
    :return: (dataframe) timestamps at which each barrier was touched
    """
    # apply stop loss/profit taking, if it takes place before t1 (end of event)
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)
    if pt_sl[0] > 0:
        pt = pt_sl[0] * events_['trgt']
    else:
        pt = pd.Series(index=events.index)  # NaNs

    if pt_sl[1] > 0:
        sl = -pt_sl[1] * events_['trgt']
    else:
        sl = pd.Series(index=events.index)  # NaNs

    for loc, t1 in events_['t1'].fillna(close.index[-1]).iteritems():
        df0 = close[loc:t1]  # path prices
        df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side']  # path returns
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()  # earliest stop loss
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()  # earliest profit taking

    return out


def get_events(close, t_events, pt_sl, target, min_ret, num_threads,
               vertical_barrier_times=False, side=None):
    """
    :param close: (series) Close prices
    :param t_events: (series) of t_events.
                     These are timestamps that will seed every triple barrier.
    :param pt_sl: (2 element array) element 0, indicates the profit taking level;
                  element 1 is stop loss level.
                  A non-negative float that sets the width of the two barriers.
                  A 0 value means that the respective horizontal barrier will be disabled.
    :param target: (series) of values that are used (in conjunction with pt_sl)
                   to determine the width of the barrier.
    :param min_ret: (float) The minimum target return required for running a triple barrier search.
    :param num_threads: (int) The number of threads concurrently used by the function.
    :param vertical_barrier_times: (series) A pandas series with the timestamps of the vertical barriers.
    :param side: (series) Side of the bet (long/short) as decided by the primary model
    :return: (data frame) of events
            -events.index is event's starttime
            -events['t1'] is event's endtime
            -events['trgt'] is event's target
            -events['side'] (optional) implies the algo's position side
    """

    # 1) Get target
    target = target.loc[target.index.intersection(t_events)]
    target = target[target > min_ret]  # min_ret

    # 2) Get vertical barrier (max holding period)
    if vertical_barrier_times is False:
        vertical_barrier_times = pd.Series(pd.NaT, index=t_events)

    # 3) Form events object, apply stop loss on vertical barrier
    if side is None:
        side_ = pd.Series(1., index=target.index)
        pt_sl_ = [pt_sl[0], pt_sl[0]]
    else:
        side_ = side.loc[target.index]
        pt_sl_ = pt_sl[:2]

    events = pd.concat({'t1': vertical_barrier_times, 'trgt': target, 'side': side_},
                       axis=1)
    events = events.dropna(subset=['trgt'])

    # Apply Triple Barrier
    df0 = MultiProcessingFunctions.mp_pandas_obj(func=apply_pt_sl_on_t1,
                                                 pd_obj=('molecule', events.index),
                                                 num_threads=num_threads,
                                                 close=close,
                                                 events=events,
                                                 pt_sl=pt_sl_)

    events['t1'] = df0.dropna(how='all').min(axis=1)  # pd.min ignores nan

    if side is None:
        events = events.drop('side', axis=1)

    return events

def barrier_touched(out_df):
    """
    :param out_df: (DataFrame) containing the returns and target
    :return: (DataFrame) containing returns, target, and labels
    """
    store = []
    for i in np.arange(len(out_df)):
        date_time = out_df.index[i]
        ret = out_df.loc[date_time, 'ret']
        target = out_df.loc[date_time, 'trgt']

        if ret > 0.0 and ret > target:
            # Top barrier reached
            store.append(1)
        elif ret < 0.0 and ret < -target:
            # Bottom barrier reached
            store.append(-1)
        else:
            # Vertical barrier reached
            store.append(0)

    out_df['bin'] = store

    return out_df


def get_bins(triple_barrier_events, close):
    """
    :param triple_barrier_events: (data frame)
                -events.index is event's starttime
                -events['t1'] is event's endtime
                -events['trgt'] is event's target
                -events['side'] (optional) implies the algo's position side
                Case 1: ('side' not in events): bin in (-1,1) <-label by price action
                Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
    :param close: (series) close prices
    :return: (data frame) of meta-labeled events
    """

    # 1) Align prices with their respective events
    events_ = triple_barrier_events.dropna(subset=['t1'])
    prices = events_.index.union(events_['t1'].values)
    prices = prices.drop_duplicates()
    prices = close.reindex(prices, method='bfill')

    # 2) Create out DataFrame
    out_df = pd.DataFrame(index=events_.index)
    # Need to take the log returns, else your results will be skewed for short positions
    out_df['ret'] = np.log(prices.loc[events_['t1'].values].values) - np.log(prices.loc[events_.index])
    out_df['trgt'] = events_['trgt']

    # Meta labeling: Events that were correct will have pos returns
    if 'side' in events_:
        out_df['ret'] = out_df['ret'] * events_['side']  # meta-labeling

    # Added code: label 0 when vertical barrier reached
    out_df = barrier_touched(out_df)

    # Meta labeling: label incorrect events with a 0
    if 'side' in events_:
        out_df.loc[out_df['ret'] <= 0, 'bin'] = 0

    # Transform the log returns back to normal returns.
    out_df['ret'] = np.exp(out_df['ret']) - 1

    # Add the side to the output. This is useful for when a meta label model must be fit
    tb_cols = triple_barrier_events.columns
    if 'side' in tb_cols:
        out_df['side'] = triple_barrier_events['side']

    out_df

    return out_df

#todo rewrite into a class decorator

# class DataHandler:
#     def __init__(self, obj):
#         self.obj = obj
#     def __call__(self, *args, **kwargs):
#         if self.obj.prices.shape[0] != self.obj.portfolio.prices.shape[1]:
#             print('ss')

def DataHandler(method):
    def inner(obj, *args, **kwargs):
        if obj.prices.shape[0] != obj.portfolio.prices.shape[1]:
            #aligned indices
            idx_to_align = obj.prices.index
            aligned_idx = [obj.portfolio.prices.index.get_loc(idx_to_align[i], method = 'nearest')
                           for i in range(len(idx_to_align))]
            #rewrite attribute
            obj.portfolio.prices = obj.portfolio.prices.iloc[aligned_idx]
            #compute returns
            obj.returns = obj.prices.pct_change().dropna().to_numpy()
            obj.portfolio.returns = obj.portfolio.prices.pct_change().dropna().to_numpy()
        #if there are backtests in the object they need to be adjusted too
        if hasattr(obj.portfolio, 'backtest'):
            idx_to_align = np.subtract(aligned_idx,1)
            for key in obj.portfolio.backtest.keys():
                obj.portfolio.backtest[key]['returns'] = obj.portfolio.backtest[key]['returns'][idx_to_align]

        return method(obj)
    return inner





