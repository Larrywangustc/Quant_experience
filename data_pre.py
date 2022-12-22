import time
from abc import abstractmethod, ABCMeta
from multiprocessing import Pool, Manager, Value
from config import *
from enums_lib import DataType
from factor_lib import *
from keras_preprocessing.sequence import TimeseriesGenerator
import numpy as np
import pandas as pd
import datetime

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class RawData:
    __day_raw_path = '%s%s.pkl' % (raw_dir, 'day_raw')
    __day_index_path = '%s%s.pkl' % (raw_dir, 'day_index')  #
    __trade_date_path = '%s%s.pkl' % (raw_dir, 'trade_dates')
    __stdata_raw_path = '%s%s.pkl' % (raw_dir, 'st')  #
    __valuation_raw_path = '%s%s.pkl' % (raw_dir, 'valuation')  #
    __description_raw_path = '%s%s.pkl' % (raw_dir, 'description')  #

    __day_raw = None
    __day_index_raw = None
    __trade_dates = None
    __valuation_raw = None
    __st_raw = None
    __description_raw = None

    @staticmethod
    def day_data():
        if RawData.__day_raw is None:
            RawData.__day_raw = pd.read_pickle(RawData.__day_raw_path)
        return RawData.__day_raw

    @staticmethod
    def day_index_data():
        if RawData.__day_index_raw is None:
            RawData.__day_index_raw = pd.read_pickle(RawData.__day_index_path)
        return RawData.__day_index_raw

    @staticmethod
    def trade_dates():
        if RawData.__trade_dates is None:
            RawData.__trade_dates = np.array(pd.read_pickle(RawData.__trade_date_path).values.flatten())
        return RawData.__trade_dates

    @staticmethod
    def valuation():
        if RawData.__valuation_raw is None:
            RawData.__valuation_raw = pd.read_pickle(RawData.__valuation_raw_path)
        return RawData.__valuation_raw

    @staticmethod
    def st_data():
        if RawData.__st_raw is None:
            RawData.__st_raw = pd.read_pickle(RawData.__stdata_raw_path)
        return RawData.__st_raw

    @staticmethod
    def desription():
        if RawData.__desription_raw is None:
            RawData.__desription_raw = pd.read_pickle(RawData.__desription_raw_path)
        return RawData.__desription_raw


class CreateDataBase(metaclass=ABCMeta):

    def __init__(self, ):
        self.process_count = create_process_count
        self.in_queue = Manager().Queue()
        self.out_queue = Manager().Queue()
        self.columns = Value('i', 0)

        self.set_version()
        self.set_data_name()
        self.set_data_type()

    def __frame_version__(self):
        return '0.1.1'

    def set_dates(self, s, start_day, end_day):
        self.generate_type = s
        self.start_day = start_day
        self.end_day = end_day

    @abstractmethod
    def set_version(self):
        # 版本号
        self.version = ''

    @abstractmethod
    def set_data_type(self):
        # 数据类型
        self.data_type = ''

    @abstractmethod
    def set_data_name(self):
        # 数据名称
        self.data_name = ''

    @abstractmethod
    def prepare(self):
        '''
            生成个股特征前
                若需要生成市场数据
                若需要生成按日期group的数据
                若需要合并指数数据
                ...

                均可在此函数中操作

            切记：数据处理完成后要对数据的code date time（高频数据需要）重新排序，以保证后续按个股数据创造特征不会出问题

        :return:处理后的pandas
        '''
        pass

    @abstractmethod
    def create_feature(self, raws_datas):
        '''

            创造特征（个股特征）
            数据切片
            标准化等

            :return:
        '''
        pass

    def __slice_raw(self):
        '''
            根据股票code，切分，把每一份传入到queue
        :return:
        '''
        print('%s: __slice_raw start' % time.strftime('%H:%M:%S'))

        # 数据预处理，切记数据处理完成后要对数据的code date time（高频数据需要）重新排序，以保证后续按个股数据创造特征不会出问题
        all_raws = self.prepare

        # 获取数据中所有代码
        codes = np.unique(all_raws['code'])
        code_nums = len(codes)

        # 拆分每个进程传入的code集合
        slice_nums = int(code_nums / self.process_count / 5) + 1
        queue_num = 0
        for i in range(0, code_nums, slice_nums):
            code = codes[i:i + slice_nums]
            raw = all_raws[all_raws['code'].isin(code)]
            # 把拆分后的代码集合数据放入队列中
            self.in_queue.put(raw)
            queue_num += 1

        print('%s: __slice_raw end' % time.strftime('%H:%M:%S'))
        # 返回队列数据条数
        return queue_num

    def __process(self):
        '''
            子进程函数
        '''
        while not self.in_queue.empty():
            raw_datas = self.in_queue.get()
            generates_x = []
            generates_y = []
            # 按照股票分组，并循环
            code_grouped = raw_datas.groupby('code')
            for code, single_code_datas in code_grouped:
                # 根据传进去的个股数据，进行特征创造/切片/标准化等处理
                x_data, y_data = self.create_feature(single_code_datas)
                generates_x.extend(x_data)
                generates_y.extend(y_data)
            generates_x = np.asarray(generates_x)
            generates_y = np.asarray(generates_y)

            # 切片数据传到主进程
            self.out_queue.put([generates_x, generates_y])

    def create(self):
        task_count = self.__slice_raw()
        pool = Pool(self.process_count, self.__process)
        datas = np.array([])
        datas_ytrue = np.array([])
        i = 0
        while i < task_count:
            if not self.out_queue.empty():
                outs, outs_ytrue = self.out_queue.get()
                if outs.shape[0] > 0:
                    if datas.shape[0] == 0:
                        datas = outs
                    else:
                        datas = np.concatenate((datas, outs))
                if outs_ytrue.shape[0] > 0:
                    if datas_ytrue.shape[0] == 0:
                        datas_ytrue = outs_ytrue
                    else:
                        datas_ytrue = np.concatenate((datas_ytrue, outs_ytrue))

                print('running process num', task_count - i, datas.shape, datas_ytrue.shape)
                i += 1

        pool.terminate()
        self.save_data(datas, 'x')
        self.save_data(datas_ytrue, 'y')

    '''
          保存切片数据
    '''

    def save_data(self, datas, s):
        '''
        保存生成切片
        :param datas:切片数据
        '''
        print('%s: %s __save start' % (time.strftime('%H:%M:%S'), self.generate_type))
        if datas.shape[0] > 0:
            save_path = '%s%s_%s_%s.npy' % (newdata_dir, self.version, self.data_name, s + self.generate_type)
            np.save(save_path, datas)  # 用于训练的数据

        print('%s: %s __save end' % (time.strftime('%H:%M:%S'), self.generate_type))

    @staticmethod
    def global_maxminscaler(subndarray):
        # 有相关关系的数据归一化处理
        v = subndarray.max() - subndarray.min()
        if v == 0:
            return np.ones(subndarray.shape) * 0.5
        return (subndarray - subndarray.min()) / v

    @staticmethod
    def local_maxminscaler(unindarray):
        # 数据单独归一化处理
        v = unindarray.max(axis=0) - unindarray.min(axis=0)
        unindarray[:, [v == 0][0]] = 0.5
        np.divide(unindarray - unindarray.min(axis=0), v, out=unindarray, where=v != 0)
        return unindarray

    @staticmethod
    def global_standardscaler(subndarray):
        m = subndarray.mean()
        s = subndarray.std()
        if s == 0:
            return np.zeros(subndarray.shape)
        return (subndarray - m) / s

    @staticmethod
    def local_standardscaler(unindarray):
        """
        多列分开标准化
        :param subndarray:
        :return:
        """
        s = unindarray.std(axis=0)
        unindarray[:, [s < 0.00000001][0]] = 0
        np.divide(unindarray - unindarray.mean(axis=0), s, out=unindarray, where=s > 0.00000001)
        return unindarray

    @staticmethod
    def sudpended(date_start, date_end, timesteps):
        # 判断样本内股票交易日期的连续性
        trade_dates = RawData.trade_dates()
        if (trade_dates[np.where(trade_dates == date_start)[0] + timesteps + 2 - 1]) == date_end:
            return True
        else:
            return False

    @staticmethod
    def last_day_limit(pre_close, high, low):
        # 判断样本最后一天是否一字板涨停
        if high == low and high > pre_close:
            return False
        else:
            return True


class D2(CreateDataBase):

    def __init__(self):
        CreateDataBase.__init__(self)

    def set_version(self):
        self.version = version

    def set_data_type(self):
        self.data_type = DataType.Day

    def set_data_name(self):
        self.data_name = 'd2'

    def date_group_data(self, data):
        d = data.copy()
        d['M_up_preclose_percent'] = d[d['close'] > d['pre_close']].shape[0] / d.shape[0]
        d['amount_rank'] = d['amount'].rank() / d.shape[0]

        return d

    @property
    def prepare(self):
        day_datas = RawData.day_data()
        trade_dates = RawData.trade_dates()
        valuation_datas = RawData.valuation().loc[:, ['code', 'date', 'turn', 'high52', 'low52', 'free_shr',
                                                      'tot_shr', 'mv', 'dq_mv', 'pcf_ncf', 'pe_ttm', 'ps_ttm'
                                                         , 'pe', 'float_shr',  'pcf', 'ps', 'assets']
                          ]

        # 去掉新股三个月
        new_stock = RawData.day_data()[['code', 'date', 'trade_status']]
        new_stock = new_stock[new_stock['trade_status'] == 'N']
        new_stock['date'] = pd.to_datetime(new_stock['date'].astype('str'))
        new_stock['cut-60-date'] = new_stock['date'] + datetime.timedelta(days=90)
        new_stock['cut-60-date'] = new_stock['cut-60-date'].dt.strftime("%Y%m%d").astype('int')
        new_stock.drop(columns=['date', 'trade_status'], inplace=True)

        day_datas = pd.merge(day_datas, new_stock, how='left', on=['code'])
        day_datas.drop(day_datas[(day_datas['date'] <= day_datas['cut-60-date'])].index, inplace=True)
        day_datas.drop(columns=['cut-60-date', 'trade_status'], inplace=True)

        dates = pd.DataFrame({})
        dates['date'] = trade_dates
        dates = dates.dropna().reset_index(drop=True).astype(np.int64)

        # 去除涨跌幅异常 第二行创业板 第三行科创板
        day_datas = day_datas[(abs(day_datas['pct_change']) < 11) |
                              ((abs(day_datas['pct_change']) < 21) & ((day_datas['date'] >= 20200824) & (
                                      (day_datas['code'] <= 399999) & (day_datas['code'] >= 300000)))) |
                              ((abs(day_datas['pct_change']) < 21) & (
                                      (day_datas['code'] <= 688999) & (day_datas['code'] >= 688000)))]


        day_datas = pd.merge(day_datas, dates, on=['date'])
        day_datas = pd.merge(day_datas, valuation_datas, on=['date', 'code'])

        day_datas = day_datas.groupby(['date']).apply(self.date_group_data).reset_index(drop=True)
        day_datas = day_datas.sort_values(by=['code', 'date'], ascending=[True, True])

        return day_datas

    def create_feature(self, raws_datas):

        features = raws_datas.copy().sort_values(by=['date'], ascending=[True]).reset_index(drop=True)
        code = features.code.iloc[0]

        if len(features) < timesteps:
            return np.array([]), np.array([])

        # 创造特征
        # 收盘价的均值
        features['date1'] = features['date'].shift(-1)
        features['date2'] = features['date'].shift(-2)
        features['ytrue'] = features['close'].shift(-2) / features['open'].shift(-1) - 1
        features['close_ma_5'] = ts_ma(features['close'], 5)
        features['close_ma_10'] = ts_ma(features['close'], 10)
        features['close_ma_20'] = ts_ma(features['close'], 20)
        features['close_ma_30'] = ts_ma(features['close'], 30)
        features['close_ma_60'] = ts_ma(features['close'], 60)
        features['open_ma_5'] = ts_ma(features['open'], 5)
        features['open_ma_10'] = ts_ma(features['open'], 10)
        features['open_ma_20'] = ts_ma(features['open'], 20)
        features['open_ma_30'] = ts_ma(features['open'], 30)
        features['open_ma_60'] = ts_ma(features['open'], 60)
        features['open_2len_10'] = 10 * ts_vardev(features['open'], 10) + np.power(features['open_ma_10'], 2)
        features['close_2len_10'] = 10 * ts_vardev(features['close'], 10) + np.power(features['close_ma_10'], 2)

        features['open_ma2_10'] = ts_ma2(features['open'], 10)
        features['close_ma2_10'] = ts_ma2(features['close'], 10)
        features['open_ma2_20'] = ts_ma2(features['open'], 20)
        features['close_ma2_20'] = ts_ma2(features['close'], 20)
        features['open_ma2_5'] = ts_ma2(features['open'], 5)
        features['close_ma2_5'] = ts_ma2(features['close'], 5)
        features['open_ma2_60'] = ts_ma2(features['open'], 60)
        features['close_ma2_60'] = ts_ma2(features['close'], 60)

        features['avg_ma2_10'] = ts_ma(ts_ma(features['avg'], 10), 10)
        features['avg_ma_10'] = ts_ma(features['avg'], 10)

        # amount
        features['amount_ma_5'] = ts_ma(features['amount'], 5)
        features['amount_ma_10'] = ts_ma(features['amount'], 10)
        features['amount_ma_30'] = ts_ma(features['amount'], 30)
        features['amount_ma_60'] = ts_ma(features['amount'], 60)

        # volume
        features['volume_ma_5'] = ts_ma(features['volume'], 5)
        features['volume_ma_10'] = ts_ma(features['volume'], 10)
        features['volume_ma_30'] = ts_ma(features['volume'], 30)
        features['volume_ma_60'] = ts_ma(features['volume'], 60)
        features['volume_ma2_10'] = ts_ma(ts_ma(features['volume'], 10), 10)
        features['VROC10'] = (features['volume'] - features['volume'].shift(10)) / features['volume'].shift(10)
        features['VROC30'] = (features['volume'] - features['volume'].shift(30)) / features['volume'].shift(30)
        features['VROC5'] = (features['volume'] - features['volume'].shift(5)) / features['volume'].shift(5)
        features['VROC60'] = (features['volume'] - features['volume'].shift(60)) / features['volume'].shift(60)

        features['VSTD5'] = ts_stddev(features['volume'], 5)
        features['VSTD10'] = ts_stddev(features['volume'], 10)
        features['VSTD20'] = ts_stddev(features['volume'], 20)
        features['VSTD30'] = ts_stddev(features['volume'], 30)
        features['VSTD60'] = ts_stddev(features['volume'], 60)
        features['VSTD10_ma_10'] = ts_ma(features['VSTD10'], 10)
        features['VSTD30_ma_10'] = ts_ma(features['VSTD30'], 10)
        features['VSTD_CORR_10_60'] = features['VSTD10'].rolling(10).corr(features['VSTD60'])
        features['VSTD_CORR_10_30'] = features['VSTD10'].rolling(10).corr(features['VSTD30'])
        features['VSTD_CORR_30_60'] = features['VSTD30'].rolling(10).corr(features['VSTD60'])

        features['turn_ma_10'] = ts_ma(features['turn'], 10)
        features['turn_ma_30'] = ts_ma(features['turn'], 30)
        features['turn_ma_60'] = ts_ma(features['turn'], 60)

        features['turn_CORR_60'] = features['turn_ma_10'].rolling(60).corr(features['turn_ma_30'])
        features['turn_CORR_30'] = features['turn_ma_10'].rolling(30).corr(features['turn_ma_30'])
        features['via_turn_10'] = ts_stddev(features['turn'], 10)
        features['via_turn_30'] = ts_stddev(features['turn'], 30)
        features['via_turn_CORR'] = features['via_turn_30'].rolling(30).corr(features['via_turn_10'])

        features['VDIFF'] = features['turn_ma_30'] - features['turn_ma_10']
        features['VDEA'] = ts_ma(features['VDIFF'], 15)
        features['VDIFF_CORR'] = features['VDEA'].rolling(60).corr(features['VDIFF'])
        features['VMACD'] = features['VDIFF'] - features['VDEA']
        features['VIAMACD_10'] = features['via_turn_10'] - features['via_turn_30'] - ts_ma(
            features['via_turn_10'] - features['via_turn_30'], 10)
        features['VIAMACD_30'] = features['via_turn_10'] - features['via_turn_30'] - ts_ma(
            features['via_turn_10'] - features['via_turn_30'], 30)
        features['VIAMACD_CORR'] = features['VIAMACD_30'].rolling(30).corr(features['VIAMACD_10'])
        features['VIAMACD_DIFF'] = features['VIAMACD_30'] - features['VIAMACD_10']

        features['rate_of_shr'] = protected_division(features['free_shr'], features['tot_shr'])

        features['day_DIFF_rate'] = (features['open'] - features['close'].shift(1))
        features['day_DIFF'] = (features['open'] - features['close'].shift(1)) / features['close'].shift(1)
        features['ytrue2'] = features['ytrue'].shift(2)

        # 中间价
        features['high_low'] = (features['high'] + features['low']) / 2
        features['middle'] = (features['close'] * 2 + features['high'] + features['low']) / 4
        features['middle_ma_5'] = ts_ma(features['middle'], 5)
        features['middle_std_5'] = ts_stddev(features['middle'], 5)
        features['middle_std_10'] = ts_stddev(features['middle'], 10)
        features['middle_std_20'] = ts_stddev(features['middle'], 20)
        features['middle_std_60'] = ts_stddev(features['middle'], 60)

        features['high_low_dif'] = features['high'] - features['low']
        features['high_close_dif'] = features['high'] - features['close']
        features['low_close_dif'] = features['low'] - features['close']

        # 差价平方
        features['high_close_dif_square'] = features['high_close_dif'] * features['high_close_dif']
        features['low_close_dif_square'] = features['low_close_dif'] * features['low_close_dif']
        features['sqrt_high_low_close'] = protected_sqrt(
            features['high_close_dif_square'] + features['low_close_dif_square'])

        st_data = RawData.st_data()
        try:
            st_data = st_data.groupby('code').get_group(code)
            for _, row in st_data.iterrows():
                features.drop(features[(features['date'] >= row.entry_dt) & (features['date'] <= row.remove_dt)].index,
                              inplace=True)
        except KeyError as e:
            pass

        columns = ['pre_close', 'date1', 'date2', 'ytrue', 'code', 'date'
            , 'high', 'low', 'close', 'avg', 'open', 'middle'
            , 'middle_std_5', 'middle_std_10', 'middle_std_20', 'middle_std_60'
            , 'close_ma_5', 'close_ma_60'
            , 'high52', 'low52'
            , 'volume', 'volume_ma_10', 'volume_ma_30', 'volume_ma_60'
            , 'amount', 'amount_ma_10', 'amount_ma_60'
            , 'pct_change', 'M_up_preclose_percent', 'amount_rank', 'rate_of_shr'
            , 'pe', 'pcf', 'ps', 'pe_ttm', 'ps_ttm'
            , 'VSTD10', 'VSTD30', 'VSTD60', 'VROC10'
            , 'VSTD_CORR_10_60', 'VSTD_CORR_30_60', 'VSTD_CORR_10_30'
            , 'via_turn_10', 'via_turn_30'
            , 'turn', 'turn_ma_10', 'turn_ma_30'
            , 'free_shr', 'tot_shr'
            , 'mv', 'dq_mv'

                   ]
        ''' , 'VDIFF', 'VDEA', 'VDIFF', 'VDEA' '''

        price_related_columns = ['high', 'low', 'close', 'avg', 'open', 'middle'
            , 'close_ma_5', 'close_ma_60'
            , 'high52', 'low52'
                                 ]
        volume_related_columns = ['volume', 'volume_ma_10', 'volume_ma_30', 'volume_ma_60']
        amount_related_columns = ['amount', 'amount_ma_10', 'amount_ma_60']
        unrelated_columns = ['pct_change', 'M_up_preclose_percent', 'amount_rank', 'rate_of_shr', 'VROC10', 'pcf'
                             ]
        ps_related_columns = ['ps', 'ps_ttm']
        rate_related_columns = ['pe', 'pe_ttm']
        turn_related_columns = ['turn', 'turn_ma_10', 'turn_ma_30']
        shizhi_related_columns = ['mv', 'dq_mv']
        shr_related_columns = ['free_shr', 'tot_shr']
        std1_related_columns = ['via_turn_10', 'via_turn_30']
        cor1_related_columns = ['VSTD10', 'VSTD30', 'VSTD60'
                               , ]
        cor2_related_columns = ['VSTD_CORR_10_60', 'VSTD_CORR_30_60', 'VSTD_CORR_10_30']
        cor3_related_columns = [ 'middle_std_5', 'middle_std_10', 'middle_std_20', 'middle_std_60']
        macd_related_columns = ['VIAMACD_10', 'VIAMACD_30']
        square_related_columns = ['high_close_dif_square', 'low_close_dif_square']
        # 去Nan值
        features = features[columns].copy()
        features = features.dropna()
        features = features.astype(float)
        features = features.reset_index(drop=True)

        features = features[features['date'] <= self.end_day]
        features = features[features['date'] >= self.start_day]
        column2index = {column: index for index, column in zip(range(len(columns)), columns)}

        # 数据标准化

        rawdata_x = []
        rawdata_y = []
        # 数据切片
        if len(features) < timesteps:
            return np.array([]), np.array([])

        elif len(features) > timesteps:
            data_gen = TimeseriesGenerator(
                data=np.asarray(features),
                targets=np.asarray(features[['code', 'date', 'ytrue']].shift(1)),
                length=timesteps,
                sampling_rate=1,
                stride=1,
                start_index=0,
                end_index=None,
                shuffle=False,
                reverse=False,
                batch_size=1,
            )
            for i in range(len(data_gen)):
                x, y = data_gen[i]

                x = np.squeeze(x)
                y = np.squeeze(y)
                rawdata_x.append(x)
                rawdata_y.append(y)

        # 手动添加最后一个样本
        # 最后一个样本需要检查是否在预测时间范围内
        x = np.array(features[columns].iloc[-timesteps:])
        y = np.array(features[['code', 'date', 'ytrue']].iloc[-1])
        rawdata_x.append(x)
        rawdata_y.append(y)

        data_x = []
        data_y = []
        for x, y in zip(rawdata_x, rawdata_y):
            sus = self.sudpended(x[0, 5], x[-1, 2], timesteps)  # 判断样本内交易日期的连续性和y两天连续性
            lim = self.last_day_limit(x[-1, 3], x[-1, 6], x[-1, 8])  # 判断样本最后一个交易日是否是一只涨停版
            if sus and lim:
                x[:, [column2index[x] for x in price_related_columns]] = self.global_standardscaler(
                    x[:, [column2index[x] for x in price_related_columns]])
                x[:, [column2index[x] for x in volume_related_columns]] = self.global_standardscaler(
                    x[:, [column2index[x] for x in volume_related_columns]])
                x[:, [column2index[x] for x in amount_related_columns]] = self.global_standardscaler(
                    x[:, [column2index[x] for x in amount_related_columns]])
                x[:, [column2index[x] for x in shr_related_columns]] = self.global_standardscaler(
                    x[:, [column2index[x] for x in shr_related_columns]])
                x[:, [column2index[x] for x in std1_related_columns]] = self.global_standardscaler(
                    x[:, [column2index[x] for x in std1_related_columns]])
                '''x[:, [column2index[x] for x in std2_related_columns]] = self.global_standardscaler(
                    x[:, [column2index[x] for x in std2_related_columns]])
                x[:, [column2index[x] for x in std3_related_columns]] = self.global_standardscaler(
                    x[:, [column2index[x] for x in std3_related_columns]])
                x[:, [column2index[x] for x in std4_related_columns]] = self.global_standardscaler(
                    x[:, [column2index[x] for x in std4_related_columns]])'''
                x[:, [column2index[x] for x in shizhi_related_columns]] = self.global_standardscaler(
                    x[:, [column2index[x] for x in shizhi_related_columns]])
                x[:, [column2index[x] for x in rate_related_columns]] = self.global_standardscaler(
                    x[:, [column2index[x] for x in rate_related_columns]])
                '''x[:, [column2index[x] for x in macd_related_columns]] = self.global_standardscaler(
                    x[:, [column2index[x] for x in macd_related_columns]])
                x[:, [column2index[x] for x in pcf_related_columns]] = self.global_standardscaler(
                    x[:, [column2index[x] for x in pcf_related_columns]])'''
                x[:, [column2index[x] for x in ps_related_columns]] = self.global_standardscaler(
                    x[:, [column2index[x] for x in ps_related_columns]])
                x[:, [column2index[x] for x in turn_related_columns]] = self.global_standardscaler(
                    x[:, [column2index[x] for x in turn_related_columns]])
                x[:, [column2index[x] for x in cor1_related_columns]] = self.global_standardscaler(
                    x[:, [column2index[x] for x in cor1_related_columns]])
                x[:, [column2index[x] for x in cor2_related_columns]] = self.global_standardscaler(
                    x[:, [column2index[x] for x in cor2_related_columns]])
                x[:, [column2index[x] for x in cor3_related_columns]] = self.global_standardscaler(
                    x[:, [column2index[x] for x in cor3_related_columns]])
                '''x[:, [column2index[x] for x in square_related_columns]] = self.global_standardscaler(
                    x[:, [column2index[x] for x in square_related_columns]])'''
                x[:, [column2index[x] for x in unrelated_columns]] = self.local_standardscaler(
                    x[:, [column2index[x] for x in unrelated_columns]])
                data_x.append(x)
                data_y.append(y)
        # 最终处理之后的数据
        return np.array(data_x), np.array(data_y)


if __name__ == '__main__':
    import sys

    try:
        create_type = sys.argv[1]
    except:
        create_type = 'D2'
    # 生成回测数据

    # 生成训练数据
    data = eval('%s()' % create_type)  # 实例化数据对象
    data.set_dates('train', train_start_date, train_end_date)
    data.create()
