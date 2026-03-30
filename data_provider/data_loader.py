import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import glob

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', train_ratio=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_ratio = train_ratio
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        train_ratio = self.train_ratio
        if train_ratio is None or train_ratio > 0.6 and train_ratio <= 0.7:
            border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
        else:
            assert train_ratio > 0 and train_ratio <= 0.6
            num_train = int(12 * 30 * 24 / 0.6 * train_ratio)
            cropped_len = 12 * 30 * 24 - num_train
            border1s = [cropped_len, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', train_ratio=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_ratio = train_ratio
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        train_ratio = self.train_ratio
        if train_ratio is None or train_ratio > 0.6 and train_ratio <= 0.7:
            border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
        else:
            assert train_ratio > 0 and train_ratio <= 0.6
            num_train = int(12 * 30 * 24 * 4 / 0.6 * train_ratio)
            cropped_len = 12 * 30 * 24 * 4 - num_train
            border1s = [cropped_len, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', train_ratio=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_ratio = train_ratio
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        
        train_ratio = self.train_ratio
        if train_ratio is None:
            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            num_vali = len(df_raw) - num_train - num_test
            border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
        else:
            assert train_ratio > 0 and train_ratio <= 0.7
            num_train = int(len(df_raw) * train_ratio)
            cropped_len = int(len(df_raw) * 0.7) - num_train
            num_test = int(len(df_raw) * 0.2)
            num_vali = len(df_raw) - num_train - num_test - cropped_len
            border1s = [cropped_len, cropped_len + num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
            border2s = [cropped_len + num_train, cropped_len + num_train + num_vali, len(df_raw)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['pred']
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# ============================================================
# PhysioNet 2012 数据集
# ============================================================

# PhysioNet 2012 中的37个时序变量
PHYSIONET2012_VARIABLES = [
    'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN', 'Cholesterol',
    'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT',
    'HR', 'K', 'Lactate', 'Mg', 'MAP', 'MechVent', 'Na', 'NIDiasABP',
    'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate',
    'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC',
    'Weight'
]

# 静态描述变量 (在 Time=00:00 记录)
PHYSIONET2012_GENERAL_DESC = ['RecordID', 'Age', 'Gender', 'Height', 'ICUType', 'Weight']


def _parse_physionet_time(time_str):
    """将 'HH:MM' 格式时间转为连续小时数 (float)"""
    parts = str(time_str).split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    return hours + minutes / 60.0


def _load_single_patient(filepath):
    """
    加载单个病人文件。
    返回: general_desc (dict), time_series (list of (time_hours, param, value))
    """
    df = pd.read_csv(filepath)
    
    general_desc = {}
    time_series = []
    
    for _, row in df.iterrows():
        time_str = str(row['Time'])
        param = str(row['Parameter'])
        value = row['Value']
        
        # 跳过无效值
        if pd.isna(value) or value == -1:
            continue
        
        time_hours = _parse_physionet_time(time_str)
        
        if time_str == '00:00' and param in PHYSIONET2012_GENERAL_DESC:
            general_desc[param] = value
        else:
            if param in PHYSIONET2012_VARIABLES:
                time_series.append((time_hours, param, float(value)))
    
    return general_desc, time_series


def _resample_patient_to_grid(time_series, grid_hours, variables):
    """
    将不规则采样数据重采样到固定时间网格。
    
    策略: 每个网格点取最近观测值，缺失时前向+后向填充。
    同时返回观测掩码和实际时间戳（用于 continuous-time embedding）。
    
    Returns:
        values: [grid_len, num_vars]
        masks:  [grid_len, num_vars] (1=有观测)
        actual_times: [grid_len] 每个网格点最近实际观测时间（小时），
                      无观测时使用网格时间本身
    """
    num_vars = len(variables)
    grid_len = len(grid_hours)
    var_to_idx = {v: i for i, v in enumerate(variables)}
    
    values = np.full((grid_len, num_vars), np.nan)
    masks = np.zeros((grid_len, num_vars))
    # 记录每个网格点各变量的实际观测时间
    actual_obs_times = np.full((grid_len, num_vars), np.nan)
    
    for t, param, val in time_series:
        if param not in var_to_idx:
            continue
        var_idx = var_to_idx[param]
        grid_idx = np.argmin(np.abs(grid_hours - t))
        values[grid_idx, var_idx] = val
        masks[grid_idx, var_idx] = 1.0
        actual_obs_times[grid_idx, var_idx] = t
    
    # 缺失位置填 0 (不做 ffill/bfill, 与 GraFITi/TSDM 协议一致)
    # mask=1 的位置保留真实值, mask=0 的位置为 0
    values = np.nan_to_num(values, nan=0.0)
    
    # 对于连续时间戳：取每行所有变量中有实际观测的时间的均值，
    # 若该行无任何观测，使用网格时间
    actual_times = np.copy(grid_hours).astype(np.float32)
    for i in range(grid_len):
        obs_mask = ~np.isnan(actual_obs_times[i])
        if obs_mask.any():
            actual_times[i] = np.nanmean(actual_obs_times[i])
    
    return values.astype(np.float32), masks.astype(np.float32), actual_times


class Dataset_Physionet2012(Dataset):
    """
    PhysioNet 2012 ICU 不规则时间序列数据集。
    
    特点:
    - 不规则采样的多变量时间序列 (37个变量)
    - 每个病人有48小时ICU数据
    - set-a: 训练集, set-b: 验证集, set-c: 测试集
    - 返回连续时间戳 (小时) 用于 continuous-time embedding
    
    __getitem__ 返回 6 个元素:
        seq_x, seq_y, seq_x_mark, seq_y_mark, timestamps_x, timestamps_y
    
    其中 timestamps_x/timestamps_y 是连续小时值，直接传入 DataEmbedding 的
    time_aware 模式。
    """
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='Physionet2012',
                 target='HR', scale=True, timeenc=0, freq='h',
                 train_ratio=None,
                 resample_interval=1.0,
                 max_hours=48.0):
        """
        Args:
            root_path: 数据根目录 (默认 './dataset')
            flag: 'train'/'val'/'test'
            size: [seq_len, label_len, pred_len]
            features: 'M'/'MS'/'S'
            data_path: 子目录名 (默认 'Physionet2012')
            target: 目标变量 (默认 'HR')
            resample_interval: 重采样间隔 (小时), 默认1小时
            max_hours: 最大时间跨度, 默认48小时
        """
        if size is None:
            self.seq_len = 24
            self.label_len = 12
            self.pred_len = 12
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        assert flag in ['train', 'val', 'test']
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_ratio = train_ratio
        self.resample_interval = resample_interval
        self.max_hours = max_hours
        self.root_path = root_path
        self.data_path = data_path
        self.variables = PHYSIONET2012_VARIABLES
        self.num_variables = len(self.variables)
        
        self.grid_hours = np.arange(0, max_hours, resample_interval).astype(np.float32)
        self.grid_len = len(self.grid_hours)
        
        self.__read_data__()
    
    def __read_data__(self):
        """
        与 GraFITi (AAAI 2024) 对齐的数据加载协议:
        1. 合并 set-a/b/c 全部 12000 名患者
        2. 随机划分 70%/20%/10% → train/val/test (固定 seed)
        3. 每个病人 = 1 个样本 (观测前 seq_len 小时, 预测后 pred_len 小时)
        4. 标准化只在训练集上 fit
        
        缓存机制: 首次解析 CSV 后保存为 .npy, 后续直接加载 (~0.5s vs ~60s)
        """
        base_dir = os.path.join(self.root_path, self.data_path)
        
        # ---- 缓存文件路径 ----
        cache_dir = os.path.join(base_dir, 'cache')
        cache_values = os.path.join(cache_dir, 'all_values.npy')
        cache_masks = os.path.join(cache_dir, 'all_masks.npy')
        cache_timestamps = os.path.join(cache_dir, 'all_timestamps.npy')
        
        if os.path.exists(cache_values) and os.path.exists(cache_masks) and os.path.exists(cache_timestamps):
            # ---- 快速加载缓存 ----
            print(f"[Physionet2012] Loading cached data from {cache_dir}")
            all_values = np.load(cache_values)
            all_masks = np.load(cache_masks)
            all_timestamps = np.load(cache_timestamps)
            print(f"[Physionet2012] Loaded {len(all_values)} patients from cache")
        else:
            # ---- 首次: 解析全部 CSV 并缓存 ----
            all_patient_files = []
            for set_name in ['set-a', 'set-b', 'set-c']:
                set_dir = os.path.join(base_dir, set_name)
                if not os.path.exists(set_dir):
                    continue
                files = sorted(glob.glob(os.path.join(set_dir, '*.txt')))
                if not files:
                    files = sorted(glob.glob(os.path.join(set_dir, '*.csv')))
                all_patient_files.extend(files)
            
            print(f"[Physionet2012] First run: parsing {len(all_patient_files)} CSV files...")
            
            all_values_list = []
            all_masks_list = []
            all_timestamps_list = []
            
            for fpath in all_patient_files:
                general_desc, time_series = _load_single_patient(fpath)
                values, masks, timestamps = _resample_patient_to_grid(
                    time_series, self.grid_hours, self.variables
                )
                all_values_list.append(values)
                all_masks_list.append(masks)
                all_timestamps_list.append(timestamps)
            
            all_values = np.array(all_values_list, dtype=np.float32)
            all_masks = np.array(all_masks_list, dtype=np.float32)
            all_timestamps = np.array(all_timestamps_list, dtype=np.float32)
            
            # 保存缓存
            os.makedirs(cache_dir, exist_ok=True)
            np.save(cache_values, all_values)
            np.save(cache_masks, all_masks)
            np.save(cache_timestamps, all_timestamps)
            print(f"[Physionet2012] Cached to {cache_dir} ({all_values.shape})")
        
        # ---- 随机划分 70%/20%/10% (固定 seed, 与 GraFITi 协议一致) ----
        num_total = len(all_values)
        rng = np.random.RandomState(2021)
        indices = rng.permutation(num_total)
        
        n_train = int(num_total * 0.7)
        n_val = int(num_total * 0.2)
        # n_test = num_total - n_train - n_val
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        flag_to_idx = {'train': train_idx, 'val': val_idx, 'test': test_idx}
        sel_idx = flag_to_idx[self.flag]
        
        self.raw_values = all_values[sel_idx]
        self.all_masks = all_masks[sel_idx]
        self.all_timestamps = all_timestamps[sel_idx]
        
        # ---- 标准化 (只在训练集的真实观测值上 fit, 与 GraFITi/TSDM 一致) ----
        self.scaler = StandardScaler()
        scaler_path = os.path.join(base_dir, 'scaler_params.npz')
        
        if self.scale:
            train_masks = all_masks[train_idx]  # 需要用全局的 train_idx
            
            if self.flag == 'train':
                # 只在 mask=1 的真实观测值上 fit
                train_vals = self.raw_values  # [n_train, grid_len, C]
                train_ms = train_masks if self.flag == 'train' else all_masks[train_idx]
                
                # 按变量收集真实观测值来 fit
                means = np.zeros(self.num_variables)
                scales = np.zeros(self.num_variables)
                MIN_SCALE = 0.1  # 防止二值/近常数变量导致标准化后值爆炸
                for c in range(self.num_variables):
                    observed = train_vals[:, :, c][train_ms[:, :, c] == 1]
                    if len(observed) > 0:
                        means[c] = observed.mean()
                        scales[c] = max(observed.std(), MIN_SCALE)
                    else:
                        means[c] = 0.0
                        scales[c] = 1.0
                
                self.scaler.mean_ = means
                self.scaler.scale_ = scales
                self.scaler.var_ = scales ** 2
                self.scaler.n_features_in_ = self.num_variables
                
                # 标准化: 只变换 mask=1 的位置, mask=0 保持 0
                self.data_values = (self.raw_values - means) / scales
                self.data_values = self.data_values * self.all_masks  # mask=0 位置归零
                
                np.savez(scaler_path, mean=means, scale=scales)
            else:
                if os.path.exists(scaler_path):
                    params = np.load(scaler_path)
                    self.scaler.mean_ = params['mean']
                    self.scaler.scale_ = params['scale']
                    self.scaler.var_ = params['scale'] ** 2
                    self.scaler.n_features_in_ = len(params['mean'])
                else:
                    print(f"[Warning] scaler_params.npz not found, fitting on {self.flag}")
                    means = np.zeros(self.num_variables)
                    scales = np.ones(self.num_variables)
                    for c in range(self.num_variables):
                        train_vals_c = all_values[train_idx][:, :, c]
                        train_masks_c = all_masks[train_idx][:, :, c]
                        observed = train_vals_c[train_masks_c == 1]
                        if len(observed) > 0:
                            means[c] = observed.mean()
                            scales[c] = max(observed.std(), 1e-8)
                    self.scaler.mean_ = means
                    self.scaler.scale_ = scales
                
                self.data_values = (self.raw_values - self.scaler.mean_) / self.scaler.scale_
                self.data_values = self.data_values * self.all_masks  # mask=0 位置归零
        else:
            self.data_values = self.raw_values.copy()
        
        # 按 features 选择变量
        if self.features == 'S':
            target_idx = self.variables.index(self.target)
            self.data_values = self.data_values[:, :, target_idx:target_idx+1]
            self.all_masks = self.all_masks[:, :, target_idx:target_idx+1]
        
        # 生成离散时间标记 (兼容 TemporalEmbedding)
        self._generate_time_marks()
        
        # ---- 每个病人 = 1 个样本 (与 GraFITi 一致) ----
        # 观测: [0, seq_len), 预测: [seq_len, seq_len + pred_len)
        self.samples_per_patient = 1
        self.num_patients = len(sel_idx)
        self.total_samples = self.num_patients
        
        print(f"[Physionet2012] {self.flag}: {self.num_patients} patients, "
              f"1 sample/patient (obs={self.seq_len}h, pred={self.pred_len}h), "
              f"grid_len={self.grid_len}, vars={self.data_values.shape[-1]}")
    
    def _generate_time_marks(self):
        """生成离散时间标记 (兼容传统 TemporalEmbedding)"""
        grid_len = self.grid_len
        if self.timeenc == 0:
            time_marks = np.zeros((grid_len, 4), dtype=np.float32)
            time_marks[:, 0] = 1                                    # month
            time_marks[:, 1] = (self.grid_hours // 24).astype(int) + 1  # day
            time_marks[:, 2] = 0                                    # weekday
            time_marks[:, 3] = self.grid_hours % 24                 # hour
        else:
            time_marks = np.zeros((grid_len, 4), dtype=np.float32)
            time_marks[:, 0] = self.grid_hours / self.max_hours
            time_marks[:, 1] = np.sin(2 * np.pi * self.grid_hours / 24)
            time_marks[:, 2] = np.cos(2 * np.pi * self.grid_hours / 24)
            time_marks[:, 3] = self.grid_hours / 24
        self.time_marks = time_marks
    
    def __getitem__(self, index):
        """
        每个病人 = 1 个样本 (与 GraFITi 协议一致)
        观测: [0, seq_len), 预测: [seq_len, seq_len + pred_len)
        
        返回 8 元素:
            seq_x:            [seq_len, C]
            seq_y:            [label_len+pred_len, C]
            seq_x_mark:       [seq_len, 4]
            seq_y_mark:       [label_len+pred_len, 4]
            timestamps_x:     [seq_len]  连续时间(小时)
            timestamps_y:     [label_len+pred_len] 连续时间(小时)
            missing_mask_x:   [seq_len, C]
            missing_mask_y:   [label_len+pred_len, C]
        """
        patient_idx = index
        
        s_begin = 0
        s_end = self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data_values[patient_idx, s_begin:s_end, :]
        seq_y = self.data_values[patient_idx, r_begin:r_end, :]
        seq_x_mark = self.time_marks[s_begin:s_end, :]
        seq_y_mark = self.time_marks[r_begin:r_end, :]
        
        timestamps_x = self.all_timestamps[patient_idx, s_begin:s_end]
        timestamps_y = self.all_timestamps[patient_idx, r_begin:r_end]
        
        missing_mask_x = self.all_masks[patient_idx, s_begin:s_end, :]
        missing_mask_y = self.all_masks[patient_idx, r_begin:r_end, :]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark, timestamps_x, timestamps_y, missing_mask_x, missing_mask_y
    
    def __len__(self):
        return self.total_samples
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# ==============================================================================
# MIMIC-IV 数据集 (与 GraFITi / t-PatchGNN 协议对齐)
# ==============================================================================

class Dataset_MIMICIV(Dataset):
    """
    MIMIC-IV ICU 不规则时间序列数据集。
    
    支持两种数据格式:
    1. t-PatchGNN 处理后的 .pt 文件 (推荐)
       - torch.load 后得到 list of (record_id, timestamps, values, mask, labels)
    2. 预处理后的 .npy 文件 (与 PhysioNet 相同格式)
       - {split}_values.npy, {split}_masks.npy, {split}_timestamps.npy
    
    实验设置 (与 GraFITi AAAI 2024 Table 3 对齐):
    - 102 个时序变量, 48 小时, 1 小时重采样
    - 随机 70%/20%/10% 划分
    - 观测/预测: 24→12, 24→24, 36→6, 36→12
    """
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='MIMICIV',
                 target=None, scale=True, timeenc=0, freq='h',
                 train_ratio=None,
                 resample_interval=1.0,
                 max_hours=48.0):
        if size is None:
            self.seq_len = 24
            self.label_len = 12
            self.pred_len = 12
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        assert flag in ['train', 'val', 'test']
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.resample_interval = resample_interval
        self.max_hours = max_hours
        
        self.grid_hours = np.arange(0, max_hours, resample_interval).astype(np.float32)
        self.grid_len = len(self.grid_hours)
        
        self.__read_data__()
    
    def __read_data__(self):
        base_dir = os.path.join(self.root_path, self.data_path)
        
        # 尝试加载 t-PatchGNN .pt 格式
        pt_path = os.path.join(base_dir, 'mimiciv.pt')
        if not os.path.exists(pt_path):
            pt_path = os.path.join(base_dir, 'mimic.pt')
        
        npy_path = os.path.join(base_dir, 'all_values.npy')
        
        if os.path.exists(pt_path):
            self._load_from_pt(pt_path, base_dir)
        elif os.path.exists(npy_path):
            self._load_from_npy(base_dir)
        else:
            raise FileNotFoundError(
                f"MIMIC-IV data not found in {base_dir}. "
                f"Please place either 'mimiciv.pt' (t-PatchGNN format) "
                f"or 'all_values.npy' + 'all_masks.npy' + 'all_timestamps.npy'."
            )
    
    def _load_from_pt(self, pt_path, base_dir):
        """
        加载 .pt 文件，支持两种格式，带缓存机制。
        第一次运行时重采样并保存为 .npy，后续直接加载（~0.5s vs ~5min）。
        """
        # 检查缓存
        cache_vals = os.path.join(base_dir, 'cache_grid_values.npy')
        cache_masks = os.path.join(base_dir, 'cache_grid_masks.npy')
        cache_ts = os.path.join(base_dir, 'cache_grid_timestamps.npy')
        
        if os.path.exists(cache_vals) and os.path.exists(cache_masks) and os.path.exists(cache_ts):
            print(f"[MIMICIV] Loading from cache in {base_dir}...")
            all_values = np.load(cache_vals)
            all_masks = np.load(cache_masks)
            all_timestamps = np.load(cache_ts)
            self.num_variables = all_values.shape[-1]
            print(f"[MIMICIV] {len(all_values)} patients, {self.num_variables} variables (cached)")
            self._split_and_scale(all_values, all_masks, all_timestamps, base_dir)
            return
        
        print(f"[MIMICIV] Loading from {pt_path} (first run, will cache)...")
        raw_data = torch.load(pt_path, map_location='cpu')
        
        # 检测格式
        if isinstance(raw_data, dict) and 'data' in raw_data:
            data_list = raw_data['data']
            num_vars = raw_data['num_vars']
            self.num_variables = num_vars
            print(f"[MIMICIV] {len(data_list)} patients, {num_vars} variables (dict format)")
            
            all_values = []
            all_masks = []
            all_timestamps = []
            
            for i, item in enumerate(data_list):
                if (i + 1) % 2000 == 0:
                    print(f"  Resampling patient {i+1}/{len(data_list)}...")
                tt = item['times'].numpy().astype(np.float32)
                vals = item['values'].numpy().astype(np.float32)
                mask = item['masks'].numpy().astype(np.float32)
                
                grid_vals, grid_masks = self._resample_to_grid(tt, vals, mask, num_vars)
                all_values.append(grid_vals)
                all_masks.append(grid_masks)
                all_timestamps.append(self.grid_hours.copy())
        else:
            sample = raw_data[0]
            if len(sample) >= 4:
                num_vars = sample[2].shape[-1]
            else:
                raise ValueError(f"Unexpected .pt format: sample has {len(sample)} elements")
            
            self.num_variables = num_vars
            print(f"[MIMICIV] {len(raw_data)} patients, {num_vars} variables (tuple format)")
            
            all_values = []
            all_masks = []
            all_timestamps = []
            
            for i, item in enumerate(raw_data):
                if (i + 1) % 2000 == 0:
                    print(f"  Resampling patient {i+1}/{len(raw_data)}...")
                if len(item) >= 4:
                    _, tt, vals, mask = item[0], item[1], item[2], item[3]
                else:
                    continue
                
                tt = tt.numpy().astype(np.float32)
                vals = vals.numpy().astype(np.float32)
                mask = mask.numpy().astype(np.float32)
                
                grid_vals, grid_masks = self._resample_to_grid(tt, vals, mask, num_vars)
                all_values.append(grid_vals)
                all_masks.append(grid_masks)
                all_timestamps.append(self.grid_hours.copy())
        
        all_values = np.array(all_values, dtype=np.float32)
        all_masks = np.array(all_masks, dtype=np.float32)
        all_timestamps = np.array(all_timestamps, dtype=np.float32)
        
        # 保存缓存
        np.save(cache_vals, all_values)
        np.save(cache_masks, all_masks)
        np.save(cache_ts, all_timestamps)
        print(f"[MIMICIV] Cache saved to {base_dir}")
        
        self._split_and_scale(all_values, all_masks, all_timestamps, base_dir)
    
    def _resample_to_grid(self, tt, vals, mask, num_vars):
        """将不规则时间序列重采样到固定网格, 每个网格点取该时间窗口内观测的均值"""
        grid_vals = np.zeros((self.grid_len, num_vars), dtype=np.float32)
        grid_masks = np.zeros((self.grid_len, num_vars), dtype=np.float32)
        
        half = self.resample_interval / 2.0
        
        for i, gh in enumerate(self.grid_hours):
            in_window = (tt >= gh - half) & (tt < gh + half)
            if not in_window.any():
                continue
            
            window_vals = vals[in_window]
            window_mask = mask[in_window]
            
            for c in range(num_vars):
                obs_idx = window_mask[:, c] > 0
                if obs_idx.any():
                    grid_vals[i, c] = window_vals[obs_idx, c].mean()
                    grid_masks[i, c] = 1.0
        
        # LOCF (Last Observation Carried Forward) 前向填充 + 后向填充
        # 只改 values，mask 保持不变（评估时仍只看 mask=1 的位置）
        # 目的: 消除大量的 0 值，让 RevIN 和 Transformer 看到合理的输入
        import pandas as pd
        df_vals = pd.DataFrame(grid_vals)
        df_vals = df_vals.replace(0, np.nan)  # 把未观测的 0 变成 NaN
        # 但要保留真实观测到的 0 值
        for c in range(num_vars):
            for i in range(self.grid_len):
                if grid_masks[i, c] == 1.0:
                    df_vals.iloc[i, c] = grid_vals[i, c]  # 恢复真实值（包括真实的 0）
        df_vals = df_vals.ffill().bfill().fillna(0.0)
        grid_vals = df_vals.values.astype(np.float32)
        
        return grid_vals, grid_masks
    
    def _load_from_npy(self, base_dir):
        """加载预处理好的 .npy 文件"""
        print(f"[MIMICIV] Loading from .npy files in {base_dir}...")
        all_values = np.load(os.path.join(base_dir, 'all_values.npy'))
        all_masks = np.load(os.path.join(base_dir, 'all_masks.npy'))
        all_timestamps = np.load(os.path.join(base_dir, 'all_timestamps.npy'))
        self.num_variables = all_values.shape[-1]
        
        self._split_and_scale(all_values, all_masks, all_timestamps, base_dir)
    
    def _split_and_scale(self, all_values, all_masks, all_timestamps, base_dir):
        """随机 70/20/10 划分 + 标准化"""
        num_total = len(all_values)
        rng = np.random.RandomState(2021)
        indices = rng.permutation(num_total)
        
        n_train = int(num_total * 0.7)
        n_val = int(num_total * 0.2)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        flag_to_idx = {'train': train_idx, 'val': val_idx, 'test': test_idx}
        sel_idx = flag_to_idx[self.flag]
        
        self.raw_values = all_values[sel_idx]
        self.all_masks = all_masks[sel_idx]
        self.all_timestamps = all_timestamps[sel_idx]
        
        # ---- 标准化 (只在训练集的真实观测值上 fit, 与 GraFITi/TSDM 一致) ----
        self.scaler = StandardScaler()
        scaler_path = os.path.join(base_dir, 'scaler_params.npz')
        
        train_masks = all_masks[train_idx]
        
        if self.scale:
            if self.flag == 'train':
                train_vals = self.raw_values
                train_ms = self.all_masks
                
                means = np.zeros(self.num_variables)
                scales = np.zeros(self.num_variables)
                MIN_SCALE = 0.1  # 防止二值/近常数变量导致标准化后值爆炸
                for c in range(self.num_variables):
                    observed = train_vals[:, :, c][train_ms[:, :, c] == 1]
                    if len(observed) > 0:
                        means[c] = observed.mean()
                        scales[c] = max(observed.std(), MIN_SCALE)
                    else:
                        means[c] = 0.0
                        scales[c] = 1.0
                
                self.scaler.mean_ = means
                self.scaler.scale_ = scales
                self.scaler.var_ = scales ** 2
                self.scaler.n_features_in_ = self.num_variables
                
                self.data_values = (self.raw_values - means) / scales
                self.data_values = self.data_values * self.all_masks
                
                np.savez(scaler_path, mean=means, scale=scales)
            else:
                if os.path.exists(scaler_path):
                    params = np.load(scaler_path)
                    self.scaler.mean_ = params['mean']
                    self.scaler.scale_ = params['scale']
                    self.scaler.var_ = params['scale'] ** 2
                    self.scaler.n_features_in_ = len(params['mean'])
                else:
                    print(f"[Warning] scaler_params.npz not found, fitting on train")
                    means = np.zeros(self.num_variables)
                    scales = np.ones(self.num_variables)
                    for c in range(self.num_variables):
                        train_vals_c = all_values[train_idx][:, :, c]
                        train_masks_c = all_masks[train_idx][:, :, c]
                        observed = train_vals_c[train_masks_c == 1]
                        if len(observed) > 0:
                            means[c] = observed.mean()
                            scales[c] = max(observed.std(), MIN_SCALE)
                    self.scaler.mean_ = means
                    self.scaler.scale_ = scales
                
                self.data_values = (self.raw_values - self.scaler.mean_) / self.scaler.scale_
                self.data_values = self.data_values * self.all_masks
        else:
            self.data_values = self.raw_values.copy()
        
        if self.features == 'S' and self.target is not None:
            target_idx = self.target  # for MIMIC-IV, target is an index
            self.data_values = self.data_values[:, :, target_idx:target_idx+1]
            self.all_masks = self.all_masks[:, :, target_idx:target_idx+1]
        
        # 生成时间标记
        self._generate_time_marks()
        
        # 每病人1样本
        self.samples_per_patient = 1
        self.num_patients = len(sel_idx)
        self.total_samples = self.num_patients
        
        print(f"[MIMICIV] {self.flag}: {self.num_patients} patients, "
              f"1 sample/patient (obs={self.seq_len}h, pred={self.pred_len}h), "
              f"grid_len={self.grid_len}, vars={self.data_values.shape[-1]}")
    
    def _generate_time_marks(self):
        grid_len = self.grid_len
        if self.timeenc == 0:
            time_marks = np.zeros((grid_len, 4), dtype=np.float32)
            time_marks[:, 0] = 1
            time_marks[:, 1] = (self.grid_hours // 24).astype(int) + 1
            time_marks[:, 2] = 0
            time_marks[:, 3] = self.grid_hours % 24
        else:
            time_marks = np.zeros((grid_len, 4), dtype=np.float32)
            time_marks[:, 0] = self.grid_hours / self.max_hours
            time_marks[:, 1] = np.sin(2 * np.pi * self.grid_hours / 24)
            time_marks[:, 2] = np.cos(2 * np.pi * self.grid_hours / 24)
            time_marks[:, 3] = self.grid_hours / 24
        self.time_marks = time_marks
    
    def __getitem__(self, index):
        patient_idx = index
        
        s_begin = 0
        s_end = self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data_values[patient_idx, s_begin:s_end, :]
        seq_y = self.data_values[patient_idx, r_begin:r_end, :]
        seq_x_mark = self.time_marks[s_begin:s_end, :]
        seq_y_mark = self.time_marks[r_begin:r_end, :]
        
        timestamps_x = self.all_timestamps[patient_idx, s_begin:s_end]
        timestamps_y = self.all_timestamps[patient_idx, r_begin:r_end]
        
        missing_mask_x = self.all_masks[patient_idx, s_begin:s_end, :]
        missing_mask_y = self.all_masks[patient_idx, r_begin:r_end, :]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark, timestamps_x, timestamps_y, missing_mask_x, missing_mask_y
    
    def __len__(self):
        return self.total_samples
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# ==============================================================================
# MIMIC-III 数据集 (复用 Dataset_MIMICIV 的逻辑)
# ==============================================================================
# MIMIC-III 和 MIMIC-IV 的处理流程完全相同:
# 同样是 .pt 格式, 同样重采样到 1h 网格, 同样的划分和标准化
# 唯一区别: 变量数不同 (MIMIC-III ~96个, MIMIC-IV 102个), 由数据自动检测

class Dataset_MIMICIII(Dataset_MIMICIV):
    """
    MIMIC-III ICU 不规则时间序列数据集。
    继承 Dataset_MIMICIV，仅修改默认路径和 .pt 文件名。
    """
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='MIMIC-III',
                 target=None, scale=True, timeenc=0, freq='h',
                 train_ratio=None,
                 resample_interval=1.0,
                 max_hours=48.0):
        super().__init__(
            root_path=root_path, flag=flag, size=size,
            features=features, data_path=data_path,
            target=target, scale=scale, timeenc=timeenc, freq=freq,
            train_ratio=train_ratio,
            resample_interval=resample_interval,
            max_hours=max_hours,
        )
    
    def __read_data__(self):
        base_dir = os.path.join(self.root_path, self.data_path)
        
        # 尝试多种 .pt 文件名
        for pt_name in ['mimiciii.pt', 'mimic.pt', 'mimiciv.pt']:
            pt_path = os.path.join(base_dir, pt_name)
            if os.path.exists(pt_path):
                self._load_from_pt(pt_path, base_dir)
                return
        
        npy_path = os.path.join(base_dir, 'all_values.npy')
        if os.path.exists(npy_path):
            self._load_from_npy(base_dir)
            return
        
        raise FileNotFoundError(
            f"MIMIC-III data not found in {base_dir}. "
            f"Please place 'mimiciii.pt' or 'mimic.pt'."
        )