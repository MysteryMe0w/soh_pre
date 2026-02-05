# %%
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
cluster_folder = 'cluster 1'  # cluster文件夹名称
# 3. 构建时间序列
window_size =25
step = 1        # 当前时刻
future_steps = 10  # 预测未来10步
# 设置数据路径
base_path = r'E:\PythonProjects\202403cluster预测 - 训练测试分开\202403'

# 获取cluster文件夹
cluster_folders = [f for f in os.listdir(base_path) if f.startswith('cluster') and cluster_folder == f]

# 定义读取state文件的函数（每个电池簇都有自己的 total state 文件夹）
def read_state_files(cluster_folder, cluster_number):
    state_data = []
    total_state_folder = os.path.join(cluster_folder, 'total state')
    for file_name in os.listdir(total_state_folder):
        if file_name.endswith('.csv'):
            # print(f"读取文件: {total_state_folder}\\{file_name}")
            file_path = os.path.join(total_state_folder, file_name)
            data = pd.read_csv(file_path, encoding='gbk')  # 可能需要根据实际情况调整编码
            state_data.append(data)
    
    # 合并数据并动态修改列名
    state_data = pd.concat(state_data, ignore_index=True)
    
    # 动态修改列名，例如 'Cluster1_Voltage' 变为 'Voltage'
    state_data.columns = [col.replace(f'Cluster{cluster_number}_', '') for col in state_data.columns]
    
    return state_data

# 获取充放电循环的位置                
def find_charge_discharge_cycles(state_data):
    charge_discharge_cycles = []
    for i in tqdm(range(1, len(state_data))):
        # 假设 'Charge_State' 是表示充放电状态的列
        if state_data.iloc[i-1]['Charge_State'] == 1 and state_data.iloc[i]['Charge_State'] == 2:
            start_time = state_data.iloc[i]['MCGS_TIME']  # 假设 'MCGS_TIME' 是时间列
            charge_discharge_cycles.append(('discharge_end', start_time))
        elif state_data.iloc[i-1]['Charge_State'] == 2 and state_data.iloc[i]['Charge_State'] == 1:
            start_time = state_data.iloc[i]['MCGS_TIME']
            charge_discharge_cycles.append(('charge_end', start_time))
    return charge_discharge_cycles

# 读取电压和温度数据
def read_cluster_data(cluster_folder, cluster_number):
    voltage_data = []
    temperature_data = []
    
    voltage_folder = os.path.join(cluster_folder, 'voltage')
    temperature_folder = os.path.join(cluster_folder, 'temperature')
    
    for file_name in os.listdir(voltage_folder):
        if file_name.endswith('.csv'):
            # print(f"读取电压文件: {voltage_folder}\\{file_name}")
            voltage_file = os.path.join(voltage_folder, file_name)
            voltage_data.append(pd.read_csv(voltage_file))
    
    for file_name in os.listdir(temperature_folder):
        if file_name.endswith('.csv'):
            # print(f"读取温度文件: {temperature_folder}\\{file_name}")
            temperature_file = os.path.join(temperature_folder, file_name)
            temperature_data.append(pd.read_csv(temperature_file))
    
    voltage_data = pd.concat(voltage_data, ignore_index=True)
    temperature_data = pd.concat(temperature_data, ignore_index=True)
    
    # 动态修改电压和温度列名，例如 'Cluster1_Voltage' 变为 'Voltage'
    voltage_data.columns = [col.replace(f'Cluster{cluster_number}', '') for col in voltage_data.columns]
    temperature_data.columns = [col.replace(f'Cluster{cluster_number}', '') for col in temperature_data.columns]
    
    return voltage_data, temperature_data


# 构建特征数据集的函数，基于时间戳对齐数据
def construct_features(state_data, voltage_data, temperature_data):
    features = pd.DataFrame()

    # 结合 MCGS_TIME 和 MCGS_TIMEMS，生成精确到毫秒的时间戳
    state_data['MCGS_TIME_FULL'] = pd.to_datetime(state_data['MCGS_TIME'], format='%Y/%m/%d %H:%M:%S') + pd.to_timedelta(state_data['MCGS_TIMEMS'], unit='ms')
    voltage_data['MCGS_TIME_FULL'] = pd.to_datetime(voltage_data['MCGS_TIME'], format='%Y/%m/%d %H:%M:%S') + pd.to_timedelta(voltage_data['MCGS_TIMEMS'], unit='ms')
    temperature_data['MCGS_TIME_FULL'] = pd.to_datetime(temperature_data['MCGS_TIME'], format='%Y/%m/%d %H:%M:%S') + pd.to_timedelta(temperature_data['MCGS_TIMEMS'], unit='ms')

    # 确保时间列排序
    state_data = state_data.sort_values('MCGS_TIME_FULL')
    voltage_data = voltage_data.sort_values('MCGS_TIME_FULL')
    temperature_data = temperature_data.sort_values('MCGS_TIME_FULL')

    # 合并数据，基于 MCGS_TIME_FULL 对齐
    merged_data = pd.merge_asof(state_data[['MCGS_TIME_FULL', 'SOC', 'SOH']],
                                voltage_data[['MCGS_TIME_FULL'] + [col for col in voltage_data.columns if col.startswith('Cluster')]],
                                on='MCGS_TIME_FULL',
                                direction='nearest')

    merged_data = pd.merge_asof(merged_data,
                                temperature_data[['MCGS_TIME_FULL'] + [col for col in temperature_data.columns if col.startswith('Cluster')]],
                                on='MCGS_TIME_FULL',
                                direction='nearest')

    # 生成统计特征
    voltage_numeric = merged_data.select_dtypes(include=[np.number]).drop(columns=['SOC', 'SOH'])
    temperature_numeric = merged_data.select_dtypes(include=[np.number])

    features['Voltage_mean'] = voltage_numeric.mean(axis=1)
    features['Voltage_max'] = voltage_numeric.max(axis=1)
    features['Voltage_min'] = voltage_numeric.min(axis=1)
    features['Voltage_std'] = voltage_numeric.std(axis=1)
    
    features['Temperature_mean'] = temperature_numeric.mean(axis=1)
    features['Temperature_max'] = temperature_numeric.max(axis=1)
    features['Temperature_min'] = temperature_numeric.min(axis=1)
    features['Temperature_std'] = temperature_numeric.std(axis=1)

    # 放电/充电时间差（如果有）
    if 'Time_diff' in state_data.columns:
        features['Time_diff'] = merged_data['Time_diff']

    # 保留SOC和SOH
    features['SOC'] = merged_data['SOC'].ffill().bfill()
    features['SOH'] = merged_data['SOH'].ffill().bfill()

    # 添加时间戳列
    features['MCGS_TIME_FULL'] = merged_data['MCGS_TIME_FULL']

    return features



# 遍历每个 cluster 文件夹，处理数据，合并特征
all_features = []

for cluster in cluster_folders:  # 示例只处理前两个 cluster
    cluster_number = cluster.split('cluster')[-1].strip()  # 提取 cluster 编号
    cluster_path = os.path.join(base_path, cluster)
    
    # 读取当前簇的 total state 文件夹下的充放电状态数据
    state_data = read_state_files(cluster_path, cluster_number)
    
    # 查找充放电循环
    print(f"处理{cluster}的充放电循环数据...")
    cycles = find_charge_discharge_cycles(state_data)
    
    # 读取电压和温度数据
    voltage_data, temperature_data = read_cluster_data(cluster_path, cluster_number)
    
    # 构建特征并处理 NaN
    features = construct_features(state_data, voltage_data, temperature_data)
    features['cluster'] = cluster
    all_features.append(features)

# 将所有簇的特征数据集整合，确保 NaN 得到妥善处理，并基于 MCGS_TIME_FULL 排序
final_features_df = pd.concat(all_features, ignore_index=True).sort_values(by='MCGS_TIME_FULL')

# 检查是否还有空值
missing_values = final_features_df.isnull().sum()
print("Missing values in final dataset:\n", missing_values)

# 检查 SOC 和 SOH 的时间序列是否正常
print(final_features_df[['MCGS_TIME_FULL', 'SOC', 'SOH']].head())

# %%
final_features_df = final_features_df.dropna()
# final_features_df.to_csv("processed_data",index=None)

# %% [markdown]
# ##　SOH

# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 卡尔曼滤波函数
def kalman_filter_fuc(observations, process_variance, measurement_variance, initial_estimate, initial_uncertainty):
    n = len(observations)
    estimates = np.zeros(n)
    estimate = initial_estimate
    uncertainty = initial_uncertainty

    for t in range(n):
        uncertainty += process_variance
        kalman_gain = uncertainty / (uncertainty + measurement_variance)
        estimate += kalman_gain * (observations[t] - estimate)
        uncertainty = (1 - kalman_gain) * uncertainty
        estimates[t] = estimate
    return estimates

# 1. 卡尔曼滤波处理并生成新的features
def apply_kalman_filter(df, features):
    for feature in features:
        df[feature + '_kalman'] = kalman_filter_fuc(
            df[feature].values, process_variance=1e-4, measurement_variance=0.5**2, initial_estimate=0.0, initial_uncertainty=1.0
        )
    all_features = features + [f + '_kalman' for f in features]
    return df, all_features

# 2. 对all_features和target进行归一化处理
def normalize_features_and_target(df, all_features, target_column):
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    df[all_features] = feature_scaler.fit_transform(df[all_features])
    df[[target_column]] = target_scaler.fit_transform(df[[target_column]])

    return df, feature_scaler, target_scaler


# 修改后的create_sequences函数
def create_sequences(df, all_features, target_column, window_size, future_steps=10,step=1):
    sequences, targets, clusters, times = [], [], [], []
    
    # 按时间排序
    df_sorted = df.sort_values(by='MCGS_TIME_FULL')

    # 按每个cluster生成序列
    for cluster in df_sorted['cluster'].unique():
        cluster_data = df_sorted[df_sorted['cluster'] == cluster]
        
        for i in range(len(cluster_data) - window_size - future_steps + 1):
            # 输入序列保持不变
            sequence = cluster_data[all_features+[target_column]].iloc[i:i + window_size].values
            
            # 获取未来multiple_steps的目标值
            future_targets = []
            future_times = []
            for step in range(future_steps):
                target = cluster_data[target_column].iloc[i + window_size + step]
                time = cluster_data['MCGS_TIME_FULL'].iloc[i + window_size + step]
                future_targets.append(target)
                future_times.append(time)
            
            sequences.append(sequence)
            targets.append(future_targets)
            clusters.append(cluster)
            times.append(future_times)

    return (np.array(sequences), 
            np.array(targets),  # 现在是二维数组 [样本数, future_steps]
            np.array(clusters), 
            np.array(times))    # 现在是二维数组 [样本数, future_steps]

# 主流程修改
final_features_df1 = final_features_df.tail(int(len(final_features_df)*0.5))
sample_ratio = 0.1
step = int(1/sample_ratio)  # 如果是5%,则step=20
df = final_features_df1.iloc[::step]


df = df.sort_values(by=['MCGS_TIME_FULL'])
df = df.drop_duplicates(subset=['MCGS_TIME_FULL'])
features = ['Voltage_mean', 'Voltage_max', 'Voltage_min', 'Voltage_std', 
            'Temperature_mean', 'Temperature_max', 'Temperature_min', 'Temperature_std']
target_column = 'SOH'

# 1. 卡尔曼滤波处理
df, all_features = apply_kalman_filter(df, features)

# 2. 归一化特征和目标
df, feature_scaler, target_scaler = normalize_features_and_target(df, all_features, target_column)


X, y, clusters, times = create_sequences(df, all_features, target_column, 
                                       window_size, future_steps=future_steps,step=step)
print(y.shape)

from tensorflow.keras.layers import GRU, Dense,LSTM,Conv1D,Flatten,MaxPooling1D,SimpleRNN
# 4. 构建并训练模型
def build_rnn_model(input_shape):
    model = Sequential()
    model.add(SimpleRNN(64, input_shape=input_shape, return_sequences=False))
    model.add(Dense(32))
    model.add(Dense(y.shape[1]))
    model.compile(optimizer='adam', loss='mse')
    return model

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
    model.add(Dense(32))
    model.add(Dense(y.shape[1]))
    model.compile(optimizer='adam', loss='mse')
    return model

def build_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Dense(y.shape[1]))
    model.compile(optimizer='adam', loss='mse')
    return model


# 5. 绘制预测与真实值
def plot_predictions(cluster, time, predictions, true_values,save_name=''):
    plt.figure(figsize=(10, 6))
    
    # 按时间排序
    sorted_indices = np.argsort(time)
    sorted_time = time[sorted_indices]
    sorted_predictions = predictions[sorted_indices]
    sorted_true_values = true_values[sorted_indices]
    
    plt.plot(sorted_time, sorted_true_values, label="True Values")
    plt.plot(sorted_time, sorted_predictions, label="Predictions")
    plt.title(f'Cluster {cluster}: True vs Predictions')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_name}_predict.jpg",dpi=500)
    plt.show()



import os

exp_name = f"soh_step-{future_steps}"

eval_dic = {}
save_dir = "./soh_models"  # 保存模型的目录

from tensorflow.keras.models import load_model

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

# 确保保存结果的目录存在
results_dir = "./soh_results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# 绘制预测与真实值
def plot_predictions(cluster, time, predictions, true_values, save_name=''):
    plt.figure(figsize=(10, 6))
    
    # 按时间排序
    sorted_indices = np.argsort(time)
    sorted_time = time[sorted_indices]
    sorted_predictions = predictions[sorted_indices]
    sorted_true_values = true_values[sorted_indices]
    
    plt.plot(sorted_time, sorted_true_values, label="True Values")
    plt.plot(sorted_time, sorted_predictions, label="Predictions")
    plt.title(f'Cluster {cluster}: True vs Predictions')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{save_name}_predict.jpg"), dpi=500)
    plt.show()

for model_name, model_func in zip(['CNN', 'RNN', 'LSTM'], [build_cnn_model, build_rnn_model, build_lstm_model]): 
    # if model_name!='RNN':
    #     continue
    print('=' * 100)
    print(f"加载 {model_name} 模型...")
    print('=' * 100)
    
    # 数据划分
    X_train, X_test, y_train, y_test, cluster_train, cluster_test, time_train, time_test = train_test_split(
        X, y, clusters, times, test_size=0.2, random_state=42, shuffle=True
    )

    # 加载保存的模型
    model_save_path = os.path.join(save_dir, f"{exp_name}_{model_name}.h5")
    model = load_model(model_save_path)
    print("模型加载完成。开始测试...")

    # 预测
    print("进行预测...")
    start_time = time.time()
    predictions = model.predict(X_test)
    end_time = time.time()
    average_prediction_time = (end_time - start_time) / len(X_test)
    print(f"预测完成。每个样本的平均预测时间: {average_prediction_time:.6f} 秒")

    # 保存每个样本的预测时间
    time_save_path = os.path.join(results_dir, f'{exp_name}_{model_name}_prediction_time.txt')
    with open(time_save_path, 'w') as f:
        f.write(f"Average prediction time per sample: {average_prediction_time:.6f} seconds\n")
    print(f"每个样本的预测时间保存到: {time_save_path}")

    # 反归一化
    print("反归一化结果...")
    predictions = target_scaler.inverse_transform(predictions)[:,0]
    y_test = target_scaler.inverse_transform(y_test)[:,0]

    # 计算评估指标
    mse = mean_squared_error(y_test, predictions)
    rmse_value = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    metrics = {
        'MSE': mse,
        'R2': r2,
        'RMSE': rmse_value,
        'MAE': mae
    }
    eval_dic[model_name] = metrics.copy()
    print(f"MSE: {mse:.4f}, R²: {r2:.4f}, RMSE: {rmse_value:.4f}, MAE: {mae:.4f}")

    # 保存评估结果
    metrics_save_path = os.path.join(results_dir, f'{exp_name}_{model_name}_evaluation.txt')
    with open(metrics_save_path, 'w') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    print(f"评估结果保存到: {metrics_save_path}")

print("所有模型测试与评估完成。结果保存完毕。")

# %%
X_test.shape

# %%
# 数据划分
X_train, X_test, y_train, y_test, cluster_train, cluster_test, time_train, time_test = train_test_split(
    X, y, clusters, times, test_size=0.2, random_state=42, shuffle=False
)


# %%
last_time_of_data = time_test[-1][-1]
last_time_of_data

# %%
# %%
y_pred = model.predict(X_test)
y_pred_unscaled = target_scaler.inverse_transform(y_pred)
last_pred = y_pred_unscaled[-1]

import pandas as pd

index = pd.date_range(start=last_time_of_data + pd.Timedelta('5s'), freq='5s', periods=len(last_pred))
dl_pred = pd.Series(last_pred, index=index)


dt = df.copy()
dt[[target_column]] = target_scaler.inverse_transform(dt[[target_column]])
dd = final_features_df.set_index('MCGS_TIME_FULL')[target_column]

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MAModel:
    def __init__(self, lags=20):
        self.lags = lags
        self.fitted = False
        self.train_data = None
        self.ma_values = None
    
    def fit(self):
        if self.train_data is None:
            raise ValueError("No training data provided")
        
        n = len(self.train_data)
        if n <= self.lags:
            raise ValueError("Training data length must be greater than lags")
        
        self.ma_values = np.zeros(n)
        for i in range(self.lags, n):
            self.ma_values[i] = np.mean(self.train_data[i-self.lags:i])
        
        self.fitted = True
        return self
    
    def predict(self, start, end):
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        steps = end - start + 1
        last_ma = self.ma_values[-1]
        forecast = np.full(steps, last_ma)
        return forecast

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
train_seq = pd.concat([dd,dl_pred], axis=0).values

last_time = dd.index[-1]

forecast_steps = 1080 
ml_model = MAModel(lags=20)
ml_model.train_data = train_seq
fitted_model = ml_model.fit()

forecast = fitted_model.predict(start=len(train_seq), end=len(train_seq)+forecast_steps-1)

time_index = pd.date_range(start=last_time + pd.Timedelta('5s'), freq='5s', periods=forecast_steps)
ml_pred = pd.Series(forecast, index=time_index)

dl_pred.loc[ml_pred.index[0]] = ml_pred.iloc[0]
dl_pred.loc[dd.index[-1]] = dd.iloc[-1]
pred_res = pd.concat([dl_pred, ml_pred], axis=0).iloc[:1080]

plt.figure(figsize=(10,6))
plt.plot(dd, label='past')
plt.plot(dl_pred, label='next 10 points')
plt.plot(ml_pred, label='long future')
plt.legend()

plt.ylabel('target')
plt.xlabel('Datetime')
plt.tight_layout()
plt.savefig('future_predict.jpg', dpi=500)
plt.show()

# %%
import pandas as pd

# 基础参数设置
BASE_SOH = 1000  # 基础SOH值（百分比）

# 充放电额定值
CHARGE_NOMINAL = 0.95  # 充电额定值 (95%)
DISCHARGE_NOMINAL = 0.05  # 放电额定值 (5%)

# 预警参数配置
WARNING_PARAMS = {
    # 超出和低于
    3: {'a_charge': 1.11, 'a_discharge': 0.11},
    2: {'a_charge': 1.20, 'a_discharge': 0.20},
    1: {'a_charge': 1.29, 'a_discharge': 0.29}
}

def calculate_warning_levels(pred_res):
    # 创建DataFrame
    df = pd.DataFrame({'soh预测值': pred_res})
    
    # 计算SOH百分比
    df['soh_percentage'] = df['soh预测值'] / BASE_SOH
    
    # 初始化预警级别列（0表示无预警）
    df['warning_level'] = 0
    
    # 初始化故障类型列（0表示正常）
    df['fault_type'] = 0
    
    # 按照预警级别从高到低检查
    for level in sorted(WARNING_PARAMS.keys(), reverse=True):
        params = WARNING_PARAMS[level]
        
        # 计算充电和放电的预警阈值
        charge_threshold = CHARGE_NOMINAL * params['a_charge']
        discharge_threshold = DISCHARGE_NOMINAL * (1 - params['a_discharge'])
        
        # 更新预警级别
        warning_mask = (df['soh_percentage'] > charge_threshold) | \
                      (df['soh_percentage'] < discharge_threshold)
        
        # 仅更新尚未被更高级别预警标记的数据点
        df.loc[warning_mask & (df['warning_level'] == 0), 'warning_level'] = level
        
        # 同时更新故障类型
        # 过充故障（标记为1）
        df.loc[warning_mask & (df['warning_level'] == level) & 
               (df['soh_percentage'] > charge_threshold), 'fault_type'] = 1
        
        # 过放故障（标记为2）
        df.loc[warning_mask & (df['warning_level'] == level) & 
               (df['soh_percentage'] < discharge_threshold), 'fault_type'] = 2
    
    # 保存结果到Excel
    df.to_excel('soh_warning_results.xlsx', index=False)
    
    return df

# 使用示例
result_df = calculate_warning_levels(pred_res)
print(result_df['warning_level'].value_counts())
print(result_df['fault_type'].value_counts())

# %%
result_df

# %%
import pandas as pd

# 基础参数设置
BASE_SOC = 1000  # 基础SOC值
CHARGE_NOMINAL = 0.95  # 充电额定值 (95%)
DISCHARGE_NOMINAL = 0.05  # 放电额定值 (5%)

# 预警级别参数
WARNING_PARAMS = {
    # 超出和低于
    3: {'a_charge': 1.11, 'a_discharge': 0.11},
    2: {'a_charge': 1.20, 'a_discharge': 0.20},
    1: {'a_charge': 1.29, 'a_discharge': 0.29}
}

def calculate_warning_levels(pred_res):
    # 创建DataFrame
    df = pd.DataFrame({'SOH预测值': pred_res})
    
    # 计算SOC百分比
    df['soc_percentage'] = df['soc预测值'] / BASE_SOC
    
    # 初始化预警级别列（0表示无预警）
    df['warning_level'] = 0
    
    # 按照预警级别从高到低检查
    for level in sorted(WARNING_PARAMS.keys(), reverse=True):
        params = WARNING_PARAMS[level]
        
        # 计算充电和放电的预警阈值
        charge_threshold = CHARGE_NOMINAL * params['a_charge']
        discharge_threshold = DISCHARGE_NOMINAL * (1 - params['a_discharge'])
        
        # 更新预警级别
        warning_mask = (df['soc_percentage'] > charge_threshold) | \
                      (df['soc_percentage'] < discharge_threshold)
        
        # 仅更新尚未被更高级别预警标记的数据点
        df.loc[warning_mask & (df['warning_level'] == 0), 'warning_level'] = level
    
    # 保存结果到Excel
    df.to_excel('soc_warning_results.xlsx', index=False)
    
    return df


result_df = calculate_warning_levels(pred_res)

result_df['warning_level'].value_counts()

result_df

# %%
import matplotlib.pyplot as plt
import numpy as np
import os

# 绘制指标对比图并保存
def plot_metrics_comparison(metrics_dict):
    metrics = ['MAE', 'RMSE', 'MSE', 'R2']
    model_names = list(metrics_dict.keys())
    cmap = plt.get_cmap('tab20')  # 使用plt的cmap选择颜色
    
    for metric in metrics:
        # 获取对应指标的值并排序
        values = np.array([metrics_dict[model][metric] for model in model_names])
        sorted_indices = np.argsort(values)[::-1]  # 从高到低排序
        sorted_model_names = [model_names[i] for i in sorted_indices]
        sorted_values = values[sorted_indices]
        
        # 计算 ylim
        min_value = sorted_values.min()
        max_value = sorted_values.max()
        avg_value = sorted_values.mean()
        ylim = [0, max_value + (max_value - avg_value)]
        
        plt.figure(figsize=(10, 6))
        bars = plt.barh(sorted_model_names, sorted_values, color=cmap(range(len(model_names))))
        plt.xlim(ylim)
        plt.xlabel(metric.upper())
        plt.ylabel('Models')
        plt.title(f'Comparison of {metric.upper()} for Different Models')

        # 在每个条形图上标记具体数值
        for bar in bars:
            xval = bar.get_width()
            plt.text(xval, bar.get_y() + bar.get_height()/2, f'{xval:.4f}', va='center', ha='left')
        
        plt.tight_layout()
        save_path = os.path.join(results_dir, f'{metric}_comparison.png')
        plt.savefig(save_path, dpi=500)
        print(f"{metric} 对比图保存到: {save_path}")
        plt.show()

# 调用函数绘制并保存图形
plot_metrics_comparison(eval_dic)


# %%



