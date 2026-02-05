import os
import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# 加载 .mat 文件数据
def load_data(file_path, battery):
    mat = loadmat(file_path)  # 加载 .mat 文件
    counter = 0  # 计数器
    dataset = []  # 存储每个样本的数据
    capacity_data = []  # 存储每个周期的容量数据
    
    # 遍历电池数据中的每个周期
    for i in range(len(mat[battery][0, 0]['cycle'][0])):
        row = mat[battery][0, 0]['cycle'][0, i]
        if row['type'][0] == 'discharge':  # 只处理放电周期
            ambient_temperature = row['ambient_temperature'][0][0]
            date_time = datetime.datetime(int(row['time'][0][0]),
                                    int(row['time'][0][1]),
                                    int(row['time'][0][2]),
                                    int(row['time'][0][3]),
                                    int(row['time'][0][4])) + datetime.timedelta(seconds=int(row['time'][0][5]))
            data = row['data']
            capacity = data[0][0]['Capacity'][0][0]
            
            # 遍历每个数据点，获取电压、电流、温度等测量值
            for j in range(len(data[0][0]['Voltage_measured'][0])):
                voltage_measured = data[0][0]['Voltage_measured'][0][j]
                current_measured = data[0][0]['Current_measured'][0][j]
                temperature_measured = data[0][0]['Temperature_measured'][0][j]
                current_load = data[0][0]['Current_load'][0][j]
                voltage_load = data[0][0]['Voltage_load'][0][j]
                time = data[0][0]['Time'][0][j]
                dataset.append([counter + 1, ambient_temperature, date_time, capacity,
                                voltage_measured, current_measured,
                                temperature_measured, current_load,
                                voltage_load, time])
            capacity_data.append([counter + 1, ambient_temperature, date_time, capacity])
            counter += 1  # 更新计数器

    return [pd.DataFrame(data=dataset,  # 返回电池数据和容量数据的 DataFrame
                        columns=['cycle', 'ambient_temperature', 'datetime',
                                'capacity', 'voltage_measured',
                                'current_measured', 'temperature_measured',
                                'current_load', 'voltage_load', 'time']),
            pd.DataFrame(data=capacity_data,
                        columns=['cycle', 'ambient_temperature', 'datetime',
                                'capacity'])]

# 可视化电池健康状态 (SOH)
def visualize_soh(capacity_df, title):
    attrib = ['cycle', 'datetime', 'capacity']
    dis_ele = capacity_df[attrib]
    C = dis_ele['capacity'][0]
    dis_ele['SoH'] = dis_ele['capacity'] / C  # 计算SOH
    
    plot_df = dis_ele.loc[(dis_ele['cycle'] >= 1), ['cycle', 'SoH']]  # 筛选周期数据
    sns.set_style("white")
    plt.figure(figsize=(8, 5))
    plt.plot(plot_df['cycle'], plot_df['SoH'])  # 绘制SOH随周期变化的曲线
    plt.ylabel('SOH')
    plt.xlabel('cycle')
    plt.title(title)
    plt.close()

# 示例：加载并可视化文件夹中的电池数据
folder_path = './1. BatteryAgingARC-FY08Q4'
mat_files = glob.glob(os.path.join(folder_path, '*.mat'))

# 遍历每个 .mat 文件
for mat_file in mat_files:
    battery_name = os.path.basename(mat_file).split('.')[0]  # 提取电池名称
    dataset, capacity = load_data(mat_file, battery_name)  # 加载电池数据
    visualize_soh(capacity, f'Discharge {battery_name}')  # 可视化SOH


# 应用卡尔曼滤波器
def kalman_filter_fuc(observations, process_variance=1e-4, measurement_variance=0.5**2, initial_estimate=0.0, initial_uncertainty=1.0):
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

# 在数据上应用卡尔曼滤波器，生成额外特征
def apply_kalman_filter(df, features):
    for feature in features:
        df[feature + '_kalman'] = kalman_filter_fuc(df[feature].values)
    all_features = features + [f + '_kalman' for f in features]
    return df, all_features

# 加载所有数据文件
def load_all_data(folder_path, test_battery='B0005'):
    mat_files = glob.glob(os.path.join(folder_path, '*.mat'))
    train_data, test_data = [], []
    train_capacity, test_capacity = [], []

    for mat_file in mat_files:
        battery_name = os.path.basename(mat_file).split('.')[0]
        data, capacity = load_data(mat_file, battery_name)

        if battery_name == test_battery:
            test_data.append(data)
            test_capacity.append(capacity)
        else:
            train_data.append(data)
            train_capacity.append(capacity)

    return np.vstack(train_data), np.hstack(train_capacity), np.vstack(test_data), np.hstack(test_capacity)

# 归一化数据
def normalize_data(train_data, train_capacity, test_data, test_capacity):
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    train_data = scaler_X.fit_transform(train_data)
    train_capacity = scaler_y.fit_transform(train_capacity.reshape(-1, 1)).flatten()
    test_data = scaler_X.transform(test_data)
    test_capacity = scaler_y.transform(test_capacity.reshape(-1, 1)).flatten()

    return train_data, train_capacity, test_data, test_capacity, scaler_X, scaler_y

# 构建时间序列数据
def construct_time_series_data(data, window_size, features):
    data_df = pd.DataFrame(data, columns=features)
    data_df, all_features = apply_kalman_filter(data_df, features)
    data = data_df[all_features].values

    X = []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
    return np.array(X)

# 测试保存的模型
def test_saved_model(model_name, test_data, test_capacity, scaler_y):
    model_path = f'./{model_name}_trained_model.h5'
    model = load_model(model_path)

    y_pred_normalized = model.predict(test_data)
    y_pred = scaler_y.inverse_transform(y_pred_normalized)

    # 反归一化测试容量
    y_test = scaler_y.inverse_transform(test_capacity.reshape(-1, 1))

    # 计算评估指标
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print(f"Model: {model_name}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.4f}")

    # 绘制实际值与预测值的比较图
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='True SOH')
    plt.plot(y_pred, label='Predicted SOH')
    plt.xlabel('Cycle')
    plt.ylabel('SOH')
    plt.title(f'{model_name} - True vs Predicted SOH')
    plt.legend()
    plt.show()

# 主程序
folder_path = './Dataset/1. BatteryAgingARC-FY08Q4'  # 数据文件夹路径
train_data, train_capacity, test_data, test_capacity = load_all_data(folder_path)

# 归一化数据
train_data, train_capacity, test_data, test_capacity, scaler_X, scaler_y = normalize_data(train_data, train_capacity, test_data, test_capacity)

# 构建时间序列数据，确保形状为 (样本数, 时间步, 特征数)
window_size = 4
features = ['voltage_measured', 'current_measured', 'temperature_measured', 'current_load', 'voltage_load']
test_data_reshaped = construct_time_series_data(test_data, window_size, features)

# 测试保存的模型
model_name = 'Transformer-CNN-BiLSTM'  # 替换为需要测试的模型名称
test_saved_model(model_name, test_data_reshaped, test_capacity[window_size:], scaler_y)
