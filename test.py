# %%
# load packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
import platform
if platform.system() == "Windows":
    plt.rcParams['font.family'] = ['SimHei'] # Windows
elif platform.system() == "Darwin":
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # MacOS
plt.rcParams['axes.unicode_minus']=False 
# 加载文件夹中的数据
folder_path = './Dataset/1. BatteryAgingARC-FY08Q4'

# 预测步长
predict_step = 5

# %%
import os
import glob
import datetime
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

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


# %%
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Conv1D, Bidirectional, LSTM, Dropout, Input,
    MultiHeadAttention, LayerNormalization, GlobalMaxPooling1D,MaxPooling1D
)

def transformer_encoder(inputs, embed_dim=32, num_heads=2, ff_dim=32, rate=0.1):
    # 调整输入维度以匹配嵌入维度
    x = Dense(embed_dim)(inputs)

    # 多头自注意力层
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    attention_output = Dropout(rate)(attention_output)
    out1 = LayerNormalization(epsilon=1e-6)(attention_output + x)

    # 前馈网络
    ffn_output = Dense(ff_dim, activation="relu")(out1)
    ffn_output = Dense(embed_dim)(ffn_output)
    ffn_output = Dropout(rate)(ffn_output)
    return LayerNormalization(epsilon=1e-6)(ffn_output + out1)


def create_transformer_cnn_bilstm_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # Transformer Encoder
    x = transformer_encoder(inputs)

    # CNN Layer
    x = Conv1D(filters=16, kernel_size=2, padding="same", activation="relu")(x)
    # BiLSTM Layer
    x = Bidirectional(LSTM(32, return_sequences=False))(x)
    x = Dropout(0.5)(x)

    # Output Layer
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


# %%
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Conv1D, SimpleRNN, GlobalAveragePooling1D, Dense, GlobalAvgPool1D, Dropout, MaxPooling1D, Bidirectional, Input, Flatten

# 设置全局随机数种子以确保可重复性
tf.random.set_seed(42)
np.random.seed(42)

# 定义GRU模型方法
def create_gru_model(input_shape):
    model = Sequential()
    model.add(GRU(16, input_shape=input_shape))  # GRU层：16个单元，输入形状为input_shape
    model.add(Dropout(0.5))  # Dropout层：防止过拟合，dropout比率为0.5
    model.add(Dense(1, activation='sigmoid'))  # 输出层：1个神经元，使用sigmoid激活函数
    return model

# 定义CNN模型方法
def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(16, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))  # Conv1D层：16个滤波器，核大小为3，使用ReLU激活函数，padding方式为'same'
    model.add(GlobalAvgPool1D())  # 全局平均池化层
    model.add(Dropout(0.5))  # Dropout层：dropout比率为0.5
    model.add(Dense(1, activation='sigmoid'))  # 输出层：1个神经元，使用sigmoid激活函数
    return model

# 定义RNN模型方法
def create_rnn_model(input_shape):
    model = Sequential()
    model.add(SimpleRNN(16, input_shape=input_shape))  # SimpleRNN层：16个单元，输入形状为input_shape
    model.add(Dropout(0.5))  # Dropout层：dropout比率为0.5
    model.add(Dense(1, activation='sigmoid'))  # 输出层：1个神经元，使用sigmoid激活函数
    return model

# 定义LSTM模型方法
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(16, input_shape=input_shape))  # LSTM层：16个单元，输入形状为input_shape
    model.add(Dropout(0.5))  # Dropout层：dropout比率为0.5
    model.add(Dense(1, activation='sigmoid'))  # 输出层：1个神经元，使用sigmoid激活函数
    return model

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Conv1D, Bidirectional, LSTM, Dropout, Input,
    MultiHeadAttention, LayerNormalization, GlobalMaxPooling1D, Flatten
)

# Transformer编码器层
def transformer_encoder(inputs, embed_dim=32, num_heads=2, ff_dim=32, rate=0.1):
    x = Dense(embed_dim)(inputs)  # 嵌入层：将输入转换为embed_dim维的向量

    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)  # 多头自注意力：2个头，注意力维度为embed_dim
    attention_output = Dropout(rate)(attention_output)  # Dropout层：dropout比率为0.1
    out1 = LayerNormalization(epsilon=1e-10)(attention_output + x)  # 残差连接和LayerNorm

    ffn_output = Dense(ff_dim, activation="relu")(out1)  # 前馈网络：隐藏层维度为ff_dim，使用ReLU激活函数
    ffn_output = Dense(embed_dim)(ffn_output)  # 输出维度回到embed_dim
    ffn_output = Dropout(rate)(ffn_output)  # Dropout层：dropout比率为0.1
    return LayerNormalization(epsilon=1e-6)(ffn_output + out1)  # 最后的LayerNorm和残差连接

# 定义Transformer模型方法
def create_transformer_model(input_shape):
    inputs = Input(shape=input_shape)  # 输入层
    x = transformer_encoder(inputs)  # Transformer编码器
    x = Flatten()(x)  # 展平操作，将高维数据转化为一维
    x = Dropout(0.9)(x)  # Dropout层：dropout比率为0.9
    outputs = Dense(1, activation='sigmoid')(x)  # 输出层：1个神经元，使用sigmoid激活函数
    model = Model(inputs=inputs, outputs=outputs)  # 构建模型
    return model

# 定义CNN-BiLSTM模型方法
def create_cnn_bilstm_model(input_shape):
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))  # Conv1D层：32个滤波器，核大小为3，使用ReLU激活函数
    model.add(MaxPooling1D(pool_size=2))  # 最大池化层：池化大小为2
    model.add(Bidirectional(LSTM(32, return_sequences=False)))  # 双向LSTM层：32个单元，不返回序列
    model.add(Dropout(0.5))  # Dropout层：dropout比率为0.5
    model.add(Dense(1, activation='sigmoid'))  # 输出层：1个神经元，使用sigmoid激活函数
    return model

# 主方法：根据模型名称返回对应模型
def get_model(model_name, input_shape):
    if model_name == 'RNN':
        return create_rnn_model(input_shape)
    elif model_name == 'LSTM':
        return create_rnn_model(input_shape)
    elif model_name == 'GRU':
        return create_gru_model(input_shape)
    elif model_name == 'CNN':
        return create_cnn_model(input_shape)
    elif model_name == 'CNN-BiLSTM':
        return create_cnn_bilstm_model(input_shape)
    elif model_name == 'Transformer':
        return create_transformer_model(input_shape)
    elif model_name == 'Transformer-CNN-BiLSTM':
        return create_transformer_cnn_bilstm_model(input_shape)
    else:
        raise ValueError(f"Invalid model name: {model_name}")


# %%
import os
import glob
import datetime
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns


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

# 构建时间序列数据
def construct_time_series_data(data, window_size, predict_step=1):
    X = []
    y = []
    battery_label = []
    cycle_labels = []
    
    for battery_name, df in data.items():
        # 计算SOH
        attrib = ['cycle', 'datetime', 'capacity']
        dis_ele = df[attrib].copy()
        C = dis_ele['capacity'][0]
        dis_ele['SoH'] = dis_ele['capacity']

        # 按cycle分组并计算平均值
        df_grouped = df.groupby('cycle').mean().reset_index()

        # 确保所有需要的列都存在
        required_columns = ['voltage_measured', 'current_measured', 'temperature_measured', 'current_load', 'voltage_load', 'time']
        if not all(col in df_grouped.columns for col in required_columns):
            print(f"Warning: Missing columns in battery {battery_name}. Skipping this battery.")
            continue
        
        # 应用卡尔曼滤波器
        df_grouped, all_features = apply_kalman_filter(df_grouped, required_columns)

        features = df_grouped[all_features].values  # 使用原始和卡尔曼滤波后的特征
        soh_values = dis_ele.groupby('cycle').mean()['SoH'].values
        cycle_values = df_grouped['cycle'].values
        
        for i in range(len(soh_values) - window_size - predict_step + 1):
            feature_window = np.hstack([features[i:i+window_size], soh_values[i:i+window_size].reshape(-1, 1)])
            X.append(feature_window)
            y.append(soh_values[i+window_size + predict_step - 1])  # 预测未来第predict_step个值
            battery_label.append(battery_name)
            cycle_labels.append(cycle_values[i+window_size + predict_step - 1])  # 对应的cycle标签改为未来predict_step的cycle值
    
    X = np.array(X)
    y = np.array(y)
    battery_label = np.array(battery_label)
    cycle_labels = np.array(cycle_labels)
    
    return X, y, battery_label, cycle_labels



mat_files = glob.glob(os.path.join(folder_path, '**', '*.mat'), recursive=True)

data = {}
for mat_file in mat_files:
    battery_name = os.path.basename(mat_file).split('.')[0]
    try:
        dataset, _ = load_data(mat_file, battery_name)
    except:
        # print(f"Error: {mat_file}")
        pass
    data[battery_name] = dataset 

# 构建时间序列数据，指定窗口大小和预测步长
window_size = 4
X, y, battery_label, cycle_labels = construct_time_series_data(data, window_size, predict_step=predict_step)

# 定义归一化和反归一化的函数
scaler_X = MinMaxScaler()  # 用于特征数据的归一化
scaler_y = MinMaxScaler()  # 用于目标数据的归一化

# 归一化函数
def normalize_data(X, y):
    X_reshaped = X.reshape(-1, X.shape[-1])  # 将X重塑为2D数组以适应scaler
    X_normalized = scaler_X.fit_transform(X_reshaped).reshape(X.shape)  # 对X进行归一化并恢复原形状
    
    y_reshaped = y.reshape(-1, 1)  # 将y重塑为2D数组以适应scaler
    y_normalized = scaler_y.fit_transform(y_reshaped).reshape(y.shape)  # 对y进行归一化并恢复原形状
    
    return X_normalized, y_normalized

# 反归一化函数
def denormalize_data(y_normalized):
    y_reshaped = y_normalized.reshape(-1, 1)  # 将y_normalized重塑为2D数组
    y_denormalized = scaler_y.inverse_transform(y_reshaped).reshape(y_normalized.shape)  # 反归一化并恢复原形状
    return y_denormalized

# 归一化特征和目标数据
X_normalized, y_normalized = normalize_data(X, y)

# 设置测试电池标签，用于划分训练集和测试集
test_label = 'B0005'  # 指定测试电池

# 划分训练集和测试集
X_train = X_normalized[battery_label != test_label]  # 训练集为不属于测试电池的数据
y_train = y_normalized[battery_label != test_label]
X_test = X_normalized[battery_label  == test_label]  # 测试集为属于指定测试电池的数据
y_test = y_normalized[battery_label == test_label]
cycle_labels_test = cycle_labels[battery_label == test_label]  # 测试集的cycle标签


# %%
exp_name = f"soh_step={predict_step}"
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import random
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# 设定全局随机数种子
set_seed(0)

# %%
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from tensorflow.keras.models import load_model
import warnings

warnings.filterwarnings("ignore")

# 确保保存目录存在
results_dir = "./results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# 指标计算函数
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = np.sqrt(mean_absolute_error(y_true, y_pred) ** 2)
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'MAPE': mape, 'RMSE': rmse, 'R2': r2}

# 绘制 SOH 真实值和预测值对比图
def plot_soh_comparison(model_name, cycle_labels_test, y_test_soh, y_pred_soh):
    cycle_df = pd.DataFrame({'cycle': cycle_labels_test, 'true_soh': y_test_soh, 'predicted_soh': y_pred_soh})
    cycle_grouped = cycle_df.groupby('cycle').mean()

    plt.figure(figsize=(10, 6))
    plt.plot(cycle_grouped.index, cycle_grouped['true_soh'], label='True SOH')
    plt.plot(cycle_grouped.index, cycle_grouped['predicted_soh'], label='Predicted SOH')
    plt.xlabel('Cycle')
    plt.ylabel('SOH')
    plt.legend()
    plt.title(f'{model_name} - True SOH vs Predicted SOH')
    plt.tight_layout()
    save_path = os.path.join(results_dir, f"{model_name}_soh_plot.png")
    plt.savefig(save_path, dpi=500)
    plt.close()
    print(f"SOH对比图已保存到: {save_path}")

# 测试模型函数
def test_model(model_name, X_test, y_test, cycle_labels_test, exp_name, save_dir, results_dir):
    # 加载模型
    model_path = os.path.join(save_dir, f"{exp_name}_{model_name}.h5")
    print(f"加载模型: {model_path}")
    model = load_model(model_path)
    print(f"模型 {model_name} 加载完成。")

    # 预测
    print("进行预测...")
    start_time = time.time()
    y_pred_normalized = model.predict(X_test)
    end_time = time.time()
    avg_prediction_time = (end_time - start_time) / len(X_test)
    print(f"预测完成，平均预测时间: {avg_prediction_time:.6f} 秒/样本")

    # 保存预测时间
    time_save_path = os.path.join(results_dir, f"{exp_name}_{model_name}_prediction_time.txt")
    with open(time_save_path, 'w') as f:
        f.write(f"Average prediction time per sample: {avg_prediction_time:.6f} seconds\n")
    print(f"预测时间保存到: {time_save_path}")

    # 反归一化
    y_pred = denormalize_data(y_pred_normalized)
    y_test_denormalized = denormalize_data(y_test)

    # 转换为 SOH
    C_max = data[test_label]['capacity'].max()
    y_test_soh = y_test_denormalized.flatten() / C_max
    y_pred_soh = y_pred.flatten() / C_max

    # 绘图
    plot_soh_comparison(model_name, cycle_labels_test, y_test_soh, y_pred_soh)

    # 计算指标
    metrics = calculate_metrics(y_test_soh, y_pred_soh)

    # 保存评估结果
    metrics_save_path = os.path.join(results_dir, f"{exp_name}_{model_name}_evaluation.txt")
    with open(metrics_save_path, 'w') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    print(f"评估结果保存到: {metrics_save_path}")

    return metrics

# 测试所有模型
print("开始测试模型...")
exp_name = f"soh_step={predict_step}"
metrics_dict = {}
model_names = ['CNN', 'LSTM', 'RNN', 'GRU', 'CNN-BiLSTM', 'Transformer', 'Transformer-CNN-BiLSTM']
save_dir = "./models"

for model_name in model_names:
    metrics = test_model(
        model_name, X_test, y_test, cycle_labels_test, exp_name, save_dir, results_dir
    )
    metrics_dict[model_name] = metrics

# 打印最终结果
print("所有模型测试完成，结果如下：")
for model, metrics in metrics_dict.items():
    print(f"{model}: {metrics}")


# %%
import os
import matplotlib.pyplot as plt
import numpy as np

# 确保保存目录存在
results_dir = "./results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# 绘制指标对比图并保存
def plot_metrics_comparison(metrics_dict):
    metrics = ['mae', 'rmse', 'mape', 'r2']
    model_names = list(metrics_dict.keys())
    cmap = plt.get_cmap('tab10')  # 使用plt的cmap选择颜色
    
    for metric in metrics:
        # 获取对应指标的值并排序
        values = np.array([metrics_dict[model][metric.upper()] for model in model_names])
        sorted_indices = np.argsort(values)[::]  # 从高到低排序
        sorted_model_names = [model_names[i] for i in sorted_indices]
        sorted_values = values[sorted_indices]
        
        # 计算ylim
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
        
        # 保存图形
        save_path = os.path.join(results_dir, f"{metric}_comparison.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=500)
        plt.close()
        print(f"{metric.upper()} 对比图保存到: {save_path}")

# 调用函数绘制并保存图形
plot_metrics_comparison(metrics_dict)


# %%



