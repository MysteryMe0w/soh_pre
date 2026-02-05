import os
import sys
import numpy as np
import tensorflow as tf
import json
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Conv1D,
    Bidirectional,
    LSTM,
    Dropout,
    Input,
    MultiHeadAttention,
    LayerNormalization,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 导入改进的PSO优化器
from pso_optimizer_improved_v1 import ImprovedPSOOptimizerV1, FastPSOOptimizer

# 导入原始train.py中的数据
from train import X_normalized, y_normalized, battery_label, test_label

import warnings

warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """设置随机种子"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def create_optimized_model(input_shape, params):
    """根据PSO优化后的参数创建模型"""
    inputs = Input(shape=input_shape)

    # Transformer Encoder
    x = Dense(params["embed_dim"])(inputs)

    attention_output = MultiHeadAttention(
        num_heads=params["num_heads"],
        key_dim=params["embed_dim"] // params["num_heads"],
    )(x, x)
    attention_output = Dropout(params["transformer_dropout"])(attention_output)
    out1 = LayerNormalization(epsilon=1e-6)(attention_output + x)

    ffn_output = Dense(params["ff_dim"], activation="relu")(out1)
    ffn_output = Dense(params["embed_dim"])(ffn_output)
    ffn_output = Dropout(params["transformer_dropout"])(ffn_output)
    x = LayerNormalization(epsilon=1e-6)(ffn_output + out1)

    # CNN Layer
    x = Conv1D(
        filters=params["cnn_filters"],
        kernel_size=params["cnn_kernel_size"],
        padding="same",
        activation="relu",
    )(x)

    # BiLSTM Layer
    x = Bidirectional(LSTM(params["lstm_units"], return_sequences=False))(x)
    x = Dropout(params["final_dropout"])(x)

    # Output Layer
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def run_pso_optimization(run_id=1, seed=42, fast_mode=False):
    """运行PSO超参数优化"""

    set_seed(seed)

    print("\n" + "=" * 80)
    print(f"🚀 PSO 优化 - Run #{run_id} (Seed: {seed})")
    if fast_mode:
        print("   模式: 快速模式 (预计 30-60分钟)")
    else:
        print("   模式: 标准模式 (预计 1.5-2小时)")
    print("=" * 80)

    # 准备数据
    X_train = X_normalized[battery_label != test_label]
    y_train = y_normalized[battery_label != test_label]
    X_test = X_normalized[battery_label == test_label]
    y_test = y_normalized[battery_label == test_label]

    # 从训练集中划分验证集
    X_t, X_val, y_t, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.15,  # 0.2 → 0.15（减少验证集大小加速）
        random_state=seed,
    )

    print(f"\n📊 数据集划分:")
    print(f"   训练集: {X_t.shape}")
    print(f"   验证集: {X_val.shape}")
    print(f"   测试集: {X_test.shape}")

    # 创建保存目录
    save_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), f"models/pso_optimized/run_{run_id}"
    )
    os.makedirs(save_dir, exist_ok=True)

    # 创建PSO优化器
    start_time = time.time()

    if fast_mode:
        print("\n⚡ 使用快速PSO配置...")
        optimizer = FastPSOOptimizer(X_t, y_t, X_val, y_val, save_dir=save_dir)
    else:
        print("\n🎯 使用标准PSO配置...")
        optimizer = ImprovedPSOOptimizerV1(
            X_t,
            y_t,
            X_val,
            y_val,
            n_particles=12,
            max_iter=20,
            train_epochs=50,
            patience=8,
            save_dir=save_dir,
            use_multi_gpu=True,
        )

    # 执行优化
    print("\n" + "=" * 80)
    print("开始PSO优化...")
    print("=" * 80)

    best_params, best_mse_val, best_metrics = optimizer.optimize()

    pso_time = time.time() - start_time
    print(f"\n⏱️  PSO优化完成，耗时: {pso_time/60:.2f} 分钟")

    # 绘制收敛曲线
    optimizer.plot_convergence(os.path.join(save_dir, "convergence.png"))

    # 用最优参数在完整训练集上重新训练
    print("\n" + "=" * 80)
    print("📈 使用最优参数在完整训练集上训练最终模型...")
    print("=" * 80)

    input_shape = (X_train.shape[1], X_train.shape[2])
    final_model = create_optimized_model(input_shape, best_params)

    final_model.compile(
        optimizer=Adam(learning_rate=best_params["learning_rate"]),
        loss="mse",
        metrics=["mae"],
    )

    callbacks = [
        EarlyStopping(
            monitor="loss", patience=30, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=10, min_lr=1e-8, verbose=1
        ),
    ]

    final_train_start = time.time()
    history = final_model.fit(
        X_train, y_train, epochs=200, batch_size=64, callbacks=callbacks, verbose=1
    )
    final_train_time = time.time() - final_train_start

    # 在测试集上评估
    print("\n" + "=" * 80)
    print("📊 在测试集上评估...")
    print("=" * 80)

    y_pred_test = final_model.predict(X_test, verbose=0)

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    test_mse = mean_squared_error(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_pred_test)

    # 保存模型
    model_path = os.path.join(save_dir, "optimized_model.h5")
    final_model.save(model_path)

    # 保存预测结果
    np.save(os.path.join(save_dir, "y_test.npy"), y_test)
    np.save(os.path.join(save_dir, "y_pred.npy"), y_pred_test)

    # 保存完整指标
    results = {
        "run_id": run_id,
        "seed": seed,
        "fast_mode": fast_mode,
        "best_params": best_params,
        "pso_optimization_time_minutes": pso_time / 60,
        "final_training_time_seconds": final_train_time,
        "total_time_minutes": (pso_time + final_train_time) / 60,
        "validation_metrics": {
            "mse": float(best_mse_val),
            "mae": float(best_metrics["mae"]),
            "rmse": float(best_metrics["rmse"]),
            "r2": float(best_metrics["r2"]),
        },
        "test_metrics": {
            "mse": float(test_mse),
            "mae": float(test_mae),
            "rmse": float(test_rmse),
            "r2": float(test_r2),
        },
        "epochs_trained": len(history.history["loss"]),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(os.path.join(save_dir, "pso_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # 打印最终结果
    print("\n" + "=" * 80)
    print("✅ PSO优化模型训练完成!")
    print("=" * 80)
    print(f"\n📈 测试集性能:")
    print(f"   MSE:  {test_mse:.6f}")
    print(f"   MAE:  {test_mae:.6f}")
    print(f"   RMSE: {test_rmse:.6f}")
    print(f"   R²:   {test_r2:.4f}")
    print(f"\n⏱️  时间统计:")
    print(f"   PSO优化: {pso_time/60:.2f} 分钟")
    print(f"   最终训练: {final_train_time:.2f} 秒")
    print(f"   总耗时:   {(pso_time + final_train_time)/60:.2f} 分钟")
    print(f"\n💾 结果已保存至: {save_dir}")
    print("=" * 80 + "\n")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PSO超参数优化训练")
    parser.add_argument("--run_id", type=int, default=1, help="运行编号")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--fast", action="store_true", help="使用快速模式")

    args = parser.parse_args()

    results = run_pso_optimization(
        run_id=args.run_id, seed=args.seed, fast_mode=args.fast
    )
