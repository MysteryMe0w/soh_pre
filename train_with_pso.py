import os
import sys
import json
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

# ── 导入PSO优化器（仅从 pso_optimizer.py 导入，忽略旧的 pso_optimizer_fast.py）
from pso_optimizer import FastPSOOptimizer

# ── 从 train.py 导入已处理好的数据（避免重复加载）
from train import X_normalized, y_normalized, battery_label, test_label

import warnings
import logging

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────


def set_seed(seed: int = 42):
    """设置全局随机种子，保证可复现"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def snap_embed_dim(embed_dim: int, num_heads: int) -> int:
    """
    将 embed_dim 向下对齐到能被 num_heads 整除的最近值。
    pso_optimizer.py 的 _decode_particle 已经做了这个处理，
    这里在最终模型构建时再做一次双重保险。
    """
    aligned = (embed_dim // num_heads) * num_heads
    return max(aligned, num_heads)


# ─────────────────────────────────────────────────────────────────
# 最终模型构建
# （结构与 train.py 中的 baseline create_transformer_cnn_bilstm_model 完全对齐）
# ─────────────────────────────────────────────────────────────────


def create_optimized_model(input_shape: tuple, params: dict) -> Model:
    """
    根据 PSO 搜索到的超参数构建 Transformer-CNN-BiLSTM 模型。

    结构（单层，与 baseline 对齐，公平对比）：
        Input
          → Dense(embed_dim)                        # 嵌入投影
          → MultiHeadAttention + Dropout + LN       # 自注意力 + 残差
          → FFN(ff_dim) + Dropout + LN              # 前馈网络 + 残差
          → Conv1D(cnn_filters, cnn_kernel_size)    # 局部特征提取
          → BiLSTM(lstm_units)                      # 时序建模
          → Dropout(final_dropout)
          → Dense(1, sigmoid)                       # SOH 回归输出
    """
    # ── 类型强转 + 整除对齐 ───────────────────────────────────────
    num_heads = int(params["num_heads"])
    embed_dim = snap_embed_dim(int(params["embed_dim"]), num_heads)
    ff_dim = int(params["ff_dim"])
    cnn_filters = int(params["cnn_filters"])
    cnn_kernel_size = int(params["cnn_kernel_size"])
    lstm_units = int(params["lstm_units"])
    t_drop = float(params["transformer_dropout"])
    f_drop = float(params["final_dropout"])
    lr = float(params["learning_rate"])

    inputs = Input(shape=input_shape)

    # ── Transformer Encoder ───────────────────────────────────────
    x = Dense(embed_dim)(inputs)

    attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(
        x, x
    )
    attn_out = Dropout(t_drop)(attn_out)
    x = LayerNormalization(epsilon=1e-6)(attn_out + x)

    ffn_out = Dense(ff_dim, activation="relu")(x)
    ffn_out = Dense(embed_dim)(ffn_out)
    ffn_out = Dropout(t_drop)(ffn_out)
    x = LayerNormalization(epsilon=1e-6)(ffn_out + x)

    # ── CNN ───────────────────────────────────────────────────────
    x = Conv1D(
        filters=cnn_filters,
        kernel_size=cnn_kernel_size,
        padding="same",
        activation="relu",
    )(x)

    # ── BiLSTM ────────────────────────────────────────────────────
    x = Bidirectional(LSTM(lstm_units, return_sequences=False))(x)
    x = Dropout(f_drop)(x)

    # ── 输出层 ───────────────────────────────────────────────────
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse", metrics=["mae"])
    return model


# ─────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────


def run_pso_optimization(run_id: int = 1, seed: int = 42) -> dict:

    set_seed(seed)

    logger.info("=" * 70)
    logger.info(f"PSO 优化 - Run #{run_id}  (Seed: {seed})")
    logger.info("=" * 70)

    # ── 数据划分 ──────────────────────────────────────────────────
    X_train_full = X_normalized[battery_label != test_label]
    y_train_full = y_normalized[battery_label != test_label]
    X_test = X_normalized[battery_label == test_label]
    y_test = y_normalized[battery_label == test_label]

    # 从完整训练集中切出验证集，供 PSO 内部评估（不碰测试集）
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=seed
    )

    logger.info(
        f"数据集大小 → 训练(PSO内): {X_tr.shape}  "
        f"验证: {X_val.shape}  测试: {X_test.shape}"
    )

    # ── 保存目录 ──────────────────────────────────────────────────
    save_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), f"models/pso_optimized/run_{run_id}"
    )
    os.makedirs(save_dir, exist_ok=True)

    # ── PSO 优化 ─────────────────────────────────────────────────
    # 使用 FastPSOOptimizer（来自 pso_optimizer.py）
    # 内部已固定: n_particles=8, max_iter=15, train_epochs=30, patience=5
    # 预估评估次数: 8×15 = 120次，约 30-60 分钟
    pso_start = time.time()

    optimizer = FastPSOOptimizer(
        X_tr,
        y_tr,
        X_val,
        y_val,
        save_dir=save_dir,
        # 如需覆盖参数，可在此传入，例如：
        # n_particles=10, max_iter=20
    )

    logger.info("\n开始 PSO 优化...")
    best_params, best_mse_val, best_metrics = optimizer.optimize()

    pso_elapsed = time.time() - pso_start

    logger.info(f"\nPSO 完成，耗时: {pso_elapsed/60:.2f} 分钟")
    logger.info(f"验证集最优 MSE : {best_mse_val:.6f}")
    logger.info(f"最优超参数     :\n{json.dumps(best_params, indent=4)}")

    # 收敛曲线
    optimizer.plot_convergence(save_path=os.path.join(save_dir, "convergence.png"))

    # ── 用最优参数在完整训练集上重新训练最终模型 ─────────────────
    logger.info("\n" + "=" * 70)
    logger.info("使用最优参数在完整训练集上训练最终模型...")
    logger.info("=" * 70)

    input_shape = (X_train_full.shape[1], X_train_full.shape[2])
    final_model = create_optimized_model(input_shape, best_params)
    final_model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=50, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10, min_lr=1e-8, verbose=1
        ),
    ]

    final_start = time.time()
    history = final_model.fit(
        X_train_full,
        y_train_full,
        epochs=500,  # 增大上限，配合EarlyStopping自动停
        batch_size=16,  # 与 baseline 训练一致
        validation_split=0.15,  # 留 15% 用于早停监控
        callbacks=callbacks,
        verbose=1,
    )
    final_elapsed = time.time() - final_start

    # ── 测试集评估 ────────────────────────────────────────────────
    y_pred_test = final_model.predict(X_test, verbose=0)

    test_mse = mean_squared_error(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_pred_test)

    # ── 保存产物 ──────────────────────────────────────────────────
    final_model.save(os.path.join(save_dir, "optimized_model.h5"))
    np.save(os.path.join(save_dir, "y_test.npy"), y_test)
    np.save(os.path.join(save_dir, "y_pred.npy"), y_pred_test)

    results = {
        "run_id": run_id,
        "seed": seed,
        "best_params": best_params,
        "pso_optimization_time_minutes": round(pso_elapsed / 60, 4),
        "final_training_time_seconds": round(final_elapsed, 2),
        "total_time_minutes": round((pso_elapsed + final_elapsed) / 60, 4),
        "validation_metrics": {
            "mse": float(best_mse_val),
            "mae": float(best_metrics.get("mae", float("nan"))),
            "rmse": float(best_metrics.get("rmse", float("nan"))),
            "r2": float(best_metrics.get("r2", float("nan"))),
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

    with open(os.path.join(save_dir, "pso_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── 控制台汇报 ────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("PSO 优化完成！")
    logger.info(f"  验证集最优 MSE : {best_mse_val:.6f}")
    logger.info(f"  测试集 MSE     : {test_mse:.6f}")
    logger.info(f"  测试集 MAE     : {test_mae:.6f}")
    logger.info(f"  测试集 RMSE    : {test_rmse:.6f}")
    logger.info(f"  测试集 R²      : {test_r2:.4f}")
    logger.info(f"  实际训练轮数   : {len(history.history['loss'])}")
    logger.info(f"  PSO 耗时       : {pso_elapsed/60:.2f} 分钟")
    logger.info(f"  最终训练耗时   : {final_elapsed/60:.2f} 分钟")
    logger.info(f"  总耗时         : {(pso_elapsed + final_elapsed)/60:.2f} 分钟")
    logger.info(f"  结果目录       : {save_dir}")
    logger.info("=" * 70)

    return results


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    run_pso_optimization(run_id=run_id, seed=seed)
