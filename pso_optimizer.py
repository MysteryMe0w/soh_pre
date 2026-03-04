import numpy as np
import tensorflow as tf
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import json
import os
import sys
from datetime import datetime
import logging

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ImprovedPSOOptimizerV1:
    """
    改进的PSO优化器

    核心改进：
    1. 自适应惯性权重（线性递减）
    2. 速度边界限制
    3. 拉丁超立方采样
    4. 早停机制
    5. 多GPU支持
    6. 详细日志和可视化
    7. 9个核心超参数
    """

    def __init__(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        n_particles=12,
        max_iter=20,
        w_start=0.9,
        w_end=0.4,
        c1=2.0,
        c2=2.0,
        v_max=0.3,
        patience=8,
        train_epochs=50,
        batch_size=64,
        save_dir="./pso_results",
        use_multi_gpu=True,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.v_max = v_max
        self.patience = patience
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.use_multi_gpu = use_multi_gpu

        os.makedirs(save_dir, exist_ok=True)
        self._setup_gpu()

        self.param_bounds = {
            "embed_dim": [16, 64],       # baseline=32, 收紧范围
            "num_heads": [1, 4],         # baseline=2
            "ff_dim": [16, 64],          # baseline=32, 收紧范围
            "transformer_dropout": [0.05, 0.3],
            "cnn_filters": [8, 48],      # baseline=16, 收紧范围
            "cnn_kernel_size": [2, 5],
            "lstm_units": [16, 64],      # baseline=32, 收紧范围
            "final_dropout": [0.3, 0.7], # baseline=0.5, 围绕中心搜索
            "learning_rate": [0.0003, 0.01],  # 更宽的学习率范围
        }

        self.int_params = [
            "embed_dim",
            "num_heads",
            "ff_dim",
            "cnn_filters",
            "cnn_kernel_size",
            "lstm_units",
        ]

        # 历史记录
        self.history = {
            "iterations": [],
            "global_best_scores": [],
            "all_particles_scores": [],
            "best_params_history": [],
            "convergence_speed": [],
        }

    # ─────────────────────────────────────────────────────────────
    # GPU 配置
    # ─────────────────────────────────────────────────────────────

    def _setup_gpu(self):
        """配置GPU策略"""
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

                if self.use_multi_gpu and len(gpus) > 1:
                    self.strategy = tf.distribute.MirroredStrategy()
                    logger.info(f"✅ 使用 {len(gpus)} 个GPU进行并行训练")
                else:
                    self.strategy = tf.distribute.get_strategy()
                    logger.info("✅ 使用单GPU训练")
            except RuntimeError as e:
                logger.warning(f"⚠️ GPU配置失败: {e}")
                self.strategy = tf.distribute.get_strategy()
        else:
            logger.warning("⚠️ 未检测到GPU，使用CPU训练")
            self.strategy = tf.distribute.get_strategy()

    # ─────────────────────────────────────────────────────────────
    # PSO 核心工具
    # ─────────────────────────────────────────────────────────────

    def _get_adaptive_weight(self, iteration):
        """自适应惯性权重（线性递减）"""
        return self.w_start - (self.w_start - self.w_end) * (iteration / self.max_iter)

    def _decode_particle(self, particle):
        """将粒子位置（0~1标准化）解码为实际超参数"""
        params = {}
        keys = list(self.param_bounds.keys())

        for i, key in enumerate(keys):
            min_val, max_val = self.param_bounds[key]
            raw_value = particle[i] * (max_val - min_val) + min_val

            if key in self.int_params:
                params[key] = int(np.round(raw_value))
            else:
                params[key] = float(raw_value)

        # 确保 embed_dim 能被 num_heads 整除（MultiHeadAttention 要求）
        if params["embed_dim"] % params["num_heads"] != 0:
            params["embed_dim"] = (params["embed_dim"] // params["num_heads"]) * params[
                "num_heads"
            ]
            params["embed_dim"] = max(params["num_heads"], params["embed_dim"])

        return params

    def _latin_hypercube_sampling(self, n_samples, n_dims):
        """
        拉丁超立方采样（LHS）
        比随机采样更均匀地覆盖搜索空间：
        每个维度分成 n_samples 个等距区间，每个区间恰好取一个点。
        """
        samples = np.zeros((n_samples, n_dims))
        for j in range(n_dims):
            perm = np.random.permutation(n_samples)
            samples[:, j] = (perm + np.random.rand(n_samples)) / n_samples
        return samples

    # ─────────────────────────────────────────────────────────────
    # 模型构建与适应度评估
    # ─────────────────────────────────────────────────────────────

    def _build_model(self, params):
        """根据超参数构建 Transformer-CNN-BiLSTM 模型"""
        input_shape = (self.X_train.shape[1], self.X_train.shape[2])

        with self.strategy.scope():
            inputs = Input(shape=input_shape)

            # Transformer Encoder
            x = Dense(params["embed_dim"])(inputs)

            attention_output = MultiHeadAttention(
                num_heads=params["num_heads"],
                key_dim=params["embed_dim"],
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
            model.compile(
                optimizer=Adam(learning_rate=params["learning_rate"]),
                loss="mse",
                metrics=["mae"],
            )

        return model

    def _fitness(self, particle, particle_idx=None):
        """
        计算粒子适应度。

        返回：
            score   : float，负MSE（PSO 内部最大化，对应最小化 MSE）
            metrics : dict，含 mse / mae / rmse / r2 / params 等详细信息；
                      异常时含 'error' 键，其余键缺失。
        """
        try:
            params = self._decode_particle(particle)
            model = self._build_model(params)

            callbacks = [
                EarlyStopping(
                    monitor="val_loss",
                    patience=15,                     # 训练早停 patience 固定为15
                    restore_best_weights=True,
                    verbose=0,
                ),
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=0,
                ),
            ]

            history = model.fit(
                self.X_train,
                self.y_train,
                epochs=self.train_epochs,
                batch_size=self.batch_size,
                validation_data=(self.X_val, self.y_val),
                callbacks=callbacks,
                verbose=0,
            )

            y_pred = model(self.X_val, training=False).numpy()

            mse = float(mean_squared_error(self.y_val, y_pred))
            mae = float(mean_absolute_error(self.y_val, y_pred))
            rmse = float(np.sqrt(mse))
            r2 = float(r2_score(self.y_val, y_pred))

            metrics = {
                "mse": mse,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "params": params,
                "final_val_loss": float(history.history["val_loss"][-1]),
                "epochs_trained": len(history.history["loss"]),
            }

            if particle_idx is not None:
                logger.debug(f"粒子{particle_idx:2d}: MSE={mse:.6f}, R²={r2:.4f}")

            # 释放显存
            tf.keras.backend.clear_session()

            return -mse, metrics

        except Exception as e:
            logger.warning(f"❌ 粒子 {particle_idx} 评估失败: {e}")
            tf.keras.backend.clear_session()
            return -float("inf"), {"error": str(e)}

    # ─────────────────────────────────────────────────────────────
    # 主优化循环
    # ─────────────────────────────────────────────────────────────

    def optimize(self):
        """执行 PSO 优化主循环，返回 (best_params, best_mse_val, best_metrics)"""
        n_params = len(self.param_bounds)

        logger.info("=" * 70)
        logger.info("🚀 开始PSO超参数优化")
        logger.info(f"   粒子数: {self.n_particles}, 最大迭代: {self.max_iter}")
        logger.info(f"   优化参数数量: {n_params}")
        logger.info(f"   参数列表: {list(self.param_bounds.keys())}")
        logger.info("=" * 70)

        # ── 初始化 ────────────────────────────────────────────────
        positions = self._latin_hypercube_sampling(self.n_particles, n_params)
        velocities = (np.random.rand(self.n_particles, n_params) - 0.5) * 0.2

        personal_best_positions = positions.copy()
        personal_best_scores = np.full(self.n_particles, -np.inf)
        personal_best_metrics = [None] * self.n_particles

        global_best_position = None
        global_best_score = -np.inf
        global_best_metrics = None

        no_improvement_count = 0

        # ── 初始评估 ──────────────────────────────────────────────
        logger.info(f"\n🔍 初始化评估 {self.n_particles} 个粒子...")
        for i in range(self.n_particles):
            score, metrics = self._fitness(positions[i], i)
            personal_best_scores[i] = score
            personal_best_metrics[i] = metrics

            if score > global_best_score:
                global_best_score = score
                global_best_position = positions[i].copy()
                global_best_metrics = metrics

            print(
                f"\r   进度: [{i+1}/{self.n_particles}] "
                f"{(i+1)/self.n_particles*100:.1f}%",
                end="",
                flush=True,
            )
        print()

        logger.info(f"✅ 初始最优 MSE: {-global_best_score:.6f}")
        logger.info(f"   初始最优参数: {global_best_metrics.get('params', {})}")

        # ── 主迭代 ────────────────────────────────────────────────
        logger.info(f"\n{'='*70}")
        logger.info("🔄 开始迭代优化")
        logger.info(f"{'='*70}\n")

        for iteration in range(self.max_iter):
            iter_start = datetime.now()
            w = self._get_adaptive_weight(iteration)

            prev_global_best_score = global_best_score
            iteration_scores = []

            for i in range(self.n_particles):
                # 速度更新
                r1, r2_rand = np.random.rand(2)
                cognitive = self.c1 * r1 * (personal_best_positions[i] - positions[i])
                social = self.c2 * r2_rand * (global_best_position - positions[i])
                velocities[i] = w * velocities[i] + cognitive + social

                # 速度边界限制
                velocities[i] = np.clip(velocities[i], -self.v_max, self.v_max)

                # 位置更新并裁剪到 [0, 1]
                positions[i] = np.clip(positions[i] + velocities[i], 0.0, 1.0)

                # 评估
                score, metrics = self._fitness(positions[i], i)
                iteration_scores.append(-score if np.isfinite(score) else np.inf)

                # 更新个体最优
                if score > personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i].copy()
                    personal_best_metrics[i] = metrics

                    # 更新全局最优
                    if score > global_best_score:
                        improvement = (
                            (-score - (-global_best_score))
                            / abs(-global_best_score)
                            * 100
                            if global_best_score != -np.inf
                            else float("nan")
                        )
                        logger.info(
                            f"   🎉 发现新最优! MSE: {-score:.6f} "
                            f"(改进 {improvement:.2f}%)"
                        )
                        global_best_score = score
                        global_best_position = positions[i].copy()
                        global_best_metrics = metrics

            # ── 记录历史 ──────────────────────────────────────────
            self.history["iterations"].append(iteration + 1)
            self.history["global_best_scores"].append(-global_best_score)
            self.history["all_particles_scores"].append(iteration_scores)
            self.history["best_params_history"].append(
                global_best_metrics.get("params", {}).copy()
                if global_best_metrics
                else {}
            )

            # 早停计数
            if global_best_score > prev_global_best_score:
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            iter_time = (datetime.now() - iter_start).total_seconds()
            min_score = min(iteration_scores)
            mean_score = float(np.mean(iteration_scores))
            std_score = float(np.std(iteration_scores))

            logger.info(
                f"迭代 {iteration+1:2d}/{self.max_iter} | "
                f"全局最优: {-global_best_score:.6f} | "
                f"本轮: 最佳={min_score:.6f} 均值={mean_score:.6f}±{std_score:.6f} | "
                f"w={w:.3f} | 耗时={iter_time:.1f}s"
            )

            # 早停
            if no_improvement_count >= self.patience:
                logger.info(f"\n⏸️  连续 {self.patience} 轮无改善，提前停止优化")
                break

        # ── 保存结果 ──────────────────────────────────────────────
        self._save_results(global_best_position, global_best_score, global_best_metrics)

        best_params = self._decode_particle(global_best_position)

        logger.info("\n" + "=" * 70)
        logger.info("✅ PSO优化完成!")
        logger.info(f"   最优 MSE  : {-global_best_score:.6f}")
        # ↓ 修复：用 .get() 防止异常粒子导致 KeyError
        logger.info(
            f"   最优 MAE  : {global_best_metrics.get('mae',  float('nan')):.6f}"
        )
        logger.info(
            f"   最优 RMSE : {global_best_metrics.get('rmse', float('nan')):.6f}"
        )
        logger.info(
            f"   最优 R²   : {global_best_metrics.get('r2',   float('nan')):.4f}"
        )
        logger.info("\n   最优超参数:")
        for key, value in best_params.items():
            logger.info(f"      {key:22s}: {value}")
        logger.info("=" * 70)

        return best_params, -global_best_score, global_best_metrics

    # ─────────────────────────────────────────────────────────────
    # 结果保存与可视化
    # ─────────────────────────────────────────────────────────────

    def _save_results(self, best_position, best_score, best_metrics):
        """保存优化结果到 JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results = {
            "best_params": self._decode_particle(best_position),
            "best_mse": float(-best_score),
            "best_metrics": {
                k: (
                    float(v)
                    if isinstance(v, (np.floating, float, np.integer, int))
                    else v
                )
                for k, v in best_metrics.items()
                if k != "params"
            },
            "optimization_config": {
                "n_particles": self.n_particles,
                "max_iter": self.max_iter,
                "w_start": self.w_start,
                "w_end": self.w_end,
                "c1": self.c1,
                "c2": self.c2,
                "v_max": self.v_max,
                "patience": self.patience,
                "train_epochs": self.train_epochs,
            },
            "history": {
                "iterations": self.history["iterations"],
                "global_best_scores": [
                    float(s) for s in self.history["global_best_scores"]
                ],
                "convergence_iteration": len(self.history["iterations"]),
            },
            "timestamp": timestamp,
        }

        result_path = os.path.join(self.save_dir, f"pso_results_{timestamp}.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"\n💾 结果已保存至: {result_path}")

    def plot_convergence(self, save_path=None):
        """绘制收敛曲线（全局最优曲线 + 粒子得分箱线图）"""
        import matplotlib.pyplot as plt

        # ── 守卫：无历史记录时直接返回 ───────────────────────────
        if not self.history["iterations"]:
            logger.warning("⚠️ 无优化历史记录，跳过绘图")
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # 左图：全局最优收敛曲线
        axes[0].plot(
            self.history["iterations"],
            self.history["global_best_scores"],
            "b-o",
            linewidth=2.5,
            markersize=6,
            label="Global Best MSE",
        )
        axes[0].set_xlabel("Iteration", fontsize=13, fontweight="bold")
        axes[0].set_ylabel("MSE", fontsize=13, fontweight="bold")
        axes[0].set_title("PSO Convergence Curve", fontsize=15, fontweight="bold")
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3, linestyle="--")

        # ── 修复：用 argmin 找到真正的最优点，而非最后一个点 ─────
        best_idx = int(np.argmin(self.history["global_best_scores"]))
        best_iter = self.history["iterations"][best_idx]
        best_score = self.history["global_best_scores"][best_idx]

        # xytext 做边界保护，避免标注跑出图外
        x_range = max(self.history["iterations"]) - min(self.history["iterations"])
        y_range = max(self.history["global_best_scores"]) - min(
            self.history["global_best_scores"]
        )
        text_x = best_iter - x_range * 0.15
        text_y = best_score + y_range * 0.15 + 1e-8  # +1e-8 防止 y_range=0 时重叠

        axes[0].annotate(
            f"Best: {best_score:.6f}",
            xy=(best_iter, best_score),
            xytext=(max(text_x, min(self.history["iterations"])), text_y),
            arrowprops=dict(arrowstyle="->", color="red", lw=2),
            fontsize=11,
            color="red",
            fontweight="bold",
        )

        # 右图：每轮粒子得分分布（箱线图）
        if self.history["all_particles_scores"]:
            axes[1].boxplot(
                self.history["all_particles_scores"],
                positions=self.history["iterations"],
                widths=0.6,
            )
            axes[1].set_xlabel("Iteration", fontsize=13, fontweight="bold")
            axes[1].set_ylabel("MSE", fontsize=13, fontweight="bold")
            axes[1].set_title(
                "Particle Score Distribution", fontsize=15, fontweight="bold"
            )
            axes[1].grid(True, alpha=0.3, linestyle="--")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"📊 收敛曲线已保存至: {save_path}")

        plt.show()


# ─────────────────────────────────────────────────────────────────
# 快速配置版本
# ─────────────────────────────────────────────────────────────────


class FastPSOOptimizer(ImprovedPSOOptimizerV1):
    """
    快速PSO优化器（30~60 分钟版本）

    相对于 ImprovedPSOOptimizerV1 的变化：
        n_particles : 12 → 8
        max_iter    : 20 → 15
        train_epochs: 50 → 30
        patience    :  8 → 5

    预计评估次数：8 × 15 = 120 次
    """

    def __init__(self, X_train, y_train, X_val, y_val, **kwargs):
        fast_config = {
            "n_particles": 15,       # 8→15: 更多粒子覆盖搜索空间
            "max_iter": 30,          # 15→30: 更多迭代轮次
            "train_epochs": 100,     # 30→100: 与baseline训练轮数一致
            "patience": 10,          # 5→10: 给PSO更多探索机会
            "w_start": 0.9,
            "w_end": 0.3,            # 0.4→0.3: 后期更强的局部搜索
            "c1": 1.5,               # 2.0→1.5: 降低认知系数
            "c2": 2.5,               # 2.0→2.5: 增强社会系数，加速收敛
            "v_max": 0.2,            # 0.3→0.2: 更细粒度的搜索步长
            "batch_size": 16,        # 64→16: 与baseline一致
            "use_multi_gpu": True,
        }
        # 用户传入的 kwargs 优先级更高，可覆盖上述默认值
        fast_config.update(kwargs)
        super().__init__(X_train, y_train, X_val, y_val, **fast_config)


# ─────────────────────────────────────────────────────────────────
# 独立测试入口（作为模块导入时不执行）
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── 修复：用命令行参数替代 input()，避免服务器后台运行时 EOFError
    # 用法：
    #   python pso_optimizer.py 1   → 标准配置
    #   python pso_optimizer.py 2   → 快速配置（默认）
    choice = sys.argv[1] if len(sys.argv) > 1 else "2"

    # 模拟数据（实际使用时替换为真实数据）
    X_train = np.random.rand(100, 4, 13).astype(np.float32)
    y_train = np.random.rand(100, 1).astype(np.float32)
    X_val = np.random.rand(20, 4, 13).astype(np.float32)
    y_val = np.random.rand(20, 1).astype(np.float32)

    if choice == "1":
        logger.info("使用标准配置 (ImprovedPSOOptimizerV1)...")
        optimizer = ImprovedPSOOptimizerV1(
            X_train,
            y_train,
            X_val,
            y_val,
            n_particles=12,
            max_iter=20,
            train_epochs=50,
            save_dir="./pso_results",
        )
    else:
        logger.info("使用快速配置 (FastPSOOptimizer)...")
        optimizer = FastPSOOptimizer(
            X_train, y_train, X_val, y_val, save_dir="./pso_results"
        )

    best_params, best_mse, best_metrics = optimizer.optimize()
    optimizer.plot_convergence(save_path="./pso_results/convergence.png")

    logger.info(f"\n{'='*70}")
    logger.info("优化结果汇总:")
    logger.info(f"{'='*70}")
    logger.info(f"最优 MSE : {best_mse:.6f}")
    logger.info(f"最优 R²  : {best_metrics.get('r2', float('nan')):.4f}")
    logger.info("\n最优超参数:")
    for key, value in best_params.items():
        logger.info(f"  {key:22s}: {value}")
    logger.info(f"{'='*70}")
