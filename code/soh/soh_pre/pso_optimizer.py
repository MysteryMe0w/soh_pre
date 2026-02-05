class FastPSOOptimizer(ImprovedPSOHyperparameterOptimizer):
    """快速PSO优化器（30-60分钟版本）"""

    def __init__(self, X_train, y_train, X_val, y_val, **kwargs):

        # 覆盖默认参数
        fast_config = {
            "n_particles": 10,  # 30 → 10
            "max_iter": 15,  # 50 → 15
            "train_epochs": 50,  # 100 → 50
            "patience": 5,  # 15 → 5
            "w_start": 0.9,
            "w_end": 0.4,
            "c1": 2.0,
            "c2": 2.0,
            "v_max": 0.3,
            "batch_size": 64,
            "use_multi_gpu": True,
        }

        # 更新用户提供的参数
        fast_config.update(kwargs)

        super().__init__(X_train, y_train, X_val, y_val, **fast_config)

        # 简化超参数空间（只优化核心参数）
        self.param_bounds = {
            "embed_dim": [32, 128],
            "num_heads": [2, 4],
            "ff_dim": [32, 128],
            "transformer_dropout": [0.1, 0.3],
            "cnn_filters": [16, 64],
            "cnn_kernel_size": [2, 5],
            "lstm_units": [32, 128],
            "final_dropout": [0.3, 0.5],
            "learning_rate": [0.0005, 0.005],
        }

        self.int_params = [
            "embed_dim",
            "num_heads",
            "ff_dim",
            "cnn_filters",
            "cnn_kernel_size",
            "lstm_units",
        ]
