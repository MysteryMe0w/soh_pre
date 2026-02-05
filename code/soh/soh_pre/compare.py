import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

# 设置中文字体
import platform

if platform.system() == "Windows":
    plt.rcParams["font.family"] = ["SimHei"]
elif platform.system() == "Darwin":
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

sns.set_style("whitegrid")


def load_baseline_results():
    """加载baseline结果"""
    baseline_dir = os.path.join(os.path.dirname(__file__), "models/baseline")
    metrics_file = os.path.join(baseline_dir, "baseline_metrics.json")

    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            return json.load(f)
    else:
        print(f"警告: 未找到baseline结果文件 {metrics_file}")
        return None


def load_pso_results():
    """加载所有PSO运行结果"""
    pso_base_dir = os.path.join(os.path.dirname(__file__), "models/pso_optimized")

    results = []
    for run_dir in sorted(glob(os.path.join(pso_base_dir, "run_*"))):
        metrics_file = os.path.join(run_dir, "pso_metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                data = json.load(f)
                data["run_dir"] = run_dir
                results.append(data)

    return results


def calculate_pso_statistics(pso_results):
    """计算PSO多次运行的统计量"""
    if not pso_results:
        return None

    test_mse = [r["test_metrics"]["mse"] for r in pso_results]
    test_mae = [r["test_metrics"]["mae"] for r in pso_results]
    test_rmse = [r["test_metrics"]["rmse"] for r in pso_results]
    test_r2 = [r["test_metrics"]["r2"] for r in pso_results]

    stats = {
        "mse": {
            "mean": np.mean(test_mse),
            "std": np.std(test_mse),
            "min": np.min(test_mse),
            "max": np.max(test_mse),
        },
        "mae": {
            "mean": np.mean(test_mae),
            "std": np.std(test_mae),
            "min": np.min(test_mae),
            "max": np.max(test_mae),
        },
        "rmse": {
            "mean": np.mean(test_rmse),
            "std": np.std(test_rmse),
            "min": np.min(test_rmse),
            "max": np.max(test_rmse),
        },
        "r2": {
            "mean": np.mean(test_r2),
            "std": np.std(test_r2),
            "min": np.min(test_r2),
            "max": np.max(test_r2),
        },
        "n_runs": len(pso_results),
    }

    return stats


def create_comparison_table(baseline, pso_stats):
    """创建对比表格"""
    if baseline is None or pso_stats is None:
        print("数据不完整，无法生成对比表格")
        return None

    data = {
        "Metric": ["MSE", "MAE", "RMSE", "R²"],
        "Baseline": [
            baseline["test_metrics"]["mse"],
            baseline["test_metrics"]["mae"],
            baseline["test_metrics"]["rmse"],
            baseline["test_metrics"]["r2"],
        ],
        "PSO Mean": [
            pso_stats["mse"]["mean"],
            pso_stats["mae"]["mean"],
            pso_stats["rmse"]["mean"],
            pso_stats["r2"]["mean"],
        ],
        "PSO Std": [
            pso_stats["mse"]["std"],
            pso_stats["mae"]["std"],
            pso_stats["rmse"]["std"],
            pso_stats["r2"]["std"],
        ],
        "PSO Best": [
            pso_stats["mse"]["min"],
            pso_stats["mae"]["min"],
            pso_stats["rmse"]["min"],
            pso_stats["r2"]["max"],  # R²越大越好
        ],
    }

    df = pd.DataFrame(data)

    # 计算改进百分比
    improvements = []
    for i, metric in enumerate(["mse", "mae", "rmse", "r2"]):
        baseline_val = df.loc[i, "Baseline"]
        pso_val = df.loc[i, "PSO Mean"]

        if metric == "r2":
            # R²越大越好
            improvement = ((pso_val - baseline_val) / baseline_val) * 100
        else:
            # MSE/MAE/RMSE越小越好
            improvement = ((baseline_val - pso_val) / baseline_val) * 100

        improvements.append(improvement)

    df["Improvement (%)"] = improvements

    return df


def plot_metrics_comparison(baseline, pso_stats, save_dir):
    """绘制指标对比图"""
    metrics = ["MSE", "MAE", "RMSE", "R²"]
    baseline_vals = [
        baseline["test_metrics"]["mse"],
        baseline["test_metrics"]["mae"],
        baseline["test_metrics"]["rmse"],
        baseline["test_metrics"]["r2"],
    ]
    pso_means = [
        pso_stats["mse"]["mean"],
        pso_stats["mae"]["mean"],
        pso_stats["rmse"]["mean"],
        pso_stats["r2"]["mean"],
    ]
    pso_stds = [
        pso_stats["mse"]["std"],
        pso_stats["mae"]["std"],
        pso_stats["rmse"]["std"],
        pso_stats["r2"]["std"],
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (ax, metric) in enumerate(zip(axes, metrics)):
        x = ["Baseline", "PSO"]
        y = [baseline_vals[idx], pso_means[idx]]
        colors = ["#3498db", "#e74c3c"]

        bars = ax.bar(x, y, color=colors, alpha=0.7, edgecolor="black")

        # 为PSO添加误差棒
        ax.errorbar(
            1,
            pso_means[idx],
            yerr=pso_stds[idx],
            fmt="none",
            ecolor="black",
            capsize=10,
            capthick=2,
        )

        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, y)):
            height = bar.get_height()
            if i == 1:  # PSO
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{val:.6f}\n±{pso_stds[idx]:.6f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{val:.6f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f"{metric} Comparison", fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "metrics_comparison.png"), dpi=300, bbox_inches="tight"
    )
    print(f"指标对比图已保存至: {os.path.join(save_dir, 'metrics_comparison.png')}")
    plt.show()


def plot_prediction_comparison(baseline_dir, pso_results, save_dir):
    """绘制预测曲线对比图"""
    # 加载baseline预测
    baseline_y_test = np.load(os.path.join(baseline_dir, "y_test.npy"))
    baseline_y_pred = np.load(os.path.join(baseline_dir, "y_pred.npy"))

    # 加载最佳PSO预测（MSE最小的那次）
    best_pso = min(pso_results, key=lambda x: x["test_metrics"]["mse"])
    pso_y_test = np.load(os.path.join(best_pso["run_dir"], "y_test.npy"))
    pso_y_pred = np.load(os.path.join(best_pso["run_dir"], "y_pred.npy"))

    # 绘图
    plt.figure(figsize=(14, 6))

    x = np.arange(len(baseline_y_test))

    plt.plot(
        x,
        baseline_y_test.flatten(),
        "o-",
        label="Ground Truth",
        color="black",
        linewidth=2,
        markersize=4,
        alpha=0.7,
    )
    plt.plot(
        x,
        baseline_y_pred.flatten(),
        "s--",
        label="Baseline Prediction",
        color="#3498db",
        linewidth=1.5,
        markersize=4,
        alpha=0.7,
    )
    plt.plot(
        x,
        pso_y_pred.flatten(),
        "^--",
        label="PSO Optimized Prediction",
        color="#e74c3c",
        linewidth=1.5,
        markersize=4,
        alpha=0.7,
    )

    plt.xlabel("Sample Index", fontsize=13)
    plt.ylabel("Normalized SOH", fontsize=13)
    plt.title(
        "Prediction Comparison: Baseline vs PSO Optimized",
        fontsize=15,
        fontweight="bold",
    )
    plt.legend(fontsize=11, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(save_dir, "prediction_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    print(f"预测对比图已保存至: {os.path.join(save_dir, 'prediction_comparison.png')}")
    plt.show()


def plot_convergence_all_runs(pso_results, save_dir):
    """绘制所有PSO运行的收敛曲线"""
    plt.figure(figsize=(12, 6))

    for result in pso_results:
        run_id = result["run_id"]
        pso_result_file = os.path.join(result["run_dir"], f"pso_results_*.json")

        # 找到PSO结果文件
        result_files = glob(pso_result_file)
        if result_files:
            with open(result_files[0], "r") as f:
                pso_data = json.load(f)
                iterations = pso_data["history"]["iterations"]
                scores = pso_data["history"]["global_best_scores"]

                plt.plot(
                    iterations,
                    scores,
                    "-o",
                    label=f"Run {run_id}",
                    linewidth=2,
                    markersize=4,
                    alpha=0.7,
                )

    plt.xlabel("Iteration", fontsize=13)
    plt.ylabel("Best MSE", fontsize=13)
    plt.title("PSO Convergence Curves (All Runs)", fontsize=15, fontweight="bold")
    plt.legend(fontsize=10, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        os.path.join(save_dir, "convergence_all_runs.png"), dpi=300, bbox_inches="tight"
    )
    print(
        f"收敛曲线对比图已保存至: {os.path.join(save_dir, 'convergence_all_runs.png')}"
    )
    plt.show()


def generate_summary_report(baseline, pso_stats, pso_results, save_dir):
    """生成文字总结报告"""
    report_path = os.path.join(save_dir, "summary_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Transformer-CNN-BiLSTM: Baseline vs PSO Optimization\n")
        f.write("实验对比总结报告\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"PSO运行次数: {pso_stats['n_runs']}\n\n")

        f.write("-" * 80 + "\n")
        f.write("测试集性能指标对比\n")
        f.write("-" * 80 + "\n\n")

        metrics_names = {"mse": "MSE", "mae": "MAE", "rmse": "RMSE", "r2": "R²"}

        for key, name in metrics_names.items():
            baseline_val = baseline["test_metrics"][key]
            pso_mean = pso_stats[key]["mean"]
            pso_std = pso_stats[key]["std"]
            pso_best = pso_stats[key]["min"] if key != "r2" else pso_stats[key]["max"]

            if key == "r2":
                improvement = ((pso_mean - baseline_val) / baseline_val) * 100
                best_improvement = ((pso_best - baseline_val) / baseline_val) * 100
            else:
                improvement = ((baseline_val - pso_mean) / baseline_val) * 100
                best_improvement = ((baseline_val - pso_best) / baseline_val) * 100

            f.write(f"{name}:\n")
            f.write(f"  Baseline:           {baseline_val:.6f}\n")
            f.write(f"  PSO Mean:           {pso_mean:.6f} ± {pso_std:.6f}\n")
            f.write(f"  PSO Best:           {pso_best:.6f}\n")
            f.write(f"  平均改进:           {improvement:+.2f}%\n")
            f.write(f"  最佳改进:           {best_improvement:+.2f}%\n\n")

        f.write("-" * 80 + "\n")
        f.write("计算时间对比\n")
        f.write("-" * 80 + "\n\n")

        baseline_time = baseline.get("training_time_seconds", 0) / 60
        avg_pso_time = np.mean([r["total_time_minutes"] for r in pso_results])

        f.write(f"Baseline训练时间:    {baseline_time:.2f} 分钟\n")
        f.write(f"PSO平均总时间:       {avg_pso_time:.2f} 分钟\n")
        f.write(f"时间倍数:            {avg_pso_time/baseline_time:.2f}x\n\n")

        f.write("-" * 80 + "\n")
        f.write("最佳PSO超参数\n")
        f.write("-" * 80 + "\n\n")

        best_pso = min(pso_results, key=lambda x: x["test_metrics"]["mse"])
        for key, value in best_pso["best_params"].items():
            f.write(f"  {key:25s}: {value}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("结论: ")

        mse_improvement = (
            (baseline["test_metrics"]["mse"] - pso_stats["mse"]["mean"])
            / baseline["test_metrics"]["mse"]
        ) * 100

        if mse_improvement > 5:
            f.write(
                f"PSO优化显著提升了模型性能，MSE平均降低了 {mse_improvement:.2f}%\n"
            )
        elif mse_improvement > 0:
            f.write(
                f"PSO优化略微提升了模型性能，MSE平均降低了 {mse_improvement:.2f}%\n"
            )
        else:
            f.write(
                f"PSO优化未能提升模型性能，MSE平均增加了 {abs(mse_improvement):.2f}%\n"
            )

        f.write("=" * 80 + "\n")

    print(f"\n总结报告已保存至: {report_path}")

    # 打印到控制台
    with open(report_path, "r", encoding="utf-8") as f:
        print("\n" + f.read())


def main():
    """主函数"""
    print("=" * 80)
    print("开始生成对比分析报告...")
    print("=" * 80)

    # 创建结果保存目录
    save_dir = os.path.join(os.path.dirname(__file__), "results/comparison_report")
    os.makedirs(save_dir, exist_ok=True)

    # 加载数据
    print("\n1. 加载Baseline结果...")
    baseline = load_baseline_results()

    print("2. 加载PSO优化结果...")
    pso_results = load_pso_results()

    if not pso_results:
        print("错误: 未找到PSO优化结果！请先运行 train_with_pso.py")
        return

    print(f"   找到 {len(pso_results)} 次PSO运行结果")

    # 计算统计量
    print("\n3. 计算PSO统计量...")
    pso_stats = calculate_pso_statistics(pso_results)

    # 生成对比表格
    print("\n4. 生成对比表格...")
    comparison_df = create_comparison_table(baseline, pso_stats)

    if comparison_df is not None:
        print("\n" + "=" * 80)
        print(comparison_df.to_string(index=False))
        print("=" * 80)

        # 保存表格
        comparison_df.to_csv(
            os.path.join(save_dir, "metrics_comparison.csv"), index=False
        )
        print(f"\n对比表格已保存至: {os.path.join(save_dir, 'metrics_comparison.csv')}")

    # 绘制图表
    if baseline and pso_stats:
        print("\n5. 绘制指标对比图...")
        plot_metrics_comparison(baseline, pso_stats, save_dir)

        print("\n6. 绘制预测曲线对比图...")
        baseline_dir = os.path.join(os.path.dirname(__file__), "models/baseline")
        plot_prediction_comparison(baseline_dir, pso_results, save_dir)

    print("\n7. 绘制收敛曲线对比图...")
    plot_convergence_all_runs(pso_results, save_dir)

    # 生成总结报告
    print("\n8. 生成总结报告...")
    generate_summary_report(baseline, pso_stats, pso_results, save_dir)

    print("\n" + "=" * 80)
    print("对比分析完成！所有结果已保存至:", save_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()
