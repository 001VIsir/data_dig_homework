from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def calculate_metrics(y_true, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='binary'),
        'Recall': recall_score(y_true, y_pred, average='binary'),
        'F1 Score': f1_score(y_true, y_pred, average='binary')
    }

def plot_and_save_results(results_df, save_path='results'):
    os.makedirs(save_path, exist_ok=True)
    
    # 保存为CSV
    csv_path = os.path.join(save_path, 'model_comparison.csv')
    results_df.to_csv(csv_path)
    print(f"结果已保存到 {csv_path}")

    # 绘制图表
    results_df.plot(kind='bar', figsize=(14, 8), rot=45)
    plt.title('不同模型和预处理方法的性能比较')
    plt.ylabel('分数')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    
    plot_path = os.path.join(save_path, 'model_comparison.png')
    plt.savefig(plot_path)
    print(f"图表已保存到 {plot_path}")
    plt.show()
