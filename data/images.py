import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# 设置中文字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_metrics_files(evaluation_dir):
    """
    从评估结果目录加载所有指标JSON文件
    返回一个包含所有指标数据的字典
    """
    metrics_data = {}
    
    # 遍历目录中的所有文件
    for filename in os.listdir(evaluation_dir):
        if not filename.endswith('.json') or '_metrics_limit' not in filename:
            continue
            
        filepath = os.path.join(evaluation_dir, filename)
        
        try:
            # 加载JSON文件
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 从meta信息中提取模型名称和样本数量
            meta = data.get('meta', {})
            evaluated_file = meta.get('evaluated_file', '')
            data_limit = meta.get('data_limit')
            
            if not evaluated_file or data_limit is None:
                print(f"文件 {filename} 中缺少必要的元信息")
                continue
                
            # 从evaluated_file中提取模型名称
            path_parts = Path(evaluated_file).parts
            
            if len(path_parts) >= 2:
                # 通常格式为 ['output', 'model-name', 'filename.json']
                model_name = path_parts[1]
            else:
                # 尝试从文件名提取
                base_filename = os.path.basename(evaluated_file)
                if '_topic_results_' in base_filename:
                    model_name = base_filename.split('_topic_results_')[0]
                else:
                    print(f"无法从 {evaluated_file} 中提取模型名称")
                    continue
                
            # 提取指标数据
            metrics = data.get('metrics', {})
            
            if model_name not in metrics_data:
                metrics_data[model_name] = {}
                
            metrics_data[model_name][data_limit] = metrics
            
            print(f"已加载 {model_name} 在样本数量 {data_limit} 下的评估指标")
                
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")
    
    return metrics_data

def create_comparison_tables(metrics_data, output_dir):
    """
    为每个指标和k值创建比较表格
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有模型名称和样本数量
    model_names = list(metrics_data.keys())
    limits = sorted(list(set(limit for model_data in metrics_data.values() for limit in model_data.keys())))
    
    # 对于每个指标创建表格
    metrics = ["precision", "recall", "f1", "success_rate"]
    metric_names = {
        "precision": "精确率",
        "recall": "召回率",
        "f1": "F1分数",
        "success_rate": "成功率"
    }
    k_values = [3, 4, 5]
    
    # 创建一个包含所有指标的主表格
    all_metrics_data = []
    
    for limit in limits:
        for k in k_values:
            for metric in metrics:
                row = {
                    "样本数量": limit, 
                    "k值": k, 
                    "指标类型": metric_names[metric]
                }
                
                for model in model_names:
                    if limit in metrics_data[model] and metric in metrics_data[model][limit]:
                        row[model] = metrics_data[model][limit][metric].get(str(k), 0)
                    else:
                        row[model] = None
                        
                all_metrics_data.append(row)
    
    # 创建DataFrame并保存为CSV
    df_all = pd.DataFrame(all_metrics_data)
    all_csv_path = os.path.join(output_dir, "all_metrics_comparison.csv")
    df_all.to_csv(all_csv_path, index=False, encoding='utf-8-sig')
    print(f"已保存所有指标的比较表格到 {all_csv_path}")
    
    # 为每个指标创建单独的表格
    for metric in metrics:
        metric_data = []
        
        for limit in limits:
            for k in k_values:
                row = {"样本数量": limit, "k值": k}
                
                for model in model_names:
                    if limit in metrics_data[model] and metric in metrics_data[model][limit]:
                        row[model] = metrics_data[model][limit][metric].get(str(k), 0)
                    else:
                        row[model] = None
                        
                metric_data.append(row)
        
        # 创建DataFrame并保存为CSV
        df_metric = pd.DataFrame(metric_data)
        metric_csv_path = os.path.join(output_dir, f"{metric}_comparison.csv")
        df_metric.to_csv(metric_csv_path, index=False, encoding='utf-8-sig')
        print(f"已保存指标 {metric} 的比较表格到 {metric_csv_path}")

def create_bar_charts(metrics_data, output_dir):
    """
    创建条形图，比较不同模型在相同样本数量和k值下的性能
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有模型名称和样本数量
    model_names = list(metrics_data.keys())
    limits = sorted(list(set(limit for model_data in metrics_data.values() for limit in model_data.keys())))
    
    # 对于每个指标创建条形图
    metrics = ["precision", "recall", "f1", "success_rate"]
    k_values = [3, 4, 5]
    metric_names = {
        "precision": "精确率",
        "recall": "召回率",
        "f1": "F1分数",
        "success_rate": "成功率"
    }
    
    # 设置颜色方案
    colors = sns.color_palette("husl", len(model_names))
    color_dict = {model: color for model, color in zip(model_names, colors)}
    
    # 为每个指标和k值创建条形图
    for metric in metrics:
        for k in k_values:
            for limit in limits:
                plt.figure(figsize=(12, 6))
                
                # 收集数据
                values = []
                models = []
                
                for model in model_names:
                    if limit in metrics_data[model] and metric in metrics_data[model][limit]:
                        value = metrics_data[model][limit][metric].get(str(k), 0)
                        values.append(value)
                        models.append(model)
                
                # 创建条形图
                bars = plt.bar(models, values, color=[color_dict[model] for model in models])
                
                # 添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.4f}', ha='center', va='bottom', rotation=0)
                
                # 设置图表属性
                plt.title(f"{metric_names[metric]}对比 (k={k}, 样本数={limit})")
                plt.ylabel(metric_names[metric])
                plt.xlabel("模型")
                plt.xticks(rotation=45, ha='right')
                plt.ylim(0, max(values) * 1.2)  # 设置合适的y轴范围
                plt.tight_layout()
                
                # 保存图表
                img_path = os.path.join(output_dir, f"{metric}_k{k}_limit{limit}_bar.png")
                plt.savefig(img_path, dpi=300)
                plt.close()
                print(f"已保存指标 {metric}，k={k}，样本数={limit} 的条形图到 {img_path}")

def create_line_charts(metrics_data, output_dir):
    """
    创建折线图，比较不同模型在k值和样本数量变化下的性能趋势
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有模型名称和样本数量
    model_names = list(metrics_data.keys())
    limits = sorted(list(set(limit for model_data in metrics_data.values() for limit in model_data.keys())))
    
    # 对于每个指标创建折线图
    metrics = ["precision", "recall", "f1", "success_rate"]
    k_values = [3, 4, 5]
    metric_names = {
        "precision": "精确率",
        "recall": "召回率",
        "f1": "F1分数",
        "success_rate": "成功率"
    }
    
    # 设置颜色方案
    colors = sns.color_palette("husl", len(model_names))
    color_dict = {model: color for model, color in zip(model_names, colors)}
    
    # 1. k值变化趋势图(固定样本数量)
    for metric in metrics:
        for limit in limits:
            plt.figure(figsize=(12, 6))
            
            for model in model_names:
                if limit in metrics_data[model] and metric in metrics_data[model][limit]:
                    values = [metrics_data[model][limit][metric].get(str(k), 0) for k in k_values]
                    plt.plot(k_values, values, marker='o', label=model, color=color_dict[model])
            
            plt.title(f"{metric_names[metric]}随k值变化 (样本数={limit})")
            plt.xlabel("k值")
            plt.ylabel(metric_names[metric])
            plt.xticks(k_values)
            plt.legend(loc='best')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # 保存图表
            img_path = os.path.join(output_dir, f"{metric}_limit{limit}_k_trend.png")
            plt.savefig(img_path, dpi=300)
            plt.close()
            print(f"已保存指标 {metric}，样本数={limit} 的k值趋势图到 {img_path}")
    
    # 2. 样本数量变化趋势图(固定k值)
    for metric in metrics:
        for k in k_values:
            plt.figure(figsize=(12, 6))
            
            for model in model_names:
                values = []
                valid_limits = []
                
                for limit in limits:
                    if limit in metrics_data[model] and metric in metrics_data[model][limit]:
                        value = metrics_data[model][limit][metric].get(str(k), 0)
                        values.append(value)
                        valid_limits.append(limit)
                
                if values:
                    plt.plot(valid_limits, values, marker='o', label=model, color=color_dict[model])
            
            plt.title(f"{metric_names[metric]}随样本数量变化 (k={k})")
            plt.xlabel("样本数量")
            plt.ylabel(metric_names[metric])
            plt.xticks(limits, [str(l) for l in limits])
            plt.legend(loc='best')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # 保存图表
            img_path = os.path.join(output_dir, f"{metric}_k{k}_sample_trend.png")
            plt.savefig(img_path, dpi=300)
            plt.close()
            print(f"已保存指标 {metric}，k={k} 的样本数量趋势图到 {img_path}")

def main():
    """
    主函数，执行所有可视化任务
    """
    # 评估结果目录
    evaluation_dir = "metrics"
    
    # 输出目录
    tables_dir = "metrics_visual/tables"
    charts_dir = "metrics_visual/charts"
    
    # 加载指标数据
    print("正在加载评估指标数据...")
    metrics_data = load_metrics_files(evaluation_dir)
    
    if not metrics_data:
        print("没有找到有效的评估指标数据。请确保评估结果目录中包含有效的JSON文件。")
        return
    
    # 创建比较表格
    print("\n正在创建比较表格...")
    create_comparison_tables(metrics_data, tables_dir)
    
    # 创建条形图
    print("\n正在创建条形图...")
    create_bar_charts(metrics_data, charts_dir)
    
    # 创建折线图
    print("\n正在创建折线图...")
    create_line_charts(metrics_data, charts_dir)
    
    print("\n所有可视化任务已完成！")
    print(f"表格保存在: {os.path.abspath(tables_dir)}")
    print(f"图表保存在: {os.path.abspath(charts_dir)}")

if __name__ == "__main__":
    main()