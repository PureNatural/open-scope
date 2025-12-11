import json
import re
from pathlib import Path
import ast
import numpy as np
import os
from typing import List, Dict, Set, Any, Tuple
from datetime import datetime

def load_results(result_file: str, limit: int = None) -> List[Dict]:
    """
    加载结果JSON文件
    
    参数:
        result_file: JSON文件路径
        limit: 限制加载的数据条数，None表示加载所有数据
    """
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        total_repos = len(results)
        
        # 如果设置了limit参数，只取前n条数据
        if limit is not None and limit < total_repos:
            results = results[:limit]
            print(f"成功加载结果文件，共包含 {total_repos} 个仓库的数据，已限制只使用前 {limit} 条")
        else:
            print(f"成功加载结果文件，共包含 {total_repos} 个仓库的数据")
            
        return results
    except Exception as e:
        print(f"加载结果文件时出错: {str(e)}")
        return []

def parse_topics(topics_str: str) -> Set[str]:
    """解析topics字符串为集合"""
    if not topics_str or topics_str == '':
        return set()
        
    # 处理可能的格式: {'topic1','topic2'} 或 ["topic1", "topic2"]
    try:
        # 尝试使用ast.literal_eval解析，这是处理Python字面量的安全方式
        topics = ast.literal_eval(topics_str)
        if isinstance(topics, set):
            return topics
        elif isinstance(topics, list):
            return set(topics)
        else:
            return {str(topics)}
    except (SyntaxError, ValueError):
        # 如果解析失败，使用正则表达式提取
        pattern = r'[\'\"]([^\'\"]*)[\'\"]'
        matches = re.findall(pattern, topics_str)
        return set(matches)

def calculate_metrics(results: List[Dict], k_values: List[int] = [5, 10]) -> Dict[str, Dict[int, float]]:
    """计算各种评估指标"""
    metrics = {
        'precision': {k: 0.0 for k in k_values},
        'recall': {k: 0.0 for k in k_values},
        'f1': {k: 0.0 for k in k_values},
        'success_rate': {k: 0.0 for k in k_values}
    }
    
    valid_repos = 0  # 有效仓库数量（包含原始topics的仓库）
    
    # 用于存储每个K值下每个仓库的精准度和召回率
    precisions = {k: [] for k in k_values}
    recalls = {k: [] for k in k_values}
    successes = {k: [] for k in k_values}
    
    for repo in results:
        # 跳过发生错误的仓库
        if 'error' in repo:
            continue
            
        # 解析原始topics集合
        original_topics_str = repo.get('original_topics', '')
        original_topics = parse_topics(original_topics_str)
        
        # 如果原始topics为空，跳过该仓库
        if not original_topics:
            continue
            
        valid_repos += 1
        
        # 获取预测的topics
        suggested_topics = repo.get('suggested_topics', [])
        
        # 对每个K值计算指标
        for k in k_values:
            # 确保k不大于预测topics的数量
            actual_k = min(k, len(suggested_topics))
            
            # 获取前K个预测topics
            top_k_topics = set(suggested_topics[:actual_k])
            
            # 计算命中数（交集数量）
            hits = len(top_k_topics.intersection(original_topics))
            
            # 计算精准度 Precision@K
            precision = hits / actual_k if actual_k > 0 else 0
            precisions[k].append(precision)
            
            # 计算召回率 Recall@K
            recall = hits / len(original_topics) if len(original_topics) > 0 else 0
            recalls[k].append(recall)
            
            # 判断是否至少有一个命中
            success = hits > 0
            successes[k].append(success)
    
    # 计算所有指标的平均值
    for k in k_values:
        if valid_repos > 0:
            # 平均精准度
            metrics['precision'][k] = np.mean(precisions[k]) if precisions[k] else 0
            
            # 平均召回率
            metrics['recall'][k] = np.mean(recalls[k]) if recalls[k] else 0
            
            # F1值 (精准度和召回率的调和平均值)
            if metrics['precision'][k] + metrics['recall'][k] > 0:
                metrics['f1'][k] = 2 * (metrics['precision'][k] * metrics['recall'][k]) / (metrics['precision'][k] + metrics['recall'][k])
            else:
                metrics['f1'][k] = 0
                
            # 成功率
            metrics['success_rate'][k] = np.mean(successes[k]) if successes[k] else 0
    
    return metrics, valid_repos

def format_results(metrics: Dict[str, Dict[int, float]], valid_repos: int, limit: int = None) -> str:
    """将结果格式化为易读的字符串"""
    # 添加限制信息到结果字符串
    limit_info = f"（限制前 {limit} 条数据）" if limit is not None else ""
    result_str = f"评估结果 (基于 {valid_repos} 个有效仓库{limit_info}):\n"
    result_str += "=" * 60 + "\n"
    
    for k in sorted(metrics['precision'].keys()):
        result_str += f"Top-{k} 指标:\n"
        result_str += f"  精准度 (Precision@{k}): {metrics['precision'][k]:.4f}\n"
        result_str += f"  召回率 (Recall@{k}):    {metrics['recall'][k]:.4f}\n"
        result_str += f"  F1值 (F1@{k}):         {metrics['f1'][k]:.4f}\n"
        result_str += f"  成功率 (Success@{k}):   {metrics['success_rate'][k]:.4f}\n"
        result_str += "-" * 60 + "\n"
    
    return result_str

def save_results_to_file(result_str: str, output_file: str) -> None:
    """将评估结果保存到文件"""
    # 确保输出目录存在
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result_str)
        print(f"评估结果已保存到文件: {output_file}")
    except Exception as e:
        print(f"保存评估结果到文件时出错: {str(e)}")

def save_metrics_json(metrics: Dict, valid_repos: int, result_file: str, output_file: str, limit: int = None) -> None:
    """将评估指标以JSON格式保存"""
    try:
        # 准备数据结构，包含评估指标和元数据
        output_data = {
            "meta": {
                "evaluated_file": result_file,
                "valid_repositories": valid_repos,
                "data_limit": limit,
                "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "metrics": metrics
        }
        
        # 确保输出目录存在
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"评估指标已保存为JSON: {output_file}")
    except Exception as e:
        print(f"保存评估指标JSON时出错: {str(e)}")

def evaluate_file(result_file: str, k_values: List[int] = [3, 4, 5], save_to_file: bool = True, limit: int = None) -> Tuple[Dict, int]:
    """
    评估单个结果文件
    
    参数:
        result_file: 结果文件路径
        k_values: 要评估的K值列表
        save_to_file: 是否保存结果到文件
        limit: 限制处理的数据条数，None表示处理所有数据
    """
    print(f"评估文件: {result_file}" + (f" (限制前 {limit} 条数据)" if limit is not None else ""))
    results = load_results(result_file, limit)
    if not results:
        return {}, 0
        
    metrics, valid_repos = calculate_metrics(results, k_values)
    summary = format_results(metrics, valid_repos, limit)
    print(summary)
    
    if save_to_file:
        # 创建输出目录
        output_dir = Path("metrics")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 从输入文件名提取基本名称
        base_name = Path(result_file).stem
        
        # 添加limit信息到文件名
        limit_suffix = f"_limit{limit}" if limit is not None else ""
        
        # 创建文本报告文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_output_file = output_dir / f"{base_name}_evaluation{limit_suffix}_{timestamp}.txt"
        
        # 保存文本报告
        save_results_to_file(summary, str(txt_output_file))
        
        # 创建JSON指标文件名
        json_output_file = output_dir / f"{base_name}_metrics{limit_suffix}_{timestamp}.json"
        
        # 保存JSON指标
        save_metrics_json(metrics, valid_repos, result_file, str(json_output_file), limit)
    
    return metrics, valid_repos

def evaluate_directory(directory: str = "output", pattern: str = "topic_results_*.json", save_to_file: bool = True, limit: int = None) -> None:
    """
    评估目录中的所有结果文件
    
    参数:
        directory: 包含结果文件的目录
        pattern: 文件匹配模式
        save_to_file: 是否保存结果到文件
        limit: 限制处理的数据条数，None表示处理所有数据
    """
    directory_path = Path(directory)
    if not directory_path.exists():
        print(f"目录 '{directory}' 不存在")
        return
        
    result_files = list(directory_path.glob(pattern))
    if not result_files:
        print(f"在 '{directory}' 中没有找到匹配 '{pattern}' 的文件")
        return
        
    limit_info = f"（每个文件限制前 {limit} 条数据）" if limit is not None else ""
    print(f"找到 {len(result_files)} 个结果文件待评估{limit_info}")
    
    # 创建汇总报告
    summary_lines = [f"# 评估汇总报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{limit_info}", ""]
    
    for file_path in result_files:
        print("\n" + "=" * 80)
        metrics, valid_repos = evaluate_file(str(file_path), save_to_file=save_to_file, limit=limit)
        
        # 将此文件的摘要添加到汇总报告
        if metrics and valid_repos > 0:
            summary_lines.append(f"## {file_path.name} (有效仓库: {valid_repos})")
            for k in sorted(metrics['precision'].keys()):
                summary_lines.append(f"- Top-{k}: P={metrics['precision'][k]:.4f}, R={metrics['recall'][k]:.4f}, F1={metrics['f1'][k]:.4f}, SR={metrics['success_rate'][k]:.4f}")
            summary_lines.append("")
    
    # 保存汇总报告
    if save_to_file and summary_lines and len(result_files) > 1:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        limit_suffix = f"_limit{limit}" if limit is not None else ""
        summary_file = Path("metrics") / f"summary_report{limit_suffix}_{timestamp}.md"
        
        # 确保目录存在
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(summary_lines))
            print(f"\n汇总报告已保存到: {summary_file}")
        except Exception as e:
            print(f"保存汇总报告时出错: {str(e)}")

def main():
    """主函数"""
    print("GitHub 仓库 Topic 提取评估工具")
    print("-" * 60)
    
    # 创建评估结果目录
    Path("metrics").mkdir(parents=True, exist_ok=True)
    
    # 设置要使用的limit值列表
    limit_values = [500, 1000, 1500, 2000, 2500, 3000]
    
    # 获取output目录下的所有JSON文件（递归搜索子目录）
    output_dir = Path("output")
    json_files = list(output_dir.glob("**/*.json"))
    
    if not json_files:
        print(f"在 {output_dir} 及其子目录中没有找到JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件需要评估")
    
    # 对每个JSON文件使用不同的limit值进行评估
    for json_file in json_files:
        model_name = json_file.parent.name
        print(f"\n{'-' * 80}")
        print(f"处理模型 {model_name} 的文件: {json_file.name}")
        
        for limit in limit_values:
            print(f"\n使用limit={limit}进行评估:")
            evaluate_file(str(json_file), save_to_file=True, limit=limit)
        
        print(f"模型 {model_name} 的文件评估完成")
    
    print("\n所有文件评估完毕!")
    
    # 选择评估方式
    
    # 1. 评估特定的结果文件，并保存结果（限制前100条数据）
    # evaluate_file("output/qwen-plus-latest/qwen-plus-latest_topic_results_20250522_004058.json", save_to_file=True, limit=3000)
    
    # 2. 评估特定的结果文件，使用所有数据
    # evaluate_file("output/topic_results_20250521_133646.json", save_to_file=True)
    
    # 3. 评估output目录中的所有结果文件，并限制每个文件的数据条数
    # evaluate_directory(save_to_file=True, limit=100)
    
    # 4. 评估output目录中的所有结果文件，使用所有数据
    # evaluate_directory(save_to_file=True)
    
    # 5. 评估特定文件但不保存结果（仅控制台输出）
    # evaluate_file("output/topic_results_20250521_133646.json", save_to_file=False, limit=20)

if __name__ == "__main__":
    main()