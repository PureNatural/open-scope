import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Tuple
import ast
import re
from collections import defaultdict, Counter

class RepositoryFailureAnalyzer:
    def __init__(self, data_dir="data", output_dir="output"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.original_data = self.load_original_data()
        self.model_results = self.load_model_results()
        
    def load_original_data(self) -> Dict:
        """加载原始仓库数据"""
        repositories = {}
        for json_file in self.data_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for key in data:
                        for repo in data[key]:
                            repo_id = repo.get('r.repo_id')
                            if repo_id:
                                repositories[repo_id] = repo
            except Exception as e:
                print(f"加载原始数据文件 {json_file} 时出错: {str(e)}")
        print(f"加载了 {len(repositories)} 个原始仓库数据")
        return repositories
    
    def load_model_results(self) -> Dict:
        """加载各模型的预测结果"""
        model_results = {}
        
        for model_dir in self.output_dir.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name
                # 查找最新的结果文件
                json_files = list(model_dir.glob("*.json"))
                if json_files:
                    latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
                    try:
                        with open(latest_file, 'r', encoding='utf-8') as f:
                            results = json.load(f)
                            model_results[model_name] = {
                                item['repo_id']: item for item in results
                            }
                        print(f"加载模型 {model_name} 的结果: {len(results)} 个仓库")
                    except Exception as e:
                        print(f"加载模型 {model_name} 结果时出错: {str(e)}")
        
        return model_results
    
    def parse_topics(self, topics_str: str) -> Set[str]:
        """解析topics字符串为集合"""
        if not topics_str or topics_str == '':
            return set()
            
        try:
            topics = ast.literal_eval(topics_str)
            if isinstance(topics, set):
                return topics
            elif isinstance(topics, list):
                return set(topics)
            else:
                return {str(topics)}
        except (SyntaxError, ValueError):
            pattern = r'[\'\"]([^\'\"]*)[\'\"]'
            matches = re.findall(pattern, topics_str)
            return set(matches)
    
    def calculate_success_for_repo(self, repo_id: int, model_name: str, k: int = 5) -> bool:
        """计算单个仓库在特定模型下的成功率"""
        if repo_id not in self.original_data or repo_id not in self.model_results.get(model_name, {}):
            return False
            
        original_repo = self.original_data[repo_id]
        predicted_repo = self.model_results[model_name][repo_id]
        
        # 跳过有错误的预测
        if 'error' in predicted_repo:
            return False
            
        # 解析原始topics
        original_topics_str = original_repo.get('a.topics', '')
        original_topics = self.parse_topics(original_topics_str)
        
        if not original_topics:
            return False
            
        # 获取预测topics的前k个
        suggested_topics = predicted_repo.get('suggested_topics', [])
        top_k_topics = set(suggested_topics[:min(k, len(suggested_topics))])
        
        # 计算交集
        hits = len(top_k_topics.intersection(original_topics))
        return hits > 0
    
    def find_failure_patterns(self, k: int = 5) -> Dict:
        """分析失败模式"""
        # 获取所有模型名称
        model_names = list(self.model_results.keys())
        print(f"分析模型: {model_names}")
        
        # 获取所有有效的仓库ID（在所有模型结果中都存在）
        all_repo_ids = set(self.original_data.keys())
        for model_name in model_names:
            model_repo_ids = set(self.model_results[model_name].keys())
            all_repo_ids = all_repo_ids.intersection(model_repo_ids)
        
        print(f"共有 {len(all_repo_ids)} 个仓库在所有模型中都有预测结果")
        
        # 分类仓库
        all_fail_repos = []  # 所有模型都失败
        claude_deepseek_only_success = []  # 只有Claude和DeepSeek成功
        claude_only_success = []  # 只有Claude成功
        deepseek_only_success = []  # 只有DeepSeek成功
        
        for repo_id in all_repo_ids:
            # 检查原始topics是否存在
            original_topics_str = self.original_data[repo_id].get('a.topics', '')
            original_topics = self.parse_topics(original_topics_str)
            if not original_topics:
                continue
                
            # 计算各模型的成功情况
            success_status = {}
            for model_name in model_names:
                success_status[model_name] = self.calculate_success_for_repo(repo_id, model_name, k)
            
            # 统计成功的模型数量
            successful_models = [model for model, success in success_status.items() if success]
            
            if len(successful_models) == 0:
                all_fail_repos.append((repo_id, success_status))
            elif set(successful_models) == {'claude-3-7-sonnet-20250219', 'deepseek-v3'}:
                claude_deepseek_only_success.append((repo_id, success_status))
            elif successful_models == ['claude-3-7-sonnet-20250219']:
                claude_only_success.append((repo_id, success_status))
            elif successful_models == ['deepseek-v3']:
                deepseek_only_success.append((repo_id, success_status))
        
        return {
            'all_fail': all_fail_repos,
            'claude_deepseek_only': claude_deepseek_only_success,
            'claude_only': claude_only_success,
            'deepseek_only': deepseek_only_success,
            'total_analyzed': len(all_repo_ids)
        }
    
    def analyze_repository_characteristics(self, repo_ids: List[int]) -> Dict:
        """分析仓库特征"""
        characteristics = {
            'total_repos': len(repo_ids),
            'avg_description_length': 0,
            'avg_readme_length': 0,
            'avg_topics_count': 0,
            'language_distribution': Counter(),
            'topic_patterns': Counter(),
            'has_description': 0,
            'has_readme': 0,
            'description_lengths': [],
            'readme_lengths': [],
            'topics_counts': []
        }
        
        for repo_id in repo_ids:
            if repo_id not in self.original_data:
                continue
                
            repo = self.original_data[repo_id]
            
            # 描述分析
            description = repo.get('a.description', '') or ''
            if description.strip():
                characteristics['has_description'] += 1
                characteristics['description_lengths'].append(len(description))
            
            # README分析
            readme = repo.get('a.readme_text', '') or ''
            if readme.strip():
                characteristics['has_readme'] += 1
                characteristics['readme_lengths'].append(len(readme))
            
            # Topics分析
            topics_str = repo.get('a.topics', '')
            topics = self.parse_topics(topics_str)
            characteristics['topics_counts'].append(len(topics))
            
            # 语言分析
            language = repo.get('a.language', '') or 'Unknown'
            characteristics['language_distribution'][language] += 1
            
            # Topics模式分析
            for topic in topics:
                characteristics['topic_patterns'][topic] += 1
        
        # 计算平均值
        if characteristics['description_lengths']:
            characteristics['avg_description_length'] = sum(characteristics['description_lengths']) / len(characteristics['description_lengths'])
        
        if characteristics['readme_lengths']:
            characteristics['avg_readme_length'] = sum(characteristics['readme_lengths']) / len(characteristics['readme_lengths'])
            
        if characteristics['topics_counts']:
            characteristics['avg_topics_count'] = sum(characteristics['topics_counts']) / len(characteristics['topics_counts'])
        
        return characteristics
    
    def analyze_prediction_patterns(self, repo_ids: List[int], model_names: List[str]) -> Dict:
        """分析预测模式"""
        patterns = {
            'common_predictions': Counter(),
            'model_specific_patterns': {model: Counter() for model in model_names},
            'prediction_lengths': {model: [] for model in model_names}
        }
        
        for repo_id in repo_ids:
            for model_name in model_names:
                if repo_id in self.model_results.get(model_name, {}):
                    result = self.model_results[model_name][repo_id]
                    if 'error' not in result:
                        suggested_topics = result.get('suggested_topics', [])
                        patterns['prediction_lengths'][model_name].append(len(suggested_topics))
                        
                        for topic in suggested_topics[:5]:  # 分析前5个预测
                            patterns['common_predictions'][topic] += 1
                            patterns['model_specific_patterns'][model_name][topic] += 1
        
        return patterns
    
    def analyze_claude_deepseek_predictions(self, repo_ids: List[int]) -> Dict:
        """专门分析Claude和DeepSeek的预测模式"""
        analysis = {
            'claude_predictions': Counter(),
            'deepseek_predictions': Counter(),
            'shared_predictions': Counter(),
            'claude_unique': Counter(),
            'deepseek_unique': Counter(),
            'prediction_overlap_stats': {
                'total_repos': len(repo_ids),
                'repos_with_shared_predictions': 0,
                'avg_overlap_rate': 0,
                'overlap_rates': []
            }
        }
        
        for repo_id in repo_ids:
            if repo_id not in self.original_data:
                continue
                
            claude_result = self.model_results.get('claude-3-7-sonnet-20250219', {}).get(repo_id, {})
            deepseek_result = self.model_results.get('deepseek-v3', {}).get(repo_id, {})
            
            if 'error' in claude_result or 'error' in deepseek_result:
                continue
                
            claude_topics = set(claude_result.get('suggested_topics', [])[:5])
            deepseek_topics = set(deepseek_result.get('suggested_topics', [])[:5])
            
            # 统计各模型预测
            for topic in claude_topics:
                analysis['claude_predictions'][topic] += 1
            
            for topic in deepseek_topics:
                analysis['deepseek_predictions'][topic] += 1
            
            # 统计共同预测和独有预测
            shared = claude_topics.intersection(deepseek_topics)
            claude_unique = claude_topics - deepseek_topics
            deepseek_unique = deepseek_topics - claude_topics
            
            for topic in shared:
                analysis['shared_predictions'][topic] += 1
            
            for topic in claude_unique:
                analysis['claude_unique'][topic] += 1
                
            for topic in deepseek_unique:
                analysis['deepseek_unique'][topic] += 1
            
            # 计算重叠率
            if claude_topics or deepseek_topics:
                if shared:
                    analysis['prediction_overlap_stats']['repos_with_shared_predictions'] += 1
                
                total_unique = len(claude_topics.union(deepseek_topics))
                if total_unique > 0:
                    overlap_rate = len(shared) / total_unique
                    analysis['prediction_overlap_stats']['overlap_rates'].append(overlap_rate)
        
        # 计算平均重叠率
        if analysis['prediction_overlap_stats']['overlap_rates']:
            analysis['prediction_overlap_stats']['avg_overlap_rate'] = sum(
                analysis['prediction_overlap_stats']['overlap_rates']
            ) / len(analysis['prediction_overlap_stats']['overlap_rates'])
        
        return analysis
    
    def generate_detailed_analysis_report(self, failure_patterns: Dict) -> str:
        """生成详细分析报告"""
        report = []
        report.append("# GitHub仓库主题预测失败模式分析报告\n")
        
        # 总体统计
        report.append("## 总体统计")
        report.append(f"- 分析仓库总数: {failure_patterns['total_analyzed']}")
        report.append(f"- 所有模型均失败: {len(failure_patterns['all_fail'])} ({len(failure_patterns['all_fail'])/failure_patterns['total_analyzed']*100:.2f}%)")
        report.append(f"- 仅Claude和DeepSeek成功: {len(failure_patterns['claude_deepseek_only'])} ({len(failure_patterns['claude_deepseek_only'])/failure_patterns['total_analyzed']*100:.2f}%)")
        report.append(f"- 仅Claude成功: {len(failure_patterns['claude_only'])} ({len(failure_patterns['claude_only'])/failure_patterns['total_analyzed']*100:.2f}%)")
        report.append(f"- 仅DeepSeek成功: {len(failure_patterns['deepseek_only'])} ({len(failure_patterns['deepseek_only'])/failure_patterns['total_analyzed']*100:.2f}%)\n")
        
        # 分析各类仓库的特征
        categories = [
            ('all_fail', '所有模型均失败的仓库'),
            ('claude_deepseek_only', '仅Claude和DeepSeek成功的仓库'),
            ('claude_only', '仅Claude成功的仓库'),
            ('deepseek_only', '仅DeepSeek成功的仓库')
        ]
        
        for category_key, category_name in categories:
            if not failure_patterns[category_key]:
                continue
                
            report.append(f"## {category_name}特征分析")
            
            repo_ids = [repo_id for repo_id, _ in failure_patterns[category_key]]
            characteristics = self.analyze_repository_characteristics(repo_ids)
            
            report.append(f"### 基本统计")
            report.append(f"- 仓库数量: {characteristics['total_repos']}")
            report.append(f"- 包含描述的仓库: {characteristics['has_description']} ({characteristics['has_description']/characteristics['total_repos']*100:.1f}%)")
            report.append(f"- 包含README的仓库: {characteristics['has_readme']} ({characteristics['has_readme']/characteristics['total_repos']*100:.1f}%)")
            report.append(f"- 平均描述长度: {characteristics['avg_description_length']:.0f} 字符")
            report.append(f"- 平均README长度: {characteristics['avg_readme_length']:.0f} 字符")
            report.append(f"- 平均主题标签数量: {characteristics['avg_topics_count']:.1f}")
            
            # 语言分布
            report.append(f"\n### 编程语言分布")
            top_languages = characteristics['language_distribution'].most_common(10)
            for lang, count in top_languages:
                report.append(f"- {lang}: {count} ({count/characteristics['total_repos']*100:.1f}%)")
            
            # 主题模式
            report.append(f"\n### 常见主题标签")
            top_topics = characteristics['topic_patterns'].most_common(15)
            for topic, count in top_topics:
                report.append(f"- {topic}: {count} ({count/characteristics['total_repos']*100:.1f}%)")
            
            # 如果是Claude和DeepSeek共同成功的情况，添加预测模式分析
            if category_key == 'claude_deepseek_only':
                claude_deepseek_analysis = self.analyze_claude_deepseek_predictions(repo_ids)
                
                report.append(f"\n### Claude和DeepSeek预测模式分析")
                report.append(f"- 有共同预测的仓库数量: {claude_deepseek_analysis['prediction_overlap_stats']['repos_with_shared_predictions']}")
                report.append(f"- 平均预测重叠率: {claude_deepseek_analysis['prediction_overlap_stats']['avg_overlap_rate']:.3f}")
                
                report.append(f"\n#### 共同预测的热门标签")
                for topic, count in claude_deepseek_analysis['shared_predictions'].most_common(10):
                    report.append(f"- {topic}: {count} 次")
                
                report.append(f"\n#### Claude独有预测的热门标签")
                for topic, count in claude_deepseek_analysis['claude_unique'].most_common(10):
                    report.append(f"- {topic}: {count} 次")
                
                report.append(f"\n#### DeepSeek独有预测的热门标签")
                for topic, count in claude_deepseek_analysis['deepseek_unique'].most_common(10):
                    report.append(f"- {topic}: {count} 次")
            
            # 具体案例分析
            report.append(f"\n### 典型案例")
            for i, (repo_id, success_status) in enumerate(failure_patterns[category_key][:5]):  # 增加到5个案例
                if repo_id in self.original_data:
                    repo = self.original_data[repo_id]
                    repo_name = repo.get('b.repo_name', 'Unknown')
                    description = repo.get('a.description', '')[:200] + '...' if repo.get('a.description', '') else '无描述'
                    topics = self.parse_topics(repo.get('a.topics', ''))
                    
                    report.append(f"\n#### 案例 {i+1}: {repo_name}")
                    report.append(f"- 仓库ID: {repo_id}")
                    report.append(f"- 描述: {description}")
                    report.append(f"- 原始主题: {list(topics)}")
                    report.append(f"- 各模型成功状态: {success_status}")
                    
                    # 如果是Claude和DeepSeek成功的案例，显示它们的具体预测
                    if category_key == 'claude_deepseek_only':
                        claude_result = self.model_results.get('claude-3-7-sonnet-20250219', {}).get(repo_id, {})
                        deepseek_result = self.model_results.get('deepseek-v3', {}).get(repo_id, {})
                        
                        if 'error' not in claude_result and 'error' not in deepseek_result:
                            claude_predictions = claude_result.get('suggested_topics', [])[:5]
                            deepseek_predictions = deepseek_result.get('suggested_topics', [])[:5]
                            
                            report.append(f"- Claude预测: {claude_predictions}")
                            report.append(f"- DeepSeek预测: {deepseek_predictions}")
                            
                            # 分析命中的标签
                            claude_hits = set(claude_predictions).intersection(topics)
                            deepseek_hits = set(deepseek_predictions).intersection(topics)
                            
                            report.append(f"- Claude命中标签: {list(claude_hits)}")
                            report.append(f"- DeepSeek命中标签: {list(deepseek_hits)}")
            
            report.append("\n" + "="*50 + "\n")
        
        return "\n".join(report)
    
    def save_detailed_case_analysis(self, failure_patterns: Dict, output_file: str = "detailed_failure_analysis.json"):
        """保存详细的案例分析数据"""
        detailed_data = {}
        
        categories = ['all_fail', 'claude_deepseek_only', 'claude_only', 'deepseek_only']
        
        for category in categories:
            detailed_data[category] = []
            
            for repo_id, success_status in failure_patterns[category]:
                if repo_id not in self.original_data:
                    continue
                    
                repo_data = self.original_data[repo_id].copy()
                
                # 添加预测结果
                predictions = {}
                for model_name in self.model_results:
                    if repo_id in self.model_results[model_name]:
                        model_result = self.model_results[model_name][repo_id]
                        predictions[model_name] = {
                            'suggested_topics': model_result.get('suggested_topics', []),
                            'has_error': 'error' in model_result,
                            'error': model_result.get('error', None)
                        }
                
                case_data = {
                    'repo_id': repo_id,
                    'repo_name': repo_data.get('b.repo_name', ''),
                    'description': repo_data.get('a.description', ''),
                    'language': repo_data.get('a.language', ''),
                    'original_topics': repo_data.get('a.topics', ''),
                    'readme_length': len(repo_data.get('a.readme_text', '')),
                    'success_status': success_status,
                    'predictions': predictions
                }
                
                detailed_data[category].append(case_data)
        
        # 保存到JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, ensure_ascii=False, indent=2)
        
        print(f"详细案例分析数据已保存到: {output_file}")
    
    def run_complete_analysis(self, k: int = 5):
        """运行完整分析"""
        print("开始分析GitHub仓库主题预测失败模式...")
        
        # 查找失败模式
        failure_patterns = self.find_failure_patterns(k)
        
        # 生成分析报告
        report = self.generate_detailed_analysis_report(failure_patterns)
        
        # 保存报告
        with open("failure_analysis_report.md", 'w', encoding='utf-8') as f:
            f.write(report)
        print("分析报告已保存到: failure_analysis_report.md")
        
        # 保存详细数据
        self.save_detailed_case_analysis(failure_patterns)
        
        return failure_patterns

def main():
    analyzer = RepositoryFailureAnalyzer()
    failure_patterns = analyzer.run_complete_analysis()
    
    print("\n分析完成！")
    print(f"所有模型均失败的仓库数量: {len(failure_patterns['all_fail'])}")
    print(f"仅Claude和DeepSeek成功的仓库数量: {len(failure_patterns['claude_deepseek_only'])}")
    print(f"仅Claude成功的仓库数量: {len(failure_patterns['claude_only'])}")
    print(f"仅DeepSeek成功的仓库数量: {len(failure_patterns['deepseek_only'])}")

if __name__ == "__main__":
    main()