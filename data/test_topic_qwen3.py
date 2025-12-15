import json
import os
import concurrent.futures
from pathlib import Path
import time
import re
from openai import OpenAI

# 初始化OpenAI客户端
client = OpenAI(
    # base_url='https://xiaoai.plus/v1',
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    # api_key='sk-jTsJsfqxQRUahM9sdEepWh3JluKmwh4gd4F3QBr4BVqgbnlQ'
    api_key='sk-65ded426d36f42b4b0277408baad14c1'
)

def load_repositories_from_json(data_dir="data"):
    """从data目录加载JSON格式的仓库数据"""
    repositories = []
    data_path = Path(data_dir)
    
    # 确保data目录存在
    if not data_path.exists():
        print(f"错误: 目录 '{data_dir}' 不存在")
        return []
    
    # 遍历目录中的所有JSON文件
    for json_file in data_path.glob("*.json"):
        print(f"加载文件: {json_file}")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # JSON数据有特殊的格式，键是一个SQL查询字符串
                for key in data:
                    repositories.extend(data[key])
            print(f"从 {json_file} 加载了 {len(data[list(data.keys())[0]])} 个仓库")
        except Exception as e:
            print(f"加载文件 {json_file} 时出错: {str(e)}")
    
    print(f"总共加载了 {len(repositories)} 个仓库")
    return repositories

def extract_topics(repo, model_name, max_retries=3, retry_delay=5):
    """使用大模型API为仓库提取话题"""
    repo_name = repo.get('b.repo_name', 'Unknown')
    repo_id = repo.get('r.repo_id', 'Unknown')
    description = repo.get('a.description', '')
    
    # 获取README内容并截断，避免超出API上下文窗口限制
    readme = repo.get('a.readme_text', '')
    max_readme_length = 30000  # 根据模型上下文窗口大小调整
    if len(readme) > max_readme_length:
        readme = readme[:max_readme_length] + "...(内容已截断)"
    
    # 准备提示词
    prompt = f"""
    我有一个GitHub仓库，信息如下:
    仓库名: {repo_name}
    描述: {description}
    
    README内容:
    {readme}
    
    请根据上述信息，为这个仓库提取适合的GitHub topics（主题标签）。
    topics应该是相关的技术类别、框架、技术或领域。
    请只返回一个JSON格式的主题数组，例如: ["python", "machine-learning", "data-science"]
    """
    
    # 使用重试机制处理API调用
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "你是一个专注于GitHub仓库分类的助手。请分析仓库信息并提供相关主题标签。"},
                    {"role": "user", "content": prompt}
                ],
            )
            
            # 解析API响应
            response_text = completion.choices[0].message.content
            
            # 尝试从响应中提取JSON数组
            try:
                # 直接尝试解析为JSON
                if response_text.strip().startswith('['):
                    suggested_topics = json.loads(response_text)
                else:
                    # 尝试通过正则表达式找到并提取JSON数组
                    json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                    if json_match:
                        suggested_topics = json.loads(json_match.group(0))
                    else:
                        # 如果没有找到JSON格式，回退到按逗号分割
                        suggested_topics = [topic.strip(' "\'[]') for topic in response_text.strip().split(',')]
            except json.JSONDecodeError:
                # 如果JSON解析失败，按逗号分割响应
                suggested_topics = [topic.strip(' "\'[]') for topic in response_text.strip().split(',')]
            
            # 返回结果
            return {
                'repo_id': repo_id,
                'repo_name': repo_name,
                'original_topics': repo.get('a.topics', ''),
                'suggested_topics': suggested_topics
            }
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"处理仓库 {repo_name} 时出错: {str(e)}. 将在 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                print(f"处理仓库 {repo_name} 失败，已达到最大重试次数: {str(e)}")
                return {
                    'repo_id': repo_id,
                    'repo_name': repo_name,
                    'original_topics': repo.get('a.topics', ''),
                    'suggested_topics': [],
                    'error': str(e)
                }

def process_repositories_concurrently(repositories, model_name, max_workers=10, max_repos=None):
    """使用并发处理多个仓库以提高效率"""
    results = []
    
    # 如果指定了最大处理仓库数，则进行限制
    if max_repos is not None:
        repositories = repositories[:max_repos]
    
    total = len(repositories)
    print(f"将使用模型 {model_name} 并发处理 {total} 个仓库，使用 {max_workers} 个工作线程")
    
    start_time = time.time()
    
    # 使用ThreadPoolExecutor进行并发处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_repo = {executor.submit(extract_topics, repo, model_name): repo for repo in repositories}
        
        # 处理完成的结果
        completed = 0
        for future in concurrent.futures.as_completed(future_to_repo):
            repo = future_to_repo[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"处理仓库 {repo.get('b.repo_name', 'Unknown')} 时发生未处理的异常: {str(e)}")
                results.append({
                    'repo_id': repo.get('r.repo_id', 'Unknown'),
                    'repo_name': repo.get('b.repo_name', 'Unknown'),
                    'error': str(e)
                })
            
            completed += 1
            # 定期打印进度
            if completed % 5 == 0 or completed == total:
                elapsed = time.time() - start_time
                remaining = (elapsed / completed) * (total - completed) if completed > 0 else 0
                print(f"已处理: {completed}/{total} ({completed/total*100:.1f}%) "
                      f"用时: {elapsed:.1f}秒 "
                      f"速率: {completed/elapsed:.2f}个/秒 "
                      f"预计剩余时间: {remaining:.1f}秒")
    
    # 计算总用时
    total_time = time.time() - start_time
    print(f"所有仓库处理完成! 总用时: {total_time:.2f}秒, 平均每个仓库: {total_time/total:.2f}秒")
    
    return results

def save_results(results, output_file="topic_results.json"):
    """将结果保存到JSON文件"""
    # 确保输出目录存在
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到 {output_file}")
    
    # 生成简单的统计
    success_count = sum(1 for r in results if 'error' not in r)
    error_count = sum(1 for r in results if 'error' in r)
    print(f"成功: {success_count}, 失败: {error_count}, 总计: {len(results)}")

def main():
    """主函数，协调整个处理流程"""
    print("开始GitHub仓库topic提取任务...")
    
    # 选择要使用的模型
    # 可选模型: "gpt-4o", "deepseek-v3", "claude-3-7-sonnet-20250219", "gemini-2.0-flash", "qwen3-235b-a22b"
    model_name = "qwen-plus-latest"
    
    # 1. 加载仓库数据
    repositories = load_repositories_from_json()
    if not repositories:
        print("没有找到仓库数据，程序退出")
        return
    
    # 2. 使用并发处理仓库数据
    results = process_repositories_concurrently(
        repositories,
        model_name=model_name,
        max_workers=10,  # 根据API限制和系统性能调整
        max_repos=3000   # 设置为None处理所有仓库，或指定数字进行测试
    )
    
    # 3. 创建模型特定的输出目录
    model_output_dir = f"output/{model_name}"
    Path(model_output_dir).mkdir(parents=True, exist_ok=True)
    
    # 4. 保存结果到模型特定目录
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_results(results, f"{model_output_dir}/{model_name}_topic_results_{timestamp}.json")
    
    print(f"任务完成! 使用模型: {model_name}")

if __name__ == "__main__":
    main()