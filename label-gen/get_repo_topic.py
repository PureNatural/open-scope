import os
import csv
import time
import requests
import pandas as pd

GITHUB_TOKEN = ''
HEADERS = {"Accept": "application/vnd.github+json"}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"

def fetch_repo_metadata(repo_full_name):
    """获取单个仓库的 topics 和 description"""
    url = f"https://api.github.com/repos/{repo_full_name}"
    topics_url = f"{url}/topics"
    try:
        # 获取仓库信息
        repo_resp = requests.get(url, headers=HEADERS, timeout=10)
        repo_resp.raise_for_status()
        repo_data = repo_resp.json()
        description = repo_data.get("description", "")

        # 获取仓库 topics
        topics_resp = requests.get(
            topics_url,
            headers={**HEADERS, "Accept": "application/vnd.github.mercy-preview+json"},
            timeout=10
        )
        topics_resp.raise_for_status()
        topics_data = topics_resp.json()
        topics = ",".join(topics_data.get("names", []))

        return description, topics
    except Exception as e:
        print(f"Error fetching {repo_full_name}: {e}")
        return "", ""

def main(input_csv, output_csv):
    # 读取输入 CSV
    df = pd.read_csv(input_csv)
    github_df = df[df["platform"].str.lower() == "github"].copy()
    total = len(github_df)

    # 打开输出文件，写表头
    fieldnames = list(github_df.columns) + ["description", "topics"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # 串行逐条处理
        for i, row in github_df.iterrows():
            description, topics = fetch_repo_metadata(row["repo_name"])
            row_dict = row.to_dict()
            row_dict["description"] = description
            row_dict["topics"] = topics
            writer.writerow(row_dict)

            print(f"[进度] 已完成 {i+1} / {total}")

            # 为了避免速率限制，每次请求间隔 1 秒（可调）
            time.sleep(1)

    print(f"Done! Saved to {output_csv}")

if __name__ == "__main__":
    main("input.csv", "output.csv")
