import pandas as pd
from collections import Counter

# 读取带有 domain 字段的文件
df = pd.read_csv("output_with_domain_v3.csv")

counter = Counter()

for domains_str in df["domain"].dropna():
    for d in domains_str.split(","):
        counter[d] += 1

# 转换为 DataFrame 方便查看和保存
stats_df = pd.DataFrame(counter.items(), columns=["domain", "count"]).sort_values(by="count", ascending=False)

print(stats_df)

stats_df.to_csv("domain_stats.csv", index=False)
