# 长期记忆与动态上下文压缩

实现与参考核对日期：2026-10-28。面向当前单用户本地 CLI；不要求本地大模型、Embedding 服务或额外数据库服务器。

## 参考了哪些成熟做法

| 官方资料 | 采用的部分 | 当前适配边界 |
|---|---|---|
| [LangGraph Stores](https://docs.langchain.com/oss/python/langgraph/stores) | 将跨任务记忆与单次运行状态分开，按 namespace/key 存储 | 采用该数据组织方式，用标准库 SQLite 持久化；没有宣称实现 LangGraph BaseStore 或接入官方 PostgresStore |
| [LangChain SummarizationMiddleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in#summarization) | 达到阈值后总结旧历史，保留近期消息，避免拆开 AI 工具调用与工具返回 | 直接复用项目已固定的 langchain 1.3.9 中间件；为已有 StateGraph 增加模型调用前适配层，没有迁移到 create_agent |
| [Anthropic Context Engineering](https://www.anthropic.com/research/effective-context-engineering-for-ai-agents) | 压缩旧上下文、保留关键工作记录、原始资料保存在上下文之外 | 保留完整运行状态/MC 轨迹；压缩文本归档可按 ID 分页读取 |

`context_management.py` 覆盖中间件的两个内部扩展点：摘要生成与 MC 动作/观察分界。前者保证费用计数及错误传播，后者适配 MC 的 user observation 格式。升级 LangChain 时必须运行配对、预算、连续压缩测试。没有复制另一项目的整套 Agent，也没有调用 Anthropic 服务。

## 长期记忆的数据流

```text
本次问题 + 项目 namespace
    → 本地召回：最多5条用户显式偏好、3条相关历史研究
    → 完整记录总计不超过6000字符
    → 作为有时间戳的历史背景进入研究提纲与研究者/MC上下文
    → 重新获取本次证据并生成报告
    → 报告写盘成功后，保存有限的研究发现、缺口及来源索引
```

默认数据库：项目根目录 `.prism/memory.sqlite3`，已加入 `.gitignore`。SQLite 事务持久化，关闭进程后保留；各 namespace 独立。研究记录默认30天后不再被召回，每个 namespace 保留最新500条研究记录。用户明确保存的偏好默认不过期，可覆盖或删除。过期数据在后续写入时物理清理。

召回采用英文词项、中文双字片段的词面相关度，按原始研究问题匹配，没有 Embedding 调用费用，也不声称能理解所有同义改写。较长记录如果不能完整放进召回预算会跳过。不同研究结论并存，不自动认定较新结论必然正确。

只有结构及引用校验通过、包含发现的报告才写入研究记忆；MC 未完成的报告不写入。记录标明它是历史模型报告，未独立验证，不是事实数据库。记忆写入失败会记录 `memory_save_failed`，已生成的报告仍保留。

历史证据 ID **不会**加入本次报告的证据目录。召回的网页 URL 经过现有域名规则校验后可以重新读取；PDF/图片必须来自本次登记输入。当前任务优先于历史偏好；记忆不授予工具执行权限。只有 `memory remember` 明确提交的内容进入偏好类别，网页和论文内容不会自动变成用户偏好。

## 使用方式

常规 `research` 默认启用记忆与上下文压缩。建议给不同项目指定不同 namespace：

```powershell
prism-research memory remember --memory-namespace prism --key writing-style --text "用中文回答，区分证据与推测"
prism-research research --question "核验实验条件" --pdf study.pdf --pdf-backend api-rerank --memory-namespace prism --env-file PATH_TO_ENV
prism-research memory list --memory-namespace prism
prism-research memory forget --memory-namespace prism --key writing-style
```

`memory` 命令不调用模型，不需要 API Key。研究记录的 key 也可用 `forget` 删除。可通过 `--memory-db PATH` 指定另一数据库；指定路径时同一组操作需使用相同路径。namespace 是本地数据分区，不是多用户服务的认证授权机制。

研究时附加 `--no-memory` 可以完全关闭记忆读取和写入。真实联调脚本 `live_smoke.py`、`pdf_api_smoke.py` 已禁用长期记忆，避免合成样本混入真实项目。

## 动态压缩具体行为

- 在 Supervisor、Researcher、研究末尾的摘要模型、MC 下一步规划调用前检查消息。默认达到约12000 Token时尝试压缩；可以通过 `--context-trigger 8000` 调整，最小1000；`--no-context-compression` 关闭。
- 估算使用文本字符启发式，并单独计入图片估算量。它不是供应商精确 tokenizer，不是账单计量，也不包含绑定工具 schema 的精确开销。
- 系统消息和最初任务固定保留。旧历史替换成摘要，默认保留最近4条消息；工具配对或 MC 动作/观察配对需要时会保留更多。
- 摘要继续保留条件、来源编号、已执行操作和证据缺口。原始图像不会转成 Base64 文本交给摘要模型，只记录图片指纹；实际观察仍由原图工具完成。
- 相同历史前缀通过指纹复用旧摘要；再次压缩时包含上一份摘要，避免每轮重做。不同研究任务的缓存彼此分开。
- 每次运行最多2次摘要调用，使用本次研究模型，默认 Flash。每次摘要预留1次请求及最多1800输出 Token，计入原来的16次总请求上限，而非额外赠送的额度。
- 摘要失败直接传播，不将失败文字当作摘要；证据模式的末尾研究压缩也不自动重复调用。预算耗尽会停止。摘要未变短时不采用；原始消息不会被修改。
- 最近消息过大、没有可安全压缩的旧消息或已到摘要调用上限时，不强行删掉工具结果。模型消息估算超过触发阈值的3倍会明确中止。该限制只适用于上述消息历史调用，不是所有证据工具单次输入的统一窗口限制。

原始旧消息文本保存到 `runs/<run>/context/<hash>.json`，摘要中给出归档 ID。原生 Researcher 在后续轮次可调用 `read_context_archive`，每次读取最多6000字符；该工具只能访问当前运行登记过的 ID，不能读取任意路径。MC 保留完整本地对话轨迹，但目前没有单独的归档回读动作。

最终报告继续从本次证据目录独立生成，不把摘要视为新的证据。摘要可能损失细节，引用编号有效也不能自动证明语义正确。

## 验证与剩余边界

`python -m pytest -q` 包含跨运行召回、namespace 隔离、TTL、覆盖与删除、失败研究不写入、凭证脱敏、历史不进入证据目录、原始任务保留、工具并行返回配对、MC 动作观察配对、重复及连续压缩、预算扣减与错误传播等离线测试。

2026-10-28 完成111项离线测试。一次最终 Flash 合成样本验证将消息估算量从6539降至2501，保留初始任务及1440p条件，重复使用缓存未增加模型请求；本地数据库重开与 namespace 隔离也通过。该结果只验证工程链路与指定字段，不代表任意长任务的摘要质量。可复现命令（最多一次真实模型调用，需新输出目录）：

```powershell
python scripts/context_memory_smoke.py --env-file PATH_TO_ENV --output runs/check-memory-context
```

最终样本结果位于本地 `runs/memory-context-verified-20261028/summary.json`；旧消息文本和生成摘要分别保存在该运行的 `context/*.json` 与 `context/*.summary.json`。示例日期目录不提交版本控制。

本实现提供跨任务记忆，不提供中断研究的自动恢复。没有接入 LangGraph checkpointer；恢复不仅需要图状态，还需要同步恢复附件、证据目录和已消费预算，不能直接重放付费工具。多人服务可再迁移到带身份认证的持久 Store，规模变大后再评测语义召回与全文检索。当前没有证明这套记忆可以提高所有任务的准确率。

新增的 [交互式 CLI](chat.md) 另外保存会话历史、当前附件、报告路径及请求计数，支持退出后的对话恢复；这仍不是中断研究图的断点续跑。聊天复用本模块的压缩器，保留最近六条消息，每轮至多一次摘要调用，缓存随会话保存。上述最近四条消息、两次摘要上限描述的是单次完整研究配置。
