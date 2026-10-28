# 商品研究、购买来源与性价比

`--shopping` 在现有 Open Deep Research 的 Supervisor → Researcher → 证据报告流程中加入商品研究工具。输入可以是一张商品图片，也可以只有型号和问题；不要求 PDF。默认市场为中国大陆、预算单位为人民币。

## 本地运行

已有虚拟环境和百炼配置时，在 PowerShell 中运行：

```powershell
Set-Location D:\agent\prism-research
.\.venv\Scripts\prism-research.exe research --shopping --question "研究 AirPods 5 的主要功能、购买来源与性价比" --usage "使用苹果手机，主要用于日常通勤" --purchase-budget 1200 --env-file "..\open_deep_research\.env" --memory-namespace shopping
```

以图片为起点时，在上述命令后加 `--image "D:\你的图片目录\product.jpg"`，并将问题改为“识别图片中的商品，研究功能、购买来源与性价比”。图片路径必须存在；型号和版本看不清时，报告应保留待确认项。可以在问题中提供预设网站内的商品链接，以提高找到对应来源的机会。

`--shopping` 自动启用网页研究，不需要另加 `--web`。运行结束后打开终端打印的 `report.html`。同目录保留：

| 文件 | 查看内容 |
|---|---|
| `report.html` / `report.json` | 研究发现、购买来源总结、性价比分析、报价表和证据 |
| `shopping_result.json` | 型号识别、预算用途、候选链接、逐页状态、提取报价与缺口 |
| `events.jsonl` | 实际工具调用、API 用量、失败和校验拒绝记录 |
| `page-*.html` / `page-*.txt` | 当次抓取的网页快照和提取文字 |
| `budget.json` | 请求与搜索次数账本；不是人民币账单 |

默认写入新的 `runs/` 子目录。`--output` 可指定新的输出目录。使用 `--no-memory` 可以关闭本次长期记忆读写；历史记忆仅提供检索线索，旧报价需要重新读取来源。

## 网站范围

预设来源包括京东、淘宝、天猫、苏宁、拼多多的指定商品域名，以及 Apple、小米、华为、联想、华硕、Dell、HP、三星的指定官网域名。准确列表是 `src/open_deep_research/shopping.py` 中的 `SHOPPING_HOSTS`，不支持任意域名或通配符。

用重复参数缩小本次商品搜索范围，例如：

```powershell
--shopping-site item.jd.com --shopping-site detail.tmall.com --shopping-site www.apple.com.cn
```

域名白名单表示允许检索、读取，并不保证平台开放数据或搜索索引覆盖。京东、淘宝等可能要求登录、执行 JavaScript 或触发验证；当前工具通过 HTTP 读取公开页面，未接入平台授权商品 API，也没有绕过登录或验证。遇到这种情况保留候选链接和未取得报价的原因。原有技术文档网站的读取范围仍保留；普通研究不会获得电商网站权限。重定向目的域名同样受检查。

## 一次研究怎样执行

1. 使用现有视觉/语言模型，从问题与图片中提取品牌、型号、版本和不确定信息。
2. 搜索购买页面，再搜索参数、限制和对比线索。默认最多两次百炼搜索，按网站轮流选择最多六个候选页面。
3. 读取页面正文和有界的 Product JSON-LD。搜索摘要只用于发现链接，不当作当前价格凭据。
4. 结构化提取报价，核对引用段落与金额是否确实存在于归档原文。单条失败不会丢弃其他通过校验的报价。
5. 在原报告中加入购买来源和性价比总结，结合预算、用途、参数、版本以及证据缺口给出判断。

购买研究工具每次运行只执行一遍，重复调用复用当次结果。`--max-searches` 是搜索次数上限（1～4），不是保证执行的次数；当前购买工具最多发起两条查询。原生 Agent 仍可以在工具额度内进一步阅读已知来源。实验性的 `--planner mcsearch` 也接入同一购买工具；真实联调目前使用默认 `native` 编排。

## 价格和结论的含义

- “页面展示价格”表示抓取时页面声明的金额，带来源、抓取时间和内容哈希；时间记录使用 UTC。未核验登录后结算价、库存、运费、个人优惠券和地区补贴。
- 标价、起价、分期、区间和优惠条件分别标注。分期对应版本待核实，不能当作商品总价；不同配置不直接比价。
- 新闻、发布稿等参考页面中的价格标为参考价，不能当作当前购买报价。
- 同款匹配由模型判断，仍需核对型号、容量、成色与销售条件。来源和数字校验不等于全部语义正确，也不构成性能实测。
- 找不到可用报价或对比性能证据时，明确说明无法判断性价比；不输出“全网最低价”。

模型总结中的具体金额必须对应被引用的已校验报价，或用户明确给出的预算。失败时最多纠正一次，仍计入原请求预算；再次失败或预算不足时输出“总结未生成”，仅保留来源与报价。这样的证据保留报告标记 `generation_status=evidence_only`，不算完整联调成功。

## 模型、预算和验证

继续使用已有百炼密钥：识别、提取与总结默认 `qwen3.5-flash`，搜索使用支持站点限制的 `qwen-flash`。不需要本地大模型或额外电商 API 密钥。输入图片会发往视觉模型；搜索使用生成的文字查询，这不是电商平台原生“以图搜同款”。

默认整次运行最多 16 次 API 请求，含最多 2 次搜索；身份识别、报价提取和可能的报告纠正均计入预算。费用取决于模型输入输出和搜索计费，代码中的次数上限不等于固定金额。先使用小样本调试，不要把整个商品目录一次输入。

离线验证：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

付费小样本联调（每次选择新的输出目录）：

```powershell
.\.venv\Scripts\python.exe scripts/shopping_smoke.py --env-file "..\open_deep_research\.env" --output runs/shopping-check-new
```

脚本用明确标记的合成商品标签测试图片输入链路，实际调用模型、搜索和公开网页。要求生成经过校验的购买及性价比总结，不接受仅保留证据的降级报告。它不写长期记忆，也不下单；其通过不能证明真实商品照片识别准确率或所有电商平台可用。

实现参考：[百炼联网搜索与站点限制](https://help.aliyun.com/zh/model-studio/web-search)、[Schema.org Offer](https://schema.org/Offer)、[Google Product 结构化数据](https://developers.google.com/search/docs/appearance/structured-data/product-snippet)。结构化商品数据仍属于页面声明。
