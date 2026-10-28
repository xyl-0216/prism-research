# 运行配置

## 输入与工具

`research` 接收一个 PNG/JPEG（最大 6 MB、1600 万像素）和/或一个 PDF（最大 20 MB；URA 最多 100 页，API 重排最多 40 页）。PDF 的文字、词级坐标、内嵌图片和向量绘图区域通过 PyMuPDF 提取；页面采用归一化 `[x0,y0,x1,y1]` 坐标。带旋转的页面需先归一化。页面检索可选择原版 URA 或多模态重排 API，不提供词元匹配回退。扫描页仍可使用图像区域，但当前未加入 OCR，不能假定其检索质量等同有文字层的页面。

`--web` 使用百炼 Qwen3.5 搜索，原文读取沿用明确的来源域名范围：docs.langchain.com、openai.github.io、learn.microsoft.com、www.nvidia.com、images.nvidia.com。需要其他来源时，应在 `evidence_tools.py` 中审查并扩展来源列表。`--web-screenshot "HTTPS_URL=LOCAL_PATH"` 可将本地网页截图关联到候选 URL；截图必须由使用者事先准备。

`--shopping` 会为本次运行增加预设商品网站范围，并自动启用网页研究，默认最多两次搜索。可用 `--shopping-site` 缩小范围；购买来源、页面报价和性价比流程见[商品研究说明](shopping.md)。下文“一次网页搜索”的默认限制指普通网页模式。

CLI 默认使用 `openai:qwen3.5-flash` 进行低成本调试；`--model openai:MODEL_NAME` 可覆盖，例如切回 `openai:qwen3.5-122b-a10b` 做效果对照。输入图片和选中 PDF 页面会发往该模型接口；本地 `.env` 不会写入报告。网页模式使用百炼接口：规划与阅读仍用所选 Qwen3.5，搜索单独用 `qwen-flash` 的原生文本接口，以支持 `assigned_site_list` 来源限制。

默认启用本地跨任务记忆和动态上下文压缩。`--memory-namespace` 隔离项目，`--no-memory` 关闭记忆读写；`--context-trigger` 调整压缩阈值，`--no-context-compression` 关闭压缩。单次最多两次新增摘要调用，同样消耗总请求预算。历史记录不能直接作为本次证据。详见[实现来源、配置和限制](memory-context.md)。

## PDF API 检索（无需本地权重）

```bash
prism-research research --question "实验采用了什么条件？" --pdf study.pdf --pdf-backend api-rerank --model openai:qwen3.5-flash --env-file PATH_TO_ENV --max-requests 16
```

`--pdf-backend` 优先于 `.env` 中的 `PRISM_PDF_BACKEND`；都未设置时保持原来的 `ura`。新 `.env.example` 选择 `api-rerank`。原版 URA 环境配置只在选择 `ura` 时检查。

API 后端使用 `qwen3-vl-rerank`：将每页渲染为最长边不超过 1120 像素的 PNG，连同问题发送到百炼原生重排接口，校验每页返回的索引和有限数值分数，选取前两页交给现有视觉问答流程。引用框仍来自本地 PDF 解析，而非重排模型生成。此方案借鉴 MPMQA 的检索后问答流程，不等同于原版 URA 模型或论文效果。

默认复用 `OPENAI_API_KEY` 与 `OPENAI_API_BASE` / `OPENAI_BASE_URL` 的百炼服务；可用 `PRISM_RERANK_API_KEY`、`PRISM_RERANK_BASE_URL` 指定独立配置。接口地址由官方 DashScope/MAAS 基地址转换，不通过聊天接口调用。模型、Key 和服务地域必须匹配。未配置或超过 40 页时在付费规划之前报错；不会静默截取前 40 页。当前尚未加入长文档 Embedding 召回，单请求仍受服务端 Token 限制。

每次重排预留一次请求，关闭自动重试；本次运行内相同 PDF 哈希、问题和后端的排序结果复用缓存。缓存命中前仍校验文件哈希。API 失败时不会偷偷回退到 URA，也不会伪造相关度。所有候选页图片会发往重排服务，选中页面再发往问答模型。

`events.jsonl` 的 `pdf_rerank_usage`、`graph_model_end` 和 `evidence_model` 记录服务返回的用量，不记录请求中的 Base64 图片。文档结果中的 `retrieval_method`、`retrieval_details` 会标明真实后端、排序分数与缓存状态。

### 有界验证与费用

先执行离线 `demo`，然后复用生成的合成 PDF（不要把私人资料用于 smoke check）：

```bash
prism-research demo --output runs/pdf-fixture
python scripts/pdf_api_smoke.py --mode rank --pdf runs/pdf-fixture/synthetic-study.pdf --expected-page 2 --env-file PATH_TO_ENV --output runs/pdf-rank-check
python scripts/pdf_api_smoke.py --mode agent --pdf runs/pdf-fixture/synthetic-study.pdf --expected-page 2 --env-file PATH_TO_ENV --output runs/pdf-agent-check --max-requests 16
```

脚本必须使用新输出目录，默认使用 `qwen3.5-flash`。`rank` 最多 1 次重排请求，检查相关页是否排第一；`agent` 检查报告非空且包含预期页的引用。这些是合成资料的工程联调，不是检索准确率评测。`summary.json` 保存排序、引用、用量和人民币标价估算。

截至 2026-10-28，百炼中国站每百万 Token 标价：`qwen3-vl-rerank` 输入 0.5 元；`qwen3.5-flash` 输入 0.2 元、输出 2 元；`qwen3.5-122b-a10b` 输入 0.8 元、输出 6.4 元。聊天价格适用单请求输入不超过 128K 的档位。估算不扣免费额度和缓存折扣，不代表账户实际账单；失败请求未返回用量时也不能据此确定实际费用。见[官方价格](https://help.aliyun.com/zh/model-studio/model-pricing)。

## MPMQA 原版检索环境

选择 `--pdf-backend ura` 时，Agent 的 Python 3.11 与作者的旧版 Torch/Transformers/Detectron2 不混装。作者模型在独立子进程运行，通过 LangChain Runnable `.ainvoke()` 接入；这是真实本地模型推理，不是 OpenAI API 请求。API PDF、网页和非 PDF 流程不需要这套依赖。

1. 准备 [MPMQA 官方仓库](https://github.com/AIM3-RUC/MPMQA)，使用提交 `5226a9aa849fd3b8f35620e7b7d7d10c09d754a2`。已有 `reference_repos/MPMQA` 可直接使用，无需再复制一套源码。
2. 按作者 README 在独立环境准备 Torch 1.10.2、Transformers 4.17.0、Detectron2 及 BUA 自定义算子、OpenCV 等。优先使用 Linux/CUDA 环境；本迁移没有验证原版 Detectron2 在当前 Windows 环境的构建。不要直接把作者的整个 requirements.txt 安装进 Agent 环境。
3. 按作者 README 的链接下载 `expr.zip`（URA 配置／checkpoint）及 `pretrained.zip`（T5、BUA）。保留作者目录结构，确认配置中的 `pretrained_dir`、`roi_config`、`roi_model` 指向存在的资产。相对路径以作者仓库为根。
4. `PRISM_MPMQA_CHECKPOINT` 必须指向完整模型权重文件，支持作者的 `model` 字典、整合后的 DeepSpeed `module` 字典或原始 state dict。对于分片 ZeRO checkpoint，先用对应 DeepSpeed 导出工具合并为完整 FP32 state dict；不能把 checkpoint 目录或任意单个分片当完整权重。加载使用 `strict=True`，配置、词表或权重不匹配会直接失败。
5. 在 `.env` 配置绝对路径。以下为 Linux 部署结构示例，具体 checkpoint 文件名以下载内容为准：

```dotenv
PRISM_MPMQA_PYTHON=/opt/conda/envs/mpmqa/bin/python
PRISM_MPMQA_REPO=/workspace/reference_repos/MPMQA
PRISM_MPMQA_CONFIG=/workspace/reference_repos/MPMQA/expr/URA/config.json
PRISM_MPMQA_CHECKPOINT=/workspace/models/ura-full-state.pt
PRISM_MPMQA_DEVICE=cuda:0
PRISM_MPMQA_TIMEOUT=600
```

两套 Python 需共享可访问的文件路径。若使用 WSL，请把 Agent 和 URA worker 都运行在 WSL 文件系统环境中，不要直接将 Linux Python 路径配置给 Windows 进程。配置可用后沿用原来的 `research --pdf ...` 命令。

检索期间不会自动下载权重，也不会因权重缺失退回词元检索。worker 每次检索加载一次模型、问题只编码一次，再逐页编码评分；当前未实现跨请求特征缓存，首次和重复调用的加载成本均需实际测试。默认本地检索超时 600 秒，外层整次研究仍受 900 秒限制。

主进程检查 PDF 哈希并渲染未标注页面；区域框和词框转换到同一张渲染图的像素空间。原版输入依赖版面标注，本项目以解析区域适配，缺少 OCR／精细版面类别时不会编造这些信息。查看报告 JSON 的 `retrieval_method` 与 `retrieval_details` 可确认评分模式、每页分数和 token 截断情况。

离线 `demo` 使用明确固定的合成页面，不调用 URA，也不用于证明检索效果。实际权重质量、显存与耗时需要在上述环境中另行验证。

## 预算与状态

CLI 默认 `--planner native`，采用 Open Deep Research 的 Supervisor / Researcher 循环。低成本配置将完整问题合并为一次研究委派（`--research-rounds 1`），单个研究者最多四轮工具调用。需要更多研究轮次时可显式增加 `--research-rounds`（1～6），并相应调整请求预算。每次研究创建独立账本，默认最多 16 次请求、一次网页搜索；工具自身限制图片观察和文档问答次数。请求预算耗尽会中止执行，不转换为成功结果。文本字符与输出 token 预留量用于执行限制，不等于实际账单金额上限。内层模型调用关闭继承的预算回调，由工具自行记账。

显式 `--planner mcsearch` 启用实验循环，CLI 最多四轮动作。底层 `mcsearch_max_steps` 支持 1～8 轮，默认值为 3，CLI 指定为 4。`No Retrieval` 也消耗动作轮数。N 轮最多产生 N+1 次规划调用；额外的证据评估、工具内部模型调用也计入总请求预算。下面的单步状态与 `mcsearch_trace.json` 仅属于 MC 实验模式，不是默认主流程的必要条件。

`answered` 是模型对该步证据充分性的判断；`completed` 表示控制器输出 Final Answer 且已执行研究步骤均为 answered，不代表独立验证通过。最终简答保存在 `final_answer`，停止原因单独记录。`insufficient`、`budget_exhausted` 等状态会作为待确认事项进入报告；证据编号校验只保证引用对象存在，语义支持仍需要评估。没有建立作者的本地图文知识库时，不提供 `Local Image Retrieval with Text Query` 能力。

## 运行目录

`runs/` 默认不提交版本控制。资料快照、观察、哈希和引用保存在本地。HTML 报告会内嵌引用页图片，因此分享报告也会分享这些页面内容。运行异常会保留已写入的过程记录。

本仓库的运行入口是本地 CLI；展示站点是静态页面。开发测试使用合成资料和替代模型接口，真实调用时使用你自己的服务账户。

## 有界真实联调

`research --env-file PATH` 可显式使用另一份本地凭证配置，不需要将密钥复制到代码中。
`python scripts/live_smoke.py --case image --env-file PATH --output runs/UNIQUE_NAME --max-requests 10`
会生成标注为合成资料的温度说明图，真实调用完整 Agent 并保存 `summary.json` 与报告。
`--case web` 测试 Microsoft Learn 官方文档的搜索、重排与原文阅读。该脚本会消耗模型／搜索额度，不属于离线测试；每次使用新的输出目录。网页 smoke 要求报告包含发现和证据，空报告不计成功；MC smoke 还要求证据链状态为 completed。

常规修改先执行 `python -m pytest -q`，不消耗 API 额度。只有离线检查通过后，才执行需要验证的单条真实链路：

```bash
python scripts/live_smoke.py --case image --env-file PATH_TO_ENV --output runs/check-image
python scripts/live_smoke.py --case image --planner mcsearch --env-file PATH_TO_ENV --output runs/check-mc
python scripts/live_smoke.py --case web --env-file PATH_TO_ENV --output runs/check-web
```

这些脚本和 PDF smoke 默认都是 Flash；不会自动升级成贵模型或无限重试。Flash 的 MC 首次动作使用受限结构化规划，保证先获取工具证据，再允许结束；后续仍沿用动作/观察循环。结构化结果如由兼容接口放入普通文本，将经过同一 Pydantic 校验后接纳，不额外调用模型修复格式。

网页只读取原问题明确提供、搜索实际返回或已归档记忆提供的待核验 URL，不根据模型记忆猜路径。指定站点没有搜索结果时保留证据缺口；普通网页模式不再发起第二次付费查询，商品模式允许预算内的第二条补充查询。没有其他可用资料时结束当前子研究。搜索服务的索引覆盖不能由调试流程保证。

脚本也默认使用原生流程，追加 `--planner mcsearch` 可选择实验模式；`summary.json` 会保存所选模式。此前 MC 模式的真实调用记录不能代表原生模式已经完成真实联调。离线测试覆盖两种模式的完整 PDF 工具与报告数据通路，真实 URA 权重、模型质量及两种编排方式的对照效果仍需另行测试。

图片观察通过 LangChain 的结构化输出约束字段类型。MC 模式的 supervisor 只绑定一次 `ConductResearch`，避免旧的思考／结束工具与单链委派规则冲突。动作解析容忍代码围栏、闭合标签和空的可选依赖标签，但仍拒绝多动作和非法依赖。搜索结果全部被来源规则排除时，工具返回 `insufficient`，由 Agent 保留证据缺口，不把零候选伪装成成功检索。

配置不完整时会显示本地预检原因；运行中的异常原因经密钥脱敏后写入 `events.jsonl`。非法规划输出也保存在本地轨迹中，供排查协议问题。
