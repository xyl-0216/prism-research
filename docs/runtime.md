# 运行配置

## 输入与工具

`research` 接收一个 PNG/JPEG（最大 6 MB、1600 万像素）和/或一个 PDF（最大 20 MB、100 页）。PDF 的文字、词级坐标、内嵌图片和向量绘图区域通过 PyMuPDF 提取；页面采用归一化 `[x0,y0,x1,y1]` 坐标。带旋转的页面需先归一化。页面检索使用作者 URA 模型，见下方独立环境配置；不再提供词元匹配回退。扫描页仍可使用图像区域，但当前未加入 OCR，不能假定其检索质量等同有文字层的页面。

`--web` 使用百炼 Qwen3.5 搜索，原文读取沿用明确的来源域名范围：docs.langchain.com、openai.github.io、learn.microsoft.com、www.nvidia.com、images.nvidia.com。需要其他来源时，应在 `evidence_tools.py` 中审查并扩展来源列表。`--web-screenshot "HTTPS_URL=LOCAL_PATH"` 可将本地网页截图关联到候选 URL；截图必须由使用者事先准备。

模型接口可使用 `--model openai:MODEL_NAME` 指定。输入图片和选中 PDF 页面会发往该模型接口；本地 `.env` 不会写入报告。网页模式要求百炼接口及 `openai:qwen3.5-...` 模型。

## MPMQA 原版检索环境

Agent 的 Python 3.11 与作者的旧版 Torch/Transformers/Detectron2 不混装。作者模型在独立子进程运行，通过 LangChain Runnable `.ainvoke()` 接入；这是真实本地模型推理，不是 OpenAI API 请求。网页和非 PDF 流程不需要这套依赖。

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

每次 CLI 研究创建独立账本，默认最多 24 次请求，一次网页搜索、两次图片观察、两次文档问答，规划最多四步。文本字符与输出 token 预留量用于执行限制，不等于实际账单金额上限。内层模型调用关闭继承的预算回调，由工具自行记账。

`answered` 是模型对该步证据充分性的判断，`completed` 表示规划器决定停止。`insufficient`、`budget_exhausted` 等状态会作为待确认事项进入报告；证据编号校验只保证引用对象存在，语义支持仍需要评估。

## 运行目录

`runs/` 默认不提交版本控制。资料快照、观察、哈希和引用保存在本地。HTML 报告会内嵌引用页图片，因此分享报告也会分享这些页面内容。运行异常会保留已写入的过程记录。

本仓库的运行入口是本地 CLI；展示站点是静态页面。开发测试使用合成资料和替代模型接口，真实调用时使用你自己的服务账户。
