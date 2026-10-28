# 系统设计与方法实现

默认研究流程由 Open Deep Research 的 Supervisor / Researcher 编排，多模态能力通过工具接入。MC-Search 是显式选择的实验分支，不再是通用研究的必经节点。以下记录各模块的来源、接入方式与实现边界。

## Open Deep Research
基线提交：`1b7d2e80db9faa586165c60e09096dbbfd483a64`。保留 `clarify_with_user`、`write_research_brief`、Supervisor / Researcher 子图和最终报告节点。CLI 默认 `--planner native`，沿用原生多轮委派与工具调用；通用研究工具按当前图片、文档和联网权限暴露，报告直接读取证据目录，无需 MC 轨迹。预算耗尽会中止执行，工具异常写入运行日志。

`--planner mcsearch` 才将一次完整委派交给 MC 实验循环，对应底层 `mcsearch_enabled=True`。两种模式共享证据工具、请求预算和引用校验。多步规划与历史保留本来就存在于基线；MC 分支用于研究不同执行协议，当前没有证明其相对基线的质量或成本收益。MC 的依赖字段和单步证据充分性自评属于该实验分支，不冒充默认流程的能力。

## MMSearch
参考 §2.1 的 requery / rerank / summarization 顺序及官方 `demo/mmsearch_engine.py`。适配层使用 LangChain，支持登记图片和候选截图；候选由百炼搜索提供，选中后读取原文。模型选择返回的编号必须合法。原方法与适配范围见函数注释。

## MC-Search（可选实验）
来源固定提交：`da9fe096592ec3555b3d3c9afe284c939c257458`，`src/gpt/agent_local_retrieval_gpt_multi.py`。原样复用 `parse_search_block` 与 `extract_final_answer`；`mcsearch_loop.py` 将作者 `vqa_agent_local_gpt` 主循环迁移为异步可注入版本：系统消息与起始输入 → assistant 动作 → user 检索观察 → 携带完整历史的下一次模型调用 → Final Answer。LangChain 负责模型调用和消息转换。保留对话历史是通用 Agent 机制，不是 MC-Search 独有创新；此处的价值是实现与作者流程对齐，而不是给 LangChain 增加一种新的记忆能力。

`mcsearch_adapter.py` 提供实际工具映射、中文动作提示词、预算和证据校验。`<Dependencies>` 不再是必须字段，原版仅含 `<Sub-Question>` / `<Search>` 的动作可以执行。原版 `No Retrieval` 动作不调用工具，但占用一轮；最终 `<End> / Final Answer` 保存为控制器简答，仍由最终报告的证据约束处理。最后一次检索后允许模型再作一次决策，修正原循环未解析最后一轮 Final Answer 的边界；若仍请求动作则以预算耗尽结束。配置上限为 8 轮，CLI 默认仍为 4 轮。

明确的适配差异：作者 Text 动作使用本地知识库，我们映射为 MMSearch 网页工具；Input Image 向规划模型提供登记原图，并通过现有视觉工具取得可编号的观察。没有接入作者的 Visualized-BGE/FAISS 图文知识库，因此文本检索图片动作可解析但会被运行能力检查拒绝，不会伪装成已执行图片检索。Document 动作、依赖与单独的证据评估是本项目扩展；评估摘要与原始证据一同返回，不取代原文。提示词不要求输出隐藏思考；无证据直接结束会被项目校验拒绝。未接入 Search-Align 训练。

计划、观察、简答及状态写入 `mcsearch_trace.json`；图片内容在模型历史中保留，保存轨迹时只记图片引用占位，不重复写入 Base64。作者主函数原样摘录于 `tests/fixtures/mcsearch_author_agent.py`，用于文本、输入图像、无检索路径的离线对话顺序对照；这些测试验证实现行为，不证明真实研究质量。来源哈希及差异列表见 `docs/third_party/mcsearch-provenance.json`。

## MPMQA
来源固定提交：`5226a9aa849fd3b8f35620e7b7d7d10c09d754a2`。页面检索通过 `pdf_backend.py` 的 LangChain Runnable 选择后端：`ura` 调用 `mpmqa_backend.py` 与独立环境中的 `mpmqa_worker.py`，加载作者 `MQAT5Model` 和完整 URA checkpoint；`api-rerank` 将页面图片发送给百炼 `qwen3-vl-rerank`，返回真实页面索引与相关度分数。两者均不使用 one-hot 或词元匹配回退。

API 模式是替代检索方案，不是原版 URA 推理：目前直接重排最多 40 页，尚未实现 Embedding 召回。相同运行内重复问题可复用排序缓存，API 请求计入预算并记录返回用量。两种后端保留相同的 PDF 哈希、页码和区域 ID；文档问答与最终报告通过结构化输出约束格式，随后校验引用确实存在。格式与引用存在性检查不等于语义正确性验证。

输入适配复用作者 `MQAContrastDataset.get_page` 的排列：区域语义 token 后接词级 subword；每个 subword 继承对应单词的 bbox 与区域类型。页面以未标注的 RGB 图像渲染，worker 用 OpenCV 读取为 BGR，并将归一化区域／单词坐标转换为同一图像的像素坐标。作者模型完成 T5 token embedding、segment embedding、BUA ROI 外观特征、位置特征融合和 T5 编码；检索投影头及 L2 归一化也被保留。tokenwise 配置调用原始双向 MaxSim，页面排序采用 query-to-page 分数；global 配置按作者全局特征点积执行。单进程推理绕过训练用的跨进程 AllGather，不修改作者编码器或位置归一化。

适配差异：PM209 提供人工版面与 OCR 标注，任意 PDF 目前使用 PyMuPDF 提取的文字块、单词框、图片和图形区域，以及粗粒度类别映射。未自动识别 Title/Table 等精细类型，也未新增 OCR 引擎；扫描页可送入视觉编码，但缺少 OCR token。页面按 checkpoint 配置限制 token 数，截断状态写入检索详情。未实现作者整套多任务训练；选中页面的回答与引用继续由现有 LangChain VLM 工具生成，而非 URA 解码器或 saliency head。

原文页码、区域坐标仍由解析器确定，回答模型只选择已存在区域。`mpmqa_scoring.py` 保留为 NumPy 数值参考，不再用于正式页面检索。离线测试覆盖接口、坐标与状态传递，不代表已验证真实权重的检索质量。

## 核验与展示
`claim_verification.py` 保留对象／指标／范围／条件规则与 NVIDIA 历史专用核验。通用研究入口使用引用受约束的报告生成器；两种路径职责不同。展示中的 NVIDIA 案例为历史运行摘要，文档演示为明确标注的合成样例。
