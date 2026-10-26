# 方法与代码对应

## Open Deep Research
基线提交：`1b7d2e80db9faa586165c60e09096dbbfd483a64`。保留 `clarify_with_user`、`write_research_brief`、Supervisor 子图和最终报告节点。通用模式将一次研究委派交给证据规划控制器；原文本模式仍保留。

## MMSearch
参考 §2.1 的 requery / rerank / summarization 顺序及官方 `demo/mmsearch_engine.py`。适配层使用 LangChain，支持登记图片和候选截图；候选由百炼搜索提供，选中后读取原文。模型选择返回的编号必须合法。原方法与适配范围见函数注释。

## MC-Search
来源固定提交：`da9fe096592ec3555b3d3c9afe284c939c257458`。原样复用 `parse_search_block` 与 `extract_final_answer`。中文提示词保留 `<Sub-Question>` / `<Search>` / `<End>` 协议，新增依赖与文档动作。本地检索动作在适配层映射到现有图像、网页工具；`Local Document Retrieval` 是本项目扩展。计划与执行状态存入 `mcsearch_trace.json`。

## MPMQA
来源固定提交：`5226a9aa849fd3b8f35620e7b7d7d10c09d754a2`。将 `MQAT5Model.similarity_score` 的双向 token 点积、max、mean 评分适配为 NumPy 单对输入。当前页面特征为词元 one-hot，采用 query-to-page 分数；其余管线使用 PyMuPDF 和 VLM。原文页码、区域坐标由解析器确定，模型只选择已存在区域。方法归属、改动和许可保留在 `mpmqa_scoring.py`。

## 核验与展示
`claim_verification.py` 保留对象／指标／范围／条件规则与 NVIDIA 历史专用核验。通用研究入口使用引用受约束的报告生成器；两种路径职责不同。展示中的 NVIDIA 案例为历史运行摘要，文档演示为明确标注的合成样例。
