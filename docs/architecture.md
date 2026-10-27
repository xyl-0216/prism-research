# 系统设计与方法实现

研究流程由 Open Deep Research 编排，证据规划控制器根据已有结果选择工具，最终将来源与结论关联到研究报告。以下记录各模块的来源、接入方式与实现边界。

## Open Deep Research
基线提交：`1b7d2e80db9faa586165c60e09096dbbfd483a64`。保留 `clarify_with_user`、`write_research_brief`、Supervisor 子图和最终报告节点。通用模式将一次研究委派交给证据规划控制器；原文本模式仍保留。

## MMSearch
参考 §2.1 的 requery / rerank / summarization 顺序及官方 `demo/mmsearch_engine.py`。适配层使用 LangChain，支持登记图片和候选截图；候选由百炼搜索提供，选中后读取原文。模型选择返回的编号必须合法。原方法与适配范围见函数注释。

## MC-Search
来源固定提交：`da9fe096592ec3555b3d3c9afe284c939c257458`。原样复用 `parse_search_block` 与 `extract_final_answer`。中文提示词保留 `<Sub-Question>` / `<Search>` / `<End>` 协议，新增依赖与文档动作。本地检索动作在适配层映射到现有图像、网页工具；`Local Document Retrieval` 是本项目扩展。计划与执行状态存入 `mcsearch_trace.json`。

## MPMQA
来源固定提交：`5226a9aa849fd3b8f35620e7b7d7d10c09d754a2`。正式页面检索通过 `mpmqa_backend.py` 的 LangChain Runnable 调用独立环境中的 `mpmqa_worker.py`，直接加载作者 `MQAT5Model` 与完整 URA checkpoint，不再使用 one-hot 或词元匹配回退。

输入适配复用作者 `MQAContrastDataset.get_page` 的排列：区域语义 token 后接词级 subword；每个 subword 继承对应单词的 bbox 与区域类型。页面以未标注的 RGB 图像渲染，worker 用 OpenCV 读取为 BGR，并将归一化区域／单词坐标转换为同一图像的像素坐标。作者模型完成 T5 token embedding、segment embedding、BUA ROI 外观特征、位置特征融合和 T5 编码；检索投影头及 L2 归一化也被保留。tokenwise 配置调用原始双向 MaxSim，页面排序采用 query-to-page 分数；global 配置按作者全局特征点积执行。单进程推理绕过训练用的跨进程 AllGather，不修改作者编码器或位置归一化。

适配差异：PM209 提供人工版面与 OCR 标注，任意 PDF 目前使用 PyMuPDF 提取的文字块、单词框、图片和图形区域，以及粗粒度类别映射。未自动识别 Title/Table 等精细类型，也未新增 OCR 引擎；扫描页可送入视觉编码，但缺少 OCR token。页面按 checkpoint 配置限制 token 数，截断状态写入检索详情。未实现作者整套多任务训练；选中页面的回答与引用继续由现有 LangChain VLM 工具生成，而非 URA 解码器或 saliency head。

原文页码、区域坐标仍由解析器确定，回答模型只选择已存在区域。`mpmqa_scoring.py` 保留为 NumPy 数值参考，不再用于正式页面检索。离线测试覆盖接口、坐标与状态传递，不代表已验证真实权重的检索质量。

## 核验与展示
`claim_verification.py` 保留对象／指标／范围／条件规则与 NVIDIA 历史专用核验。通用研究入口使用引用受约束的报告生成器；两种路径职责不同。展示中的 NVIDIA 案例为历史运行摘要，文档演示为明确标注的合成样例。
