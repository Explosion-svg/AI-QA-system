# RAG 知识库 AI 问答系统技术分析与优化建议

## 说明

本文基于当前仓库的源码静态审阅完成，重点阅读了以下模块：

- `src/api/*`
- `src/services/*`
- `src/rag/*`
- `src/index/*`
- `src/memory/*`
- `src/infra/*`
- `main.py`
- `api.py`
- `app.py`

结论以“代码真实实现”为准，而不是以 README 或架构文档中的目标描述为准。

这意味着本文会同时覆盖两类内容：

1. 当前系统已经做了什么
2. 当前系统声称做了，但代码实际上还没有真正完成什么

另外，本文主要是源码层面的技术分析，并未基于真实业务数据集做离线评测，也没有对外部模型 API 做在线压测。因此，关于效果和性能的结论分为两类：

- 高置信度结论：可直接从代码逻辑推出
- 经验性建议：结合 RAG 系统通用工程实践提出

---

## 一、系统现状概览

### 1. 当前架构分层

从目录结构和 `main.py` + `src/container.py` 看，这个项目已经具备比较清晰的分层思想：

- API 层：接收 HTTP 请求，做参数校验，调用 Service
- Service 层：编排聊天与上传流程
- RAG 层：负责检索相关逻辑
- Index 层：负责文档加载、切分、向量存储
- Memory 层：负责会话历史
- Infra 层：负责配置、日志、LLM、Embedding

这说明项目在“工程组织形式”上是有意识地往企业级结构靠拢的，这一点是优点。

### 2. 当前真实 RAG 链路

根据 `src/services/chat_service.py`、`src/rag/rag_engine.py`、`src/index/document_loader.py`、`src/index/chroma_store.py`，当前真实链路大致是：

1. 上传文件
2. `DocumentLoader` 按文件类型加载文档
3. 使用 `RecursiveCharacterTextSplitter` 进行字符级切分
4. 通过 `Chroma` + `HuggingFaceEmbeddings` 构建向量库
5. 用户提问时，调用 `QueryRewriter`
6. 进入向量检索
7. 进入 rerank
8. 进入 context filter
9. 将拼接后的上下文喂给 LLM 生成答案

### 3. 当前代码层面的优点

如果只看设计意图，这个系统已经具备了一个 RAG 系统的核心骨架：

- 有独立的文档加载、切分、索引构建模块
- 有独立的检索、改写、重排、过滤模块
- 有持久化向量库，而不是纯内存 demo
- 有多模型 provider 抽象
- 有会话历史管理
- 有容器对象统一管理依赖生命周期

这些都说明它不是“一段脚本式的问答 demo”，而是朝着可维护系统演进过的。

但从“真实可用性”和“检索效果”角度看，目前还存在不少关键断点。

---

## 二、当前实现与目标设计的主要偏差

这一节很关键，因为很多优化建议的优先级，取决于系统现在究竟卡在“效果不够好”，还是“链路本身还没打通”。

### 1. 文档宣称了“混合检索 + RRF”，但代码没有真正实现

架构文档中描述的检索流程是：

- BM25 检索
- 向量检索
- RRF 融合
- rerank

但从 `src/rag/retriever.py` 和 `src/rag/rag_engine.py` 看，当前真实实现只有向量检索，没有以下能力：

- 没有 BM25 索引构建
- 没有关键词召回通路
- 没有 RRF 融合逻辑
- 没有 sparse + dense 的协同召回

这意味着当前系统并不是“混合检索 RAG”，而是“单路 dense retrieval + rerank”。

### 2. Query Rewrite 名义上存在，但没有真正提高召回

`src/rag/rewrite.py` 中有 query rewrite 模块，但在 `src/rag/rag_engine.py` 里虽然生成了多个 `rewritten_queries`，实际检索时只使用了第一个 query：

- `rewritten_queries = self.query_rewriter.rewrite(query)`
- `search_query = rewritten_queries[0] if rewritten_queries else query`

由于第一个 query 永远是原始 query，因此：

- 改写结果没有真正用于多路召回
- 关键词提取没有进入检索融合
- 多 query 召回增益基本没有实现

此外，当前 `_extract_keywords()` 对中文的处理也比较弱：

- 主要依赖 `split()`
- 中文没有显式分词
- 中文查询若没有空格，关键词抽取几乎不成立

因此现在的 query rewrite 更像“接口预留”，不是有效的召回增强机制。

### 3. 检索结果类型在模块之间不一致，RAG 主链路存在高概率运行时错误

这是当前代码里最严重的问题之一。

`src/rag/retriever.py` 的 `search()` 返回：

- `List[Document]`

但 `src/rag/rerank.py` 的 `rerank()` 期望输入：

- `List[Tuple[Document, float]]`

而 `src/rag/rag_engine.py` 中调用方式是：

- `results = self.retriever.search(...)`
- `results = self.reranker.rerank(query, results, top_k=k)`

这会导致 reranker 内部对 `for doc, _ in candidates` 的解包和实际输入不匹配。按当前实现推断，这一段非常可能直接抛异常，然后被 `retrieve()` 的总异常处理吞掉，最终返回空结果。

也就是说，系统很可能在“表面上有 RAG 模块”，但实际问答时经常检索不到上下文，退化成纯 LLM 回答。

### 4. rerank 之后的数据结构与后续 context 拼接也不一致

`src/rag/context_filter.py` 处理的是：

- `List[Tuple[Document, float]]`

而 `src/rag/rag_engine.py` 的 `get_context_with_sources()` 却按下面方式使用返回值：

- `doc.page_content`
- `doc.metadata`

这说明下游又把结果当成了纯 `Document`，而不是 `(Document, score)`。

即使前面的 rerank 成功，这里也仍然存在结构不一致问题。

### 5. 清空索引逻辑不完整

`src/rag/rag_engine.py` 的 `clear_index()` 调用了：

- `self.retriever.clear_index()`

但 `src/rag/retriever.py` 并没有定义 `clear_index()`。

这意味着：

- 清空知识库接口可能直接报错
- 系统生命周期管理与知识库维护流程不完整

### 6. 知识库状态接口不准确

`src/rag/rag_engine.py` 的 `list_sources()` 目前直接返回空列表。

这会导致：

- 即使索引已构建，来源列表也拿不到
- `/chat/status`、`/upload/status` 的信息不完整
- 前端和运维都无法准确观察知识库内容

### 7. 配置集中化并未完全落地

虽然项目有 `src/infra/config.py`，但仍存在硬编码：

- `UploadService` 内部写死 `knowledge_base`
- `Container.vector_store()` 写死 `collection_name="vector_store"`
- `Container.vector_store()` 写死 `persist_directory="vector_db"`
- `RAGEngine(top_k=4)` 没有使用 `RAG_TOP_K`

这会导致：

- 配置项名义上集中，实际仍然散落
- 部署时修改路径和检索参数不够可靠
- 环境切换更容易出现“文档说一套、代码跑一套”

### 8. 新旧架构并存，系统边界有漂移

仓库中同时存在两套接口和调用路径：

- 新链路：`main.py` + `src/api/*` + `src/container.py`
- 旧链路：`api.py` + `app.py` + 一些旧的 import 路径

并且旧代码中还能看到以下问题：

- `app.py` 依赖 `/knowledge-base/*` 路径，而新 API 用的是 `/upload/*` 和 `/chat/*`
- `app.py` 引用了 `src.utils.history_manager`，但当前目录下不存在该模块
- `src/cli.py` 中 `RAGEngine()` 的构造方式与当前实现不匹配
- `src/cli.py` 还尝试从 `src.infra.config` 导入 `setup_logging`，但当前该函数已不在可用定义中

这说明系统存在明显的“演进中遗留代码分叉”。这不只是代码风格问题，而是会直接影响：

- 哪条链路才是真正可运行的
- 前端是否还能兼容后端
- 新功能改动是否会改漏
- 线上排查时是否会定位错入口

---

## 三、从技术层面看，当前系统最值得优先解决的问题

如果按优先级划分，我会把问题分成四个层次。

### P0：先修复系统正确性

这是最先要做的，否则后面的优化价值会被大幅削弱。

#### 1. 统一检索链路的数据结构

建议明确约定一个统一的数据对象，例如：

- `RetrievedChunk`
  - `document`
  - `dense_score`
  - `sparse_score`
  - `fusion_score`
  - `rerank_score`
  - `source`
  - `chunk_id`

然后 rewrite、retrieve、fusion、rerank、filter 全链路都围绕这个结构处理，而不要在不同模块之间来回切换 `Document` 和 `(Document, score)`。

#### 2. 修复 RAG 主链路中的运行时断点

至少应修复：

- `Retriever.search()` 与 `Reranker.rerank()` 的输入输出一致性
- `ContextFilter.filter()` 与 `get_context_with_sources()` 的一致性
- `clear_index()` 中对不存在方法的调用
- `list_sources()` 的真实实现

#### 3. 区分“无检索结果”与“检索链路异常”

现在 `RAGEngine.retrieve()` 里一旦异常就直接返回 `[]`，这会把两种完全不同的状态混在一起：

- 本来就没有召回
- 代码出错导致没召回

建议至少记录结构化错误类型，例如：

- `retrieval_empty`
- `rerank_failed`
- `vector_store_failed`
- `context_filter_failed`

这样在日志和指标层面才能看得出系统到底是“效果差”还是“实现坏”。

### P1：提升召回质量

这是 RAG 效果提升的第一主战场。

#### 1. 真正实现混合检索

对于中文知识库问答，尤其是企业文档、制度、术语、接口文档、产品说明书场景，单路向量检索通常不够。

建议实现：

- dense retrieval：语义召回
- sparse retrieval：BM25 / TF-IDF / Elasticsearch sparse 通路
- fusion：RRF 或加权融合

为什么混合检索重要：

- 当用户问法接近原文措辞时，BM25 命中往往更准
- 当用户换说法、口语化表达时，dense 检索更强
- 当查询包含专有名词、编号、字段名、报错码时，稀疏检索价值非常高

如果不做混合检索，系统会在以下场景明显掉分：

- 术语名、零件号、接口参数名、政策编号
- 表格字段名
- 用户只输入短 query
- query 与文档表达高度接近但语义 embedding 不够稳

#### 2. 让 query rewrite 真正参与召回

当前 query rewrite 更适合升级为多路召回框架：

- 原始 query
- 关键词 query
- 术语扩展 query
- 子问题拆解 query

然后对每一路分别做检索，再融合。

实际工程中可以设计成：

1. 简单问题：只走原始 query
2. 术语问答：原始 query + 关键词 query
3. 复杂问题：query decomposition，拆成多个子 query

这样既能控制成本，也能提升召回覆盖率。

#### 3. 把 top-k 设计成分阶段参数，而不是单一常量

当前 `top_k=4` 太粗糙。更合理的方式是分层控制：

- dense recall top_k = 20~100
- sparse recall top_k = 20~100
- fusion top_k = 20~50
- rerank top_k = 5~10
- final context top_k = 3~6

原因是：

- 召回阶段要偏大，优先保 recall
- rerank 阶段再压 precision
- 最终给 LLM 的上下文要控制长度和噪声

如果一开始就把 top-k 卡得很小，后续 rerank 再强也没有候选可排。

#### 4. 增加 metadata filter

当前 metadata 基本只有：

- `source`
- `file_path`

建议扩展成标准 schema：

- `doc_id`
- `chunk_id`
- `source`
- `file_path`
- `page`
- `section`
- `title`
- `doc_type`
- `language`
- `created_at`
- `version`
- `tenant_id`
- `knowledge_base_id`

一旦 metadata 完整，就可以支持：

- 按文档范围过滤
- 按知识库过滤
- 按租户过滤
- 按时间版本过滤
- 按文档类型过滤

这对企业知识库非常重要，因为很多情况下“不应该从全库召回”。

#### 5. 引入 Parent-Child / Hierarchical Retrieval

当前切分是平铺式 chunk。对于长文档，这会带来两个问题：

- chunk 太小，语义不足
- chunk 太大，噪声太多

建议采用分层索引：

- child chunk 用于召回
- parent chunk / section 用于返回上下文

典型流程是：

1. 用小 chunk 做召回，提高定位精度
2. 将命中的 child chunk 映射回其 parent section
3. 将 parent section 或窗口扩展后的上下文交给 LLM

这通常比“直接把固定大小 chunk 拼起来”更稳。

#### 6. 为“无答案”设置阈值策略

现在 prompt 中写的是：

- 如果资料不足，可以结合自身知识补充

这会降低系统的可控性，尤其在企业知识库场景下容易出现“像是在回答，但其实不是基于知识库”的情况。

建议增加阈值策略：

- 如果召回分数整体过低，直接返回“知识库中未找到可靠依据”
- 如果 rerank 最高分低于阈值，拒答或弱回答
- 如果上下文之间冲突较大，提示“信息存在冲突”

这样能显著降低幻觉风险。

### P2：提升索引质量

索引质量对最终效果的影响，经常不亚于模型本身。

#### 1. 优化切块策略

当前 `RecursiveCharacterTextSplitter` 参数是：

- `chunk_size=500`
- `chunk_overlap=50`

这是一个能跑起来的默认值，但对不同文档并不一定合适。

建议从以下维度升级：

- 按标题/小节切块，而不是纯字符切块
- 表格单独处理，不要切坏行列关系
- FAQ / QA 文档按问答对切块
- API 文档按接口、参数、返回值、错误码切块
- 规章制度按章、节、条切块

经验上，最优切块不是固定数字，而是“结构感知 + 业务类型感知”。

#### 2. 补足 chunk 级元数据

现在切分后的 chunk 只有很基础的 metadata。建议每个 chunk 至少保存：

- `chunk_index`
- `start_offset`
- `end_offset`
- `page`
- `section_title`
- `parent_id`
- `token_count`
- `content_hash`

这些元数据会直接影响：

- 去重
- 增量更新
- chunk 回溯
- 引用展示
- 评测定位

#### 3. 做内容去重和增量索引

当前 `upload_and_index()` 的行为本质上是“保存文件后直接 add_documents”，没有真正的去重和 upsert。

这会导致：

- 重复上传同一文件时产生重复 chunk
- 文档更新时旧版本残留
- 向量库越来越脏

建议加入：

- 文件级 hash
- chunk 级 hash
- 基于 `doc_id + chunk_id` 的稳定 ID
- upsert 策略
- 文档版本字段

这样才能做真正可维护的知识库。

#### 4. 提升文档解析质量

当前支持 PDF / TXT / DOCX，但真实业务里最容易出问题的是 PDF。

尤其是以下类型：

- 扫描版 PDF
- 双栏排版
- 表格密集
- 页眉页脚很多
- 图文混排

如果解析质量差，后续 embedding 再强也没用。

建议补充：

- OCR 能力
- 表格抽取
- layout-aware 解析
- 页眉页脚清洗
- 目录和正文分离

在很多实际项目里，RAG 效果差并不是检索器差，而是“入库文本已经坏了”。

### P3：提升生成质量与可控性

#### 1. Prompt 需要从“泛问答”切换到“基于证据回答”

当前 prompt 倾向于：

- 有资料就参考
- 不足时可用模型自身知识补充

如果面向企业知识库，这种策略通常不够稳。

建议改成更明确的回答协议：

- 优先基于检索证据回答
- 不足时明确说明证据不足
- 回答中给出引用来源
- 不允许编造文档中未出现的事实

更理想的形式是输出结构化答案：

- `answer`
- `evidence`
- `citations`
- `confidence`

#### 2. 增加 source-aware generation

建议不要只把 chunk 文本拼接给模型，而是把来源信息一起显式注入，例如：

- 文档名
- 页码
- 节标题
- chunk 编号

这样模型更容易：

- 做引用
- 做归因
- 区分不同来源
- 避免把多个 chunk 混成一个事实

#### 3. 将 rerank 与答案生成解耦

当前 rerank 的目标是“找到最相关 chunk”，但最终生成时还可以再做一步 contextual compression：

- 保留相关句子
- 去掉噪声段落
- 提取与问题直接相关的 span

这会降低上下文冗余，提高回答稳定性。

---

## 四、检索策略还能如何改进

这一节重点回答“从检索角度还能做什么”。

### 1. 从单路 Dense Retrieval 升级到真正的 Hybrid Retrieval

推荐路线：

1. 先保留 Chroma dense 检索
2. 增加 BM25 检索器
3. 对两个召回列表做 RRF 融合
4. 对融合后的候选做 rerank

这样是投入产出比最高的一条路线。

#### 适用原因

- 当前系统已经有向量库和 rerank 模块，新增 BM25 的工程代价不算特别高
- `requirements.txt` 已经包含 `rank-bm25`
- 架构文档本身也已经围绕 hybrid retrieval 设计过

#### 预期收益

- 对精确术语、字段名、编号类问题效果提升明显
- 对短 query 的召回更稳
- 对问法偏口语化的 query 仍能保留 dense 优势

### 2. 使用多阶段检索

推荐把检索拆成三层：

1. 召回层：尽量多找候选
2. 精排层：rerank 压缩候选
3. 上下文构建层：控制最终给 LLM 的片段

不要把这三件事混成“直接检索 top 4”。

### 3. 增加 Query Classification

可以先判断用户问题属于哪一类，再决定检索策略：

- 事实型：直接检索 + rerank
- 解释型：扩展召回范围
- 多跳型：子问题分解
- 对比型：分别召回多个对象再汇总
- 指南型：偏向召回步骤文档、流程文档

这样比所有 query 都走同一套参数更稳。

### 4. 引入窗口扩展策略

很多时候被召回的 chunk 本身是对的，但上下文不够。

可在命中某个 chunk 后做窗口扩展：

- 向前补 1 个 chunk
- 向后补 1 个 chunk
- 或回到 parent section

这比单纯增大 `chunk_size` 更精细。

### 5. 对多文档答案做聚合

对于需要跨文档总结的问题，不要只依赖单轮拼接。

可以采用：

1. 先对每个来源分别抽取要点
2. 再做二次汇总

这对“对比多个制度版本”“总结多个章节”类问题更可靠。

---

## 五、向量库策略还能如何改进

这一节重点回答“向量库层面还能怎么做”。

### 1. Chroma 目前可继续使用，但要先补齐数据管理能力

对于小到中等规模知识库，Chroma 是可以继续使用的。当前真正的问题不在于“必须马上换库”，而在于你还没有把向量库当作可运维的数据系统来管理。

建议先补齐：

- 稳定主键 ID
- upsert 而不是无脑 add
- metadata schema
- 文档版本管理
- 删除单文档能力
- 来源枚举能力
- 索引重建流程

如果这些不做，换成别的向量库也不会本质改善效果。

### 2. 设计统一的向量记录模型

每条向量记录建议至少包含：

- `id`
- `doc_id`
- `chunk_id`
- `embedding_model_version`
- `content`
- `content_hash`
- `source`
- `page`
- `section`
- `tenant_id`
- `created_at`
- `updated_at`
- `version`

这样才能支撑：

- 重新 embedding
- 增量更新
- 数据回滚
- 多租户隔离
- 线上问题定位

### 3. 为未来迁移预留接口，而不是过早绑定 Chroma 细节

当前已经有 `VectorStore` 抽象，这是好事。但接口仍偏简单，只覆盖了：

- add
- similarity_search
- similarity_search_with_score
- delete_collection

建议向业务真正需要的能力扩展，例如：

- `upsert_documents`
- `delete_by_doc_id`
- `delete_by_filter`
- `list_sources`
- `list_documents`
- `search_hybrid`
- `search_mmr`
- `search_with_metadata`

这样以后无论切到：

- pgvector
- Elasticsearch / OpenSearch
- Milvus
- Qdrant
- Weaviate

都更平滑。

### 4. 什么时候应该考虑迁移向量库

如果出现以下需求，Chroma 可能就不是最优解：

- 百万级以上 chunk
- 高频并发查询
- 多租户隔离
- 稳定的过滤检索
- 混合检索和复杂排序
- 更强的运维能力和备份机制

这时可以考虑：

- 如果你偏数据库一体化：`pgvector`
- 如果你偏搜索引擎能力：`Elasticsearch/OpenSearch`
- 如果你偏专用向量检索性能：`Milvus/Qdrant/Weaviate`

### 5. Embedding 版本化必须纳入向量库策略

向量库不是“存进去就完了”，而是和 embedding 模型强绑定。

一旦更换 embedding 模型，就要考虑：

- 是否全量重建
- 新老向量是否混存
- 如何标识 embedding 版本

如果没有版本字段，后期很容易出现：

- 一部分向量是老模型生成
- 一部分向量是新模型生成
- 效果下降但排查困难

---

## 六、模型层面还能如何改进

这一节分成 embedding、reranker、generator 三块。

### 1. Embedding 模型

当前容器中使用的是多语 embedding，这个选择是合理的保守起点，但不一定是当前业务的最优点。

选 embedding 模型时要看三个因素：

- 语种：中文、英文、混合语料
- 领域：通用文本、技术文档、法务、医疗、代码
- 目标：偏召回还是偏精准

建议思路：

- 中文知识库优先考虑中文或中英双语检索优化模型
- 如果语料是中英混合、术语多，优先多语检索模型
- 如果是高专业领域，尽量做领域验证，而不是只看公开榜单

模型选择原则比具体型号更重要：

- 看召回效果
- 看向量维度与存储成本
- 看推理延迟
- 看是否适合 CPU / GPU 环境

### 2. Reranker 模型

当前已经引入 CrossEncoder，这是非常正确的方向。

但还可以进一步优化：

- 选更适合中文或多语的 reranker
- 对 rerank 输入长度做更严格控制
- 支持批量推理
- 支持 GPU 推理
- 支持 query/document 模板优化

要注意，reranker 不是越大越好，关键是看：

- 候选规模多大
- 延迟预算多少
- 实际中文问答效果如何

如果候选量是 20~50，轻量级 reranker 往往已经足够。

### 3. 生成模型

当前默认模型配置偏老，且系统 prompt 更偏“通用聊天”，不够像一个受控的知识库问答模型。

生成模型选择建议关注：

- 指令遵循能力
- 长上下文稳定性
- 中文表达质量
- 引用意识
- 拒答能力
- 成本 / 延迟

更合理的模型路由策略通常是：

- 轻问答：低成本模型
- 复杂归纳总结：更强模型
- 多文档综合：更强模型
- 高风险回答：高约束模型 + 更严格 prompt

### 4. 模型层推荐做路由，而不是固定单模型

实际系统里，效果最好且成本可控的方案往往不是“所有问题都走同一个模型”，而是：

- query 先分类
- 根据 query 复杂度选择生成模型
- 根据场景选择是否启用 rerank
- 根据召回置信度决定是否拒答

这会比单纯升级一个更贵的模型更划算。

### 5. Prompt 需要引入证据约束

建议改成类似规则：

- 仅基于给定资料回答
- 资料不足时明确说不知道或依据不足
- 每个结论尽量绑定来源
- 不要将多个来源中未明确对应的信息强行合并

如果业务上允许“知识库 + 模型先验知识”混合回答，也应该显式区分：

- 哪些内容来自知识库
- 哪些内容来自模型补充推断

---

## 七、影响系统效果和稳定性的关键因素

这一节回答“哪些因素最影响系统”。

### 1. 文档解析质量

这是第一位因素。

如果原始文档解析出来已经是脏文本，后续所有环节都会一起变差：

- chunk 错
- embedding 错
- 检索错
- 引用错
- 回答错

### 2. Chunk 策略

chunk 太小会导致：

- 语义碎片化
- 上下文不足

chunk 太大会导致：

- 召回不精准
- rerank 成本升高
- LLM 输入噪声增加

### 3. Metadata 完整度

metadata 不全时，系统会失去很多高级能力：

- 精准过滤
- 引用页码
- 版本区分
- 租户隔离
- 线上回溯

### 4. Embedding 与业务语料的匹配程度

即使是很强的 embedding 模型，只要和你的语料风格不匹配，召回也会掉分。

例如：

- 企业术语多
- 表格字段多
- 缩写多
- 中英混杂
- 编号和规范名很多

这时模型匹配程度比通用榜单排名更重要。

### 5. 是否使用真正的混合检索

对于术语密集型知识库，是否具备 dense + sparse 混合召回，通常是影响 top-k 命中率的关键因素。

### 6. Rerank 质量

召回到候选之后，rerank 决定了“最终喂给 LLM 的是不是最该看的几段”。

这一步对最终答案影响非常直接。

### 7. 上下文构建策略

不是召回到了就一定能答好。还要看：

- 给了多少段
- 段落顺序是否合理
- 是否有重复
- 是否有冲突
- 是否控制了总长度

### 8. Prompt 约束强度

Prompt 过松时，模型会倾向自由发挥；Prompt 过紧时，模型可能机械拒答。

因此需要在：

- 可回答性
- 真实性
- 拒答率

之间找到平衡。

### 9. 历史对话处理方式

多轮对话里，历史消息会影响：

- 指代消解
- 上下文污染
- token 成本
- 检索 query 重写

如果只是简单把历史全丢给模型，很容易让当前问答被旧上下文干扰。

### 10. 并发与共享状态

当前 `ChatService` 中会对全局共享 `LLMClient` 做 `switch()`，这在单用户 demo 中问题不大，但在多用户服务中会带来共享状态污染风险：

- A 用户刚切到 provider/model
- B 用户请求可能复用同一个共享 client

这会直接影响回答稳定性和问题复现。

### 11. API 同步阻塞

当前 FastAPI 接口是 async，但内部 LLM 调用、磁盘 I/O、文件处理大多是同步实现。

这会影响：

- 并发吞吐
- 响应延迟
- 超时概率

### 12. 评测闭环是否存在

没有评测集、没有离线指标、没有线上监控时，系统即使做了很多优化，也很难知道到底哪一步起效了。

---

## 八、如何评价这个系统

评估 RAG 系统不能只看“回答像不像”，必须拆成多维指标。

### 1. 检索层指标

这是最应该先建立的一层。

建议至少统计：

- `Recall@K`
- `Hit Rate@K`
- `MRR`
- `nDCG@K`

如果你有人工标注的“问题 -> 相关文档块”，就可以直接评估召回能力。

#### 检索层最关键的问题

- 正确答案所在 chunk 是否进入 top-k
- 正确文档是否至少有一条进入候选
- 正确 chunk 的排序是否足够靠前

如果检索层就没命中，生成层再强也无能为力。

### 2. Rerank 层指标

对 rerank 也应单独评估：

- rerank 前 top-k 命中率
- rerank 后 top-k 命中率
- rerank 是否把正确 chunk 排得更靠前

这样才能知道 reranker 到底是在帮忙还是在帮倒忙。

### 3. 端到端回答质量指标

建议至少看四个维度：

- `Correctness`：答案是否正确
- `Faithfulness / Groundedness`：答案是否真正由证据支持
- `Completeness`：是否回答完整
- `Citation Accuracy`：引用是否对应正确来源

如果是知识库问答系统，我认为 `Faithfulness` 的优先级通常高于“表达好不好看”。

### 4. 拒答能力指标

一个好的企业 RAG 不能只会答，还要会“不乱答”。

建议单独准备一批无法从知识库中回答的问题，评估：

- 是否能正确拒答
- 是否误答
- 是否把模型先验知识当成知识库事实

可以统计：

- 拒答准确率
- 误答率
- 无依据回答率

### 5. 系统性能指标

除了效果，还应该看系统层指标：

- 检索延迟
- rerank 延迟
- 总响应延迟
- p50 / p95 / p99
- 上传建库耗时
- 单文档解析耗时
- 向量化耗时
- 单问答 token 成本

### 6. 稳定性指标

建议监控：

- 检索异常率
- RAG 降级率
- LLM 超时率
- 文件解析失败率
- 空结果率
- 索引重建失败率

### 7. 用户体验指标

如果系统已经对外给人使用，还应看：

- 首字返回时间
- 最终回答时间
- 用户追问率
- 用户复制引用率
- “未解决问题”比例

---

## 九、建议的评测方法

这一节不是讲指标名，而是讲怎么实际落地。

### 1. 构建一个小而高质量的评测集

不要一开始追求几千条，先做 100~300 条高质量样本即可。

每条样本建议包含：

- 用户问题
- 标准答案
- 相关文档
- 相关 chunk
- 是否应拒答
- 问题类型

问题类型可以标注为：

- 事实问答
- 定义解释
- 流程说明
- 多跳推理
- 对比总结
- 无答案问题

### 2. 做分层评测，而不是只看最终回答

建议每次改动都看三层：

1. 检索是否提升
2. rerank 是否提升
3. 端到端是否提升

否则很容易出现：

- 端到端看起来没提升，但其实检索已经变好，只是 prompt 变差了
- 或 rerank 变好了，但 final context 拼接变差了

### 3. 做消融实验

建议按组件做 ablation：

- 不用 rewrite
- 不用 BM25
- 不用 rerank
- 不用 context filter
- 不同 chunk size
- 不同 embedding 模型
- 不同生成模型

这样才能知道真正带来收益的是哪一层。

### 4. 做数据切片分析

不要只看整体平均分，还要按类型切片看：

- 短 query / 长 query
- 单文档问题 / 多文档问题
- 术语类问题
- 表格类问题
- 扫描 PDF 问题
- 中文纯文本 / 中英混合问题

很多系统平均分不低，但在某些关键类型上会明显失效。

### 5. 引入线上观测

即使没有完整评测平台，也建议至少记录：

- query
- rewrite query
- recall chunks
- rerank top chunks
- final context 长度
- 引用来源
- 回答耗时
- 异常信息

这样排查具体 bad case 会容易很多。

---

## 十、工程与架构层面的具体优化建议

除了 RAG 算法本身，这个项目在工程上也有不少可以加强的地方。

### 1. 统一并裁剪旧代码入口

建议明确保留一套主入口：

- 保留 `main.py` + `src/api/*` + `src/container.py`
- 明确废弃或迁移 `api.py`、`app.py`、旧式 CLI 依赖路径

否则后续任何改动都可能出现：

- 新接口改了，旧前端没改
- 容器链路改了，CLI 还在调用旧构造方式

### 2. LLM 客户端改为请求级实例或无状态工厂

当前容器里共享一个 `LLMClient` 实例，再通过 `switch()` 切模型，这在并发环境下不安全。

更合理的方式：

- 每次请求根据 provider/model 创建独立 client
- 或维护无状态 client factory

这样不会出现跨请求污染。

### 3. 将同步 I/O 改成更合理的异步/后台任务模式

建议区分：

- 在线问答：尽量低延迟
- 上传建库：可以做后台任务

例如：

- 上传后返回任务 ID
- 后台异步解析、切块、向量化
- 前端轮询任务状态

这会比当前“上传接口里同步完成所有事情”更适合真实系统。

### 4. 上传流程需要更安全

当前上传流程存在几个典型问题：

- `await file.read()` 会一次性把文件读入内存
- 文件名未做严格清洗
- 缺少文件大小、类型、内容校验

建议使用：

- 流式写入
- 文件哈希
- 文件大小限制
- 安全文件名
- MIME 与后缀双校验

### 5. 增加可观测性

至少建议有以下日志和指标：

- 每次查询的召回文档数
- 每阶段耗时
- 召回来源
- rerank 前后排序变化
- token 使用量
- provider / model 使用分布
- 失败原因分类

如果后续要做效果迭代，这些是必须的。

### 6. 建立测试体系

当前仓库看不到成体系的测试。

建议至少补：

- 单元测试：切块、query rewrite、metadata、状态接口
- 集成测试：上传 -> 建库 -> 检索 -> 问答
- 回归测试：典型问题集

### 7. 会话存储不要长期停留在本地 JSON 文件

本地 JSON 适合 demo，但不适合长期服务化。

后续建议迁移到：

- SQLite：本地轻量
- PostgreSQL：服务化更稳
- Redis：做短期会话缓存

并区分：

- 会话元数据
- 历史消息
- 检索轨迹

---

## 十一、一个更合理的目标架构

如果把这个系统往“可用的 RAG 服务”方向继续演进，我建议目标形态是：

### 1. 入库链路

1. 文件上传
2. 文档解析
3. 清洗与结构化
4. chunk 切分
5. metadata 生成
6. embedding
7. 向量库 upsert
8. 建立 sparse 索引

### 2. 查询链路

1. query 规范化
2. query 分类
3. 多路召回
4. 融合
5. rerank
6. context assembly
7. answer generation
8. citation generation
9. 置信度判断 / 拒答

### 3. 观测链路

1. 检索日志
2. 生成日志
3. 指标上报
4. 评测数据沉淀
5. bad case 回放

---

## 十二、建议的分阶段落地路线

如果希望尽快把系统从“可演示”推进到“效果和工程都更稳”，我建议这样排优先级。

### 第一阶段：先修正确性和可观测性

目标：

- 让 RAG 主链路真正稳定跑通
- 能明确看到系统到底检索了什么

建议事项：

- 修复 `Document` / `score` 结构不一致
- 修复 `clear_index()` 和 `list_sources()`
- 补齐真实 source 列表
- 为每个阶段加日志和耗时指标
- 统一主入口，清理旧链路

### 第二阶段：补齐真正的混合检索

目标：

- 显著提升召回率和术语命中率

建议事项：

- 加入 BM25
- 实现 RRF / 加权融合
- query rewrite 多路召回
- 分阶段 top-k

### 第三阶段：升级索引和上下文构建

目标：

- 提升答案稳定性与可解释性

建议事项：

- 结构化切块
- metadata schema
- parent-child retrieval
- source-aware prompt
- 引用输出

### 第四阶段：建立评测和模型路由

目标：

- 从“感觉变好”切换到“数据证明变好”

建议事项：

- 建立评测集
- 离线评测脚本
- 模型路由
- 阈值拒答
- A/B 对比

---

## 十三、总结

从工程结构上看，这个项目已经有比较完整的 RAG 系统雏形：

- 分层清楚
- 模块职责明确
- 已经有向量库、改写、rerank、历史、API、容器等核心组件

但从“当前真实技术状态”看，它还没有进入一个成熟可评估的 RAG 系统阶段，主要原因不是“模型不够强”，而是：

- 检索链路还没有完全打通
- 混合检索还停留在设计意图
- 数据结构在模块之间不统一
- 索引管理、去重、版本化、来源管理不完整
- 新旧代码入口并存，工程边界不够稳定

如果只允许给一个最重要判断，我的结论是：

**这套系统现在最需要的不是立刻换一个更强的大模型，而是先把“检索正确性、数据结构一致性、混合检索、索引治理和评测闭环”补齐。**

对 RAG 系统来说，很多时候真正决定上限的顺序是：

1. 文档质量
2. 切块与 metadata
3. 召回与 rerank
4. 上下文构建
5. 生成模型

也就是说，在当前阶段，最值得投入的改进方向依次是：

- 先修系统正确性
- 再做混合检索
- 再做索引治理和 chunk 优化
- 最后再系统化升级模型与路由

如果这些基础能力补齐，这个项目是有机会从“课程式/demo 式 RAG”走到“中小规模生产可用 RAG 服务”的。
