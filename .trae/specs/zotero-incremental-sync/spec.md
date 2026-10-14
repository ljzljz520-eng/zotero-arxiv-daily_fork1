# Zotero 增量同步与内容寻址嵌入缓存 - 产品需求文档

## Overview
- **Summary**: 为 zotero-arxiv-daily 增加基于 Zotero Web API version 机制的增量语料同步层，以及按"内容 hash × 嵌入模型/版本"寻址的本地向量缓存；新增/更新/删除/移动只作用于受影响数据，水位与缓存事务原子提交，并产出每轮同步审计清单。
- **Purpose**: 当前 `Executor.fetch_zotero_corpus()` 每次运行全量拉取整个文库、reranker 对全部语料重新嵌入。文库变大后慢且贵（API 费用 / GitHub Actions 时长），且删除、移动、回收站变化只能靠全量数据隐式体现，没有水位、没有崩溃恢复。
- **Target Users**: 在 GitHub Actions（ephemeral runner）或本地定时运行本项目、拥有只读 Zotero API key 的用户。

## Goals
- 持久化 libraryVersion（同步水位）、每个 item 的 version、collection membership 与集合层级。
- 按 Zotero 增量 API 正确应用新增、内容/元数据更新、删除（tombstone + 回收站 + 对账兜底）与 collection 移动。
- 对规范化的标题+摘要做内容寻址缓存（content hash + embedding model/version fingerprint），只重算受影响向量。
- 缓存提交与水位严格原子一致：任何失败/中断都不会出现"宣布同步完成但缺向量"。
- 正确处理：分页水位、删除 tombstone 过期、API key 权限变化、同内容下的嵌入模型升级、同步中断续跑。
- 每轮同步生成机器可读 + 人工可核对的审计清单。
- Executor 重排阶段复用缓存的语料向量，只嵌入当日候选论文。

## Non-Goals
- 双向同步 / 任何对 Zotero 的写操作（本项目只读）。
- group libraries 的产品化支持（现有配置仅 user library；schema 不强制耦合，但不新增配置入口）。
- 附件、PDF 全文、notes、tags 的同步与索引。
- 对候选论文（arXiv/bioRxiv 等每日新文）做向量持久化——只缓存 Zotero 语料侧向量。
- 多进程/多实例并发同步同一状态目录。
- 状态目录之外的部署系统（S3、自建存储等）接入。

## Background & Context
- 现状：[executor.py](file:///Users/kkcarrot/swe-project/zotero-arxiv-daily_fork1/src/zotero_arxiv_daily/executor.py) 用 `zot.everything(zot.collections())` 与 `zot.everything(zot.items(itemType=...))` 全量拉取；[reranker/base.py](file:///Users/kkcarrot/swe-project/zotero-arxiv-daily_fork1/src/zotero_arxiv_daily/reranker/base.py) 每次对语料与候选全部重新嵌入；无任何本地状态。
- 项目锁定 pyzotero 1.11.0（uv.lock），已核实其能力（httpx 实现，`@retrieve` 装饰器）：
  - `zot.items(since=, itemType=, limit=, includeTrashed=)`、`zot.collections(since=)`、`zot.deleted(since=)`（返回 `{items, collections, searches, tags}`）、`zot.trash(since=)`、`zot.item_versions()` / `collection_versions()`（`format=versions`，返回 key→version dict）、`zot.key_info()`（返回 `userID/username/access`）。
  - `zot.everything(query)` 跟随响应 `Link: rel="next"` 分页；每次请求后 `zot.request` 为 httpx.Response，可读 `Last-Modified-Version` 头。
- Zotero 官方同步协议（[syncing 文档](https://www.zotero.org/support/dev/web_api/v3/syncing)）：版本号单调递增但不保证连续；`?since=<libraryVersion>` 取变更；`/deleted?since=` 取 tombstone；`format=versions` 不受最大返回条数限制；`includeTrashed` 默认 0（回收站 item 不出现在普通 items/versions 列表）；`/keys/current` 用于校验权限。
- 运行环境：GitHub Actions runner 存储不跨运行保留，需要 actions/cache 恢复/上传状态目录（缓存约 7 天未使用会被回收，回收后自动降级为全量重建）。
- 用户已确认两项决策：
  1. 嵌入文本 = **规范化标题 + 摘要拼接**（语料与候选两侧一致；相似度行为相对旧版有变化，首次运行全量重建）。
  2. GitHub Actions 用 **actions/cache** 持久化状态目录。

## Functional Requirements

### 同步状态与存储
- **FR-1**: 使用 Python 标准库 sqlite3 持久化到 `zotero.state_dir`（配置项，env `ZOTERO_STATE_DIR` 可覆盖，默认 `.zotero_sync`）下的单个 SQLite 文件；含 schema 版本与初始化/迁移逻辑；目录不存在时自动创建。
- **FR-2**: 状态库至少持久化：库身份（user_id/username/library_type）、已提交 libraryVersion 水位、当前 embedding fingerprint、每个合格类型 item 的 key/version/title/abstract/dateAdded/content_hash/trash 标记、collection key/version/name/parent、item↔collection membership、内容寻址向量 blob、每轮同步审计记录。
- **FR-3**: 配置的库身份（zotero.user_id）与状态库中记录不一致时，拒绝同步并抛出明确错误（防止把另一个库的变更写入现有缓存），不得静默重置或继续。

### 远端校验
- **FR-4**: 每轮同步先调用 key 信息端点：校验返回 userID 与配置一致、`access.user.library` 为真。403/401/404 或权限不足时抛出专用权限错误，本轮不写任何业务数据、不推进水位，但在审计中记录 `failed_permission`；权限恢复后下一轮可从原水位继续。

### 增量同步协议
- **FR-5**: 同步按固定顺序执行：
  1. key 校验（FR-4）；
  2. `items?format=versions` 全量 key→version 映射，取该响应 `Last-Modified-Version` 作为本轮快照版本 V；若 V 等于本地水位 W 且 fingerprint 未变化，本轮短路成功（no-op）；
  3. `items?since=W`（itemType 限定 `conferencePaper || journalArticle || preprint`，limit=100，跟随分页）拉取变更 item 完整 JSON；
  4. `collections?since=W` 拉取变更集合（含 name/parentCollection/version）；
  5. `deleted?since=W` 取永久删除 tombstone，`items/trash?since=W` 取移入回收站的 key；
  6. 用第 2 步的全量 version 映射做对账：本地 tracked key 在映射中缺失的（删除且 tombstone 已过期 / 已在回收站 / 权限范围内消失）一律按删除处理；远端 version 与本地不一致但未在 since 列表出现的对象兜底补取/更新。
- **FR-6**: 集合层级同样以全量 collection versions 映射对账：本地有、远端无的集合删除；删除集合时在同一事务内摘除全部 item 对该集合的 membership（item 向量不重算）；成员关系以变更 item payload 的 `data.collections` 为准做集合差量。
- **FR-7**: 变更分类：added（本地无此 key）、updated（version 变化，再细分 content_changed / metadata_only）、moved（content hash 不变但 membership 集合变化，记录新增/移除的 collection key 与解析后的路径）、deleted（区分 tombstoned / trashed / reconciled）。
- **FR-8**: 分页正确性：只有在所有变更页、collections、deleted、trash 全部获取并处理完成后才允许提交水位 V；水位提交后同步过程中新出现的更高版本对象（分页期间文库继续前进）留待下一轮 `since=V` 处理，不得漏（下轮必被拾取）也不得把水位越过 V。

### 内容寻址嵌入缓存
- **FR-9**: 统一的文本规范与嵌入输入：`normalize_text()` 做 Unicode NFKC、折叠连续空白、去首尾空白；`build_embedding_text(title, abstract)` 用规范化标题与摘要按固定分隔拼接；`content_hash = sha256(embedding_text)`。语料侧与候选侧必须使用同一函数。
- **FR-10**: embedding fingerprint 由 reranker 后端标识（local/api）、配置模型名、观测到的向量维度，以及可选显式版本配置 `reranker.embedding_model_version`（默认 null）组成；blob 以 `(fingerprint, content_hash)` 为主键，向量以 float32 字节存储并记录维度。
- **FR-11**: 仅以下情况请求嵌入：新 item 且摘要非空、item 内容 hash 变化、fingerprint 变化（模型升级）；membership-only 移动、仅元数据（如版本号）变化必须零嵌入调用。空摘要 item 保留跟踪行与 membership 但不嵌入、不进入语料；日后摘要补齐时自动嵌入并进入语料。
- **FR-12**: 同内容模型升级：fingerprint 变化时，为全部活跃合格 item 按新 fingerprint 准备向量；未变化内容不重复请求（若 blob 表中已存在相同 fingerprint+hash 则复用）；提交成功后旧 fingerprint 的链接与 blob 被清理；审计记录 model upgrade 及重算/复用数量。

### 原子提交与崩溃恢复
- **FR-13**: 采用多阶段协议：
  - Phase A（拉取）：所有远端分页数据取回到内存，中途失败则什么都不写；
  - Phase B（blob 预取）：把缺失的 `(fingerprint, content_hash)` 向量分批嵌入，以独立小事务写入 blob 表——blob 内容寻址、不可变，提前提交无破坏；崩溃后这些 blob 可被复用，不重复嵌入；
  - Phase C（原子提交）：在**单个** SQLite 事务（BEGIN IMMEDIATE → COMMIT）内完成 item/collection/membership 全部 upsert 与删除、item→blob 链接、水位推进到 V、fingerprint 记录、审计行写入；COMMIT 之前在事务内校验不变量"每个活跃且摘要非空的 tracked item 都存在当前 fingerprint 的向量链接"，违反则抛错并回滚，水位保持 W；
  - Phase D（提交后）：清理无引用 blob、写审计 JSON 文件、输出日志摘要（这些派生操作失败不回滚已提交状态）。
- **FR-14**: 任意阶段中断（进程被杀、异常、网络错误、嵌入 API 错误）后重跑：从水位 W 重新 since 重放（操作幂等：upsert 按 key+version、blob 按内容寻址去重）；已存在 blob 不重复请求嵌入；成功后不变量成立。

### 审计
- **FR-15**: 每轮同步（含失败轮）产出审计记录：状态 DB 内一行 + `state_dir/audit/` 下原子写入的 JSON 文件，字段包含：run 起止时间（UTC ISO）、最终状态（success/no_changes/failed/failed_permission）、库身份、W→V、是否全量对账/首轮、远端请求页数与端点计数、added/updated(content_changed/metadata_only)/moved(每项含 key 与 added/removed 路径)/deleted(tombstoned/trashed/reconciled) 计数及 key 列表、embeddings_computed / embeddings_reused、model_upgraded 标志与新旧 fingerprint、错误信息。成功轮日志打印人读摘要。

### Executor 接入
- **FR-16**: `Executor` 启动时构造同步器（复用已配置的 reranker 作为嵌入提供者），执行同步后从状态库加载语料：返回有序的 `CorpusPaper` 列表（新增内部字段携带 item key，不破坏现有 dataclass 构造方式）与严格对齐的向量矩阵；`filter_corpus` 之后按 item key 取对应向量子集，顺序对齐。
- **FR-17**: reranker 接口演进：BaseReranker 新增 `embed_texts(texts) -> np.ndarray`（api 端懒构造并复用 OpenAI client、按 batch_size 分批；local 端懒加载并复用 SentenceTransformer 实例）；`rerank()` 支持传入语料向量矩阵——有则只嵌入候选、无则保持旧行为；ApiReranker/LocalReranker 实现 fingerprint 与维度观测。
- **FR-18**: 配置与工作流：`config/base.yaml` 新增 `zotero.state_dir`（`${oc.env:ZOTERO_STATE_DIR,.zotero_sync}`）与 `reranker.embedding_model_version: null`（带注释）；`.github/workflows/main.yml` 在运行前用 actions/cache 恢复状态目录、运行后 `if: always()` 上传；`.gitignore` 忽略状态目录。

## Non-Functional Requirements
- **NFR-1（测试离线）**: 全部新测试遵循现有约定（pytest monkeypatch + SimpleNamespace stub，无 unittest.mock、无 Docker、无网络）；需要真实 sentence-transformers 模型的测试标 `@pytest.mark.slow`，默认 `uv run pytest` 跳过。
- **NFR-2（零新依赖）**: 仅使用标准库与现有依赖（sqlite3、numpy、pyzotero、openai 等），不新增第三方包。
- **NFR-3（失败安全）**: 代码结构上保证"水位更新"与"向量链接写入"同事务；不存在任何先推进水位后补向量的代码路径。
- **NFR-4（向后兼容）**: 无状态目录时首轮自动全量同步，不需要手工初始化；现有配置项与 `filter_corpus` glob 语义不变；测试 fixture（conftest/canned_responses）同步更新。
- **NFR-5（大库成本）**: 全量 versions 映射每轮仅 1 个紧凑请求；变更 item 每页 100；嵌入按批（复用 `reranker.api.batch_size`，默认 64）；本地模型每次进程只加载一次；无内容变化时同步轮应只有 2 个轻请求（versions + deleted/trash 短路前可省——见 FR-5 第 2 步短路）。
- **NFR-6（Python 3.13）**: 类型标注与语法与 `requires-python >=3.13` 一致。

## Constraints
- **Technical**: pyzotero 1.11.0 的 httpx 客户端行为（`zot.request.headers`、`everything()` 分页、`deleted()` 内部去 limit）；SQLite 单文件多事务；Hydra/OmegaConf 配置体系；GitHub Actions 缓存语义。
- **Business**: 只读 API key；零运行成本目标（GitHub Actions 免费额度、嵌入 API 按 token 计费）。
- **Dependencies**: 现有 `pyzotero>=1.10.0` 约束（锁定 1.11.0）；numpy；actions/cache 第三方 action 已在仓库工作流中被同类 action 使用（checkout/setup-uv）。

## Assumptions
- `/items/trash` 与普通 items 端点共享 `since` 参数（官方文档说明 trash 为 items 端点变体，同步指南建议 includeTrashed=1；本实现选择显式 trash 端点以区分删除类别）。
- `format=versions` 全量映射单请求返回全部 key（官方文档明确不受最大条数限制），作为删除对账的正确性兜底，从而不依赖 tombstone 保留时长。
- 单用户库、单进程运行；状态目录随仓库/用户隔离。
- actions/cache 被回收时表现为状态目录缺失，首轮全量重建为预期降级路径。
- Zotero item 的 version 在 collection 移动/集合删除导致 membership 变化时会递增；即使个别情况不递增，collections 全量对账与 payload membership 差量也能在下一轮修正路径。

## Acceptance Criteria

### AC-1: 首轮全量同步正确建库
- **Type**: `rule`
- **Given**: 空状态目录，stub 远端含多个合格类型 item（含空摘要 item）、父子集合层级
- **When**: 执行一次同步
- **Then**: items/collections/membership/水位=快照 V/全部非空摘要 item 的当前 fingerprint 向量均落库；空摘要 item 无向量；审计状态 success 且计数与 stub 一致
- **Pass Condition**: 断言上述 DB 行、水位、向量链接数与审计计数
- **Evidence**: tests/sync 新增测试 + DB 查询输出

### AC-2: 增量新增/更新且只重算受影响向量
- **Type**: `rule`
- **Given**: 已完成首轮同步的状态库
- **When**: 第二轮远端新增 1 个 item、1 个 item 修改摘要、1 个 item 仅涨 version（内容 hash 不变）
- **Then**: added=1、content_changed=1（该 item 重算）、metadata_only=1（零嵌入调用、向量复用）、水位推进至新 V
- **Pass Condition**: 嵌入调用次数恰好=新增+内容变化数；其余 item 链接 hash 不变
- **Evidence**: 增量同步测试中嵌入调用计数

### AC-3: collection 移动零嵌入并记录路径级变化
- **Type**: `rule`
- **Given**: 已同步 item 属于集合 A/B
- **When**: 远端 payload 使其 collections 从 {A} 变为 {B}（version 涨、内容不变）
- **Then**: membership 更新、解析路径变化、embeddings_computed=0；审计 moved 项含 item key 与移除/新增的 collection 路径
- **Pass Condition**: 路径断言 + 嵌入计数为 0 + 审计字段断言
- **Evidence**: 移动场景测试

### AC-4: tombstone 与回收站删除生效
- **Type**: `rule`
- **When**: `deleted(since)` 含一个 key、`trash(since)` 含另一个 key
- **Then**: 两 item 的行/membership/向量链接均删除，blob 在 GC 后无引用；审计分别记 tombstoned 与 trashed 及 key 列表
- **Pass Condition**: DB 查询无残留 + 审计分类计数
- **Evidence**: 删除场景测试

### AC-5: tombstone 过期/缺失由全量 versions 对账兜底
- **Type**: `rule`
- **When**: 本地 tracked key 既不在 since payload、也不在 deleted/trash 响应，但全量 versions 映射中缺失
- **Then**: 按 reconciled 删除并在审计记录；水位正常提交；其余 item 不受影响
- **Pass Condition**: 对账删除测试
- **Evidence**: 测试断言 reconciled 计数

### AC-6: 跨页同步的分页水位
- **Type**: `rule`
- **Given**: 变更 item 超过单页（stub 多页 + next 链接）
- **When**: 同步在第二页获取时注入异常；随后重跑
- **Then**: 第一次水位停留 W、无部分提交；重跑成功消费全部页后水位=V，无重复/遗漏
- **Pass Condition**: 两次运行后的水位、行数、审计计数
- **Evidence**: 分页+中断测试

### AC-7: API key 权限变化安全失败
- **Type**: `rule`
- **When**: key 信息端点 403 / userID 不匹配 / access.user.library=false 三种情形
- **Then**: 抛专用权限错误，业务表与水位不变；审计记录 failed_permission；恢复权限后下一轮从原水位成功同步
- **Pass Condition**: 三参数化场景断言
- **Evidence**: 权限测试

### AC-8: 同内容模型升级
- **Type**: `rule`
- **When**: fingerprint 变化（切换模型或设置 embedding_model_version）
- **Then**: 所有活跃非空摘要 item 以新 fingerprint 链接向量（已存在 blob 复用），旧 fingerprint 链接与 blob 清理，水位与新链接在同一事务提交，审计含新旧 fingerprint 与 model_upgraded=true
- **Pass Condition**: 新旧 blob/链接查询 + 审计断言
- **Evidence**: 模型升级测试

### AC-9: 中断后不缺向量且不重复付费
- **Type**: `rule`
- **Given**: Phase B 分批嵌入，在第二批注入异常；Phase C 前异常
- **When**: 重跑直至成功
- **Then**: 失败后水位=W；第一批 blob 已保留；重跑对第一批 hash 嵌入调用为 0；成功后 SQL 不变量"活跃非空摘要 item 全部存在当前 fingerprint 向量链接"成立
- **Pass Condition**: 嵌入调用计数 + 不变量 SQL 查询
- **Evidence**: 崩溃恢复测试

### AC-10: 水位与缓存原子提交的代码不变量
- **Type**: `rule`
- **Pass Condition**: 同步提交路径中水位 upsert、item_embeddings 链接写、审计 success 行处于同一事务（无中间 commit）；通过代码审查与"提交回调中抛错→整体回滚"测试证明
- **Evidence**: 代码审查 + 事务回滚注入测试

### AC-11: Executor 端到端复用语料缓存
- **Type**: `rule`
- **When**: 用现有 stub 体系连续执行两次 Executor.run()
- **Then**: 首次建立缓存；第二次 rerank 时语料侧零嵌入调用、候选侧正常嵌入；两次邮件均正常发送；include/ignore glob 过滤后的语料与向量顺序严格对齐
- **Pass Condition**: 更新后的 test_executor E2E 断言通过
- **Evidence**: pytest tests/test_executor.py

### AC-12: 配置、工作流与忽略规则
- **Type**: `rule`
- **Pass Condition**: base.yaml 含 state_dir/embedding_model_version 默认与注释；custom.yaml 可覆盖；main.yml 含 cache 恢复步骤与 `if: always()` 的上传步骤且路径与 env 一致；.gitignore 含状态目录；Hydra compose 测试通过
- **Evidence**: 配置审查 + conftest compose + 工作流 YAML 断言

### AC-13: 审计清单完整且原子落盘
- **Type**: `rule`
- **Pass Condition**: 成功轮与失败轮均有 DB 审计行；JSON 文件字段完整（FR-15 所列）且与 DB 行一致；JSON 为原子写（临时文件+replace）；moved 项含路径
- **Evidence**: 审计测试 + 文件检查

### AC-14: 空摘要与集合删除的成员关系维护
- **Type**: `rule`
- **When**: 空摘要 item 补摘要；某集合被远端删除
- **Then**: 补摘要后 item 进入语料并出现向量；删集合后 membership 摘除、路径不再包含该集合、item 不重算嵌入
- **Pass Condition**: 组合场景测试
- **Evidence**: 测试断言

### AC-15: 同步进行中文库继续变化不越水位
- **Type**: `rule`
- **Given**: 分页响应中混入 version > 本轮快照 V 的 item
- **When**: 本轮提交后再跑一轮（stub 切换到 V'）
- **Then**: 本轮水位只到 V；更高版本 item 下一轮 since=V 被拾取并更新，不丢失、不依赖客户端去重之外的副作用
- **Pass Condition**: 两轮状态断言
- **Evidence**: 快照版本测试

### AC-16: 测试套件全绿
- **Type**: `rule`
- **Pass Condition**: `uv run pytest`（非 slow）全部通过，新增 sync 模块测试覆盖上述场景；CI 使用的 `uv run pytest -m ""` 在可接受的 slow 测试声明下不被破坏（slow 测试默认环境可跳过）
- **Evidence**: 命令输出

### AC-17: 同步层设计质量
- **Type**: `rubric`
- **Dimension**: 模块职责清晰度（state/content/client/syncer/audit 分离）、错误处理一致性、命名与注释可读性
- **Scale**: 1-5
- **Anchors**: 1 = 逻辑耦合在 Executor 中难以测试；3 = 分层存在但边界模糊；5 = 各模块单一职责、纯函数易测、错误类型明确
- **Pass Threshold**: >= 4
- **Evidence**: 独立代码审查

### AC-18: 审计输出的可观测性
- **Type**: `rubric`
- **Dimension**: 仅凭日志摘要 + 审计 JSON 即可人工核对一轮同步发生了什么、是否健康
- **Scale**: 1-5
- **Anchors**: 1 = 只有成功/失败字样；3 = 有计数但难定位具体 item；5 = 计数、key、路径、fingerprint、水位、耗时、错误俱全
- **Pass Threshold**: >= 4
- **Evidence**: 独立审查审计样例

## Open Questions
- 无（嵌入文本构成与 Actions 持久化方式已确认；其余按 Assumptions 执行）。
