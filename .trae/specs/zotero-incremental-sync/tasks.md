# Zotero 增量同步与内容寻址嵌入缓存 - 实现计划

## Task 1: SQLite 同步状态存储层（sync/state.py）
- **Status**: `done`
- **Priority**: high
- **Depends On**: None
- **Description**:
  - 新建 `src/zotero_arxiv_daily/sync/` 包（`__init__.py`）与 `state.py`
  - 定义 schema（带 `schema_version` 与 `user_version` 初始化）：
    - `library_state`：单行，library_id/library_type/username/schema 版本、`library_version` 水位、`embedding_fingerprint`、`last_sync_at`
    - `items`：key(PK)、version、item_type、title、abstract、added_date、content_hash、trashed、exists_remote
    - `collections`：key(PK)、version、name、parent_key
    - `item_collections`：(item_key, collection_key) 联合主键
    - `embedding_blobs`：(fingerprint, content_hash) PK、dim、vector BLOB(float32)、created_at
    - `item_embeddings`：(item_key, fingerprint) PK、content_hash（应用层校验 blob 存在）
    - `sync_audit`：自增 id、started_at、finished_at、status、old_version、new_version、detail_json
  - 封装连接工厂（`connect(state_dir)`，WAL 或默认 rollback journal、`BEGIN IMMEDIATE` 支持、外键开）、目录自动创建、库身份读写、水位读写、各类 upsert/delete 的薄 SQL 方法、blob 独立事务写入、审计行写入
  - 提供不变量查询：`missing_vectors(fingerprint)` → 活跃且摘要非空但缺当前 fingerprint 链接的 item key 集合；`unreferenced_blobs()`；`active_item_hashes()`
  - 不写业务决策逻辑（只存储）；不允许在存储层内出现"水位+链接分两次 commit"的便捷方法
- **Acceptance Criteria Addressed**: AC-1, AC-9, AC-10, AC-14（存储部分）
- **Test Requirements**:
  - `rule` TR-1.1: 空目录初始化后所有表与 schema_version 存在（查询 sqlite_master），重复初始化幂等
  - `rule` TR-1.2: 一个事务内写 items+membership+水位后关闭重开，数据完整可见；事务中抛错则回滚（水位与链接同生共死，AC-10 存储侧证据）
  - `rule` TR-1.3: blob 独立事务可先于水位提交；回滚主事务后 blob 仍在（Phase B 语义）
  - `rule` TR-1.4: `missing_vectors` 对空摘要 item、trashed item、已链接 item 三种情况返回正确集合
  - `rule` TR-1.5: `unreferenced_blobs` 能识别被删 item/旧 fingerprint 的无引用 blob
- **Notes**: 全部纯 sqlite3，离线测试，tmp_path fixture

## Task 2: 内容规范化与 reranker 嵌入接口演进
- **Status**: `done`
- **Priority**: high
- **Depends On**: None
- **Description**:
  - 新建 `src/zotero_arxiv_daily/sync/content.py`：
    - `normalize_text(s)`：NFKC、折叠空白（含全角空格）、strip；None/非 str 安全处理
    - `build_embedding_text(title, abstract)`：规范化标题+固定分隔+规范化摘要
    - `content_hash(title, abstract)`：sha256 hex
  - `reranker/base.py`：
    - 新增抽象/基类方法 `embed_texts(self, texts: list[str]) -> np.ndarray`（返回 float32，不做归一化）
    - 新增 `embedding_fingerprint_base(self) -> str`（后端+模型+可选 embedding_model_version）与 `resolve_fingerprint(base, dim)`
    - `get_similarity_score` 重构为调用 embed_texts 两侧 + 归一化点积，保持数组形状与现有断言
    - `rerank(candidates, corpus, corpus_embeddings=None)`：传入语料矩阵时只嵌入候选（候选文本同样用 `build_embedding_text(title, abstract)`），时间衰减加权逻辑不变
  - `reranker/api.py`：懒构造并复用 OpenAI client；embed_texts 按 batch_size 分批、返回矩阵；fingerprint=`api:{model}:{version?}:dim{dim}`（dim 首次嵌入后解析）
  - `reranker/local.py`：懒加载并复用 SentenceTransformer；embed_texts 调 encode；fingerprint=`local:{model}:{version?}:dim{dim}`
  - config 新增 `reranker.embedding_model_version: null`，读取处缺省 null 安全
- **Acceptance Criteria Addressed**: AC-17, AC-11（嵌入侧）, FR-9, FR-10, FR-17
- **Test Requirements**:
  - `rule` TR-2.1: normalize/build_embedding_text/content_hash 的规范化用例（大小写不改、全角/多空白折叠、NFKC、标题或摘要为空的行为、相同输入同 hash、空白差异不影响 hash）
  - `rule` TR-2.2: ApiReranker.embed_texts 用 stub client 按 batch 分批调用并拼接矩阵；client 只构造一次；rerank 传入 corpus_embeddings 时 stub 收到的嵌入输入仅含候选
  - `rule` TR-2.3: fingerprint base 含后端与模型名；resolve_fingerprint 追加 dim；embedding_model_version 改变会改变 fingerprint
  - `rule` TR-2.4: 现有 reranker/api 相关测试全部通过（get_similarity_score 形状与余弦语义不回归）
  - `rule` TR-2.5: LocalReranker 用注入 fake encoder 的子类（不下载模型）验证 embed_texts/复用/fingerprint；真实模型路径仅保留 slow 标记
- **Notes**: 语料与候选必须共用 build_embedding_text，防止两侧文本构成漂移

## Task 3: Zotero 增量客户端封装（sync/zotero_client.py）
- **Status**: `done`
- **Priority**: high
- **Depends On**: None
- **Description**:
  - 新建 `sync/zotero_client.py`，以 pyzotero 1.11.0 为后端定义窄接口 `IncrementalZoteroClient`：
    - `check_key(expected_user_id) -> KeyIdentity`：调 `key_info()`，校验 userID 与 `access.user.library`；401/403/404 映射为 `SyncPermissionError`
    - `library_versions_snapshot() -> (dict[key, version], V)`：`item_versions()` 全量映射 + 从 `zot.request.headers["Last-Modified-Version"]` 读 V
    - `collection_versions_snapshot() -> dict`
    - `changed_items(since_watermark) -> list[payload]`：`items(since=W, itemType="conferencePaper || journalArticle || preprint", limit=100, includeTrashed=0)` + everything 分页；返回完整 item 对象（含 key/version/data）
    - `changed_collections(W)`、`deleted_since(W)`（dict）、`trashed_keys(W)`（trash(since=) 取 key）
  - 自定义异常：`SyncError`（基类）、`SyncPermissionError`、`SyncProtocolError`（缺头/畸形响应）
  - 网络/5xx/429 由 pyzotero 自身重试/backoff 抛出，包装为 SyncError，不吞错
  - 构造方式与 Executor 现有 `zotero.Zotero(user_id,'user',api_key)` 一致，便于 monkeypatch
- **Acceptance Criteria Addressed**: AC-7, FR-4, FR-5, FR-6
- **Test Requirements**:
  - `rule` TR-3.1: stub pyzotero（SimpleNamespace，含 request.headers、links、everything 行为）验证各方法 URL 参数（since/itemType/limit）与返回解析
  - `rule` TR-3.2: key_info 403/userID 不符/access.user.library 假值三种场景均抛 SyncPermissionError 且带可读信息
  - `rule` TR-3.3: 从多页（links.next 两次 follow）聚合全部 item；Last-Modified-Version 缺失抛 SyncProtocolError
  - `rule` TR-3.4: deleted/trash payload 正确解析为 key 列表/分类 dict
- **Notes**: stub 扩展现有 canned_responses.make_stub_zotero_client 风格

## Task 4: 同步编排器：变更分类、对账、两阶段提交、审计（sync/syncer.py + sync/audit.py）
- **Status**: `done`
- **Priority**: high
- **Depends On**: Task 1, Task 2, Task 3
- **Description**:
  - 新建 `sync/audit.py`：`AuditReport` 数据结构（FR-15 全字段）、计数器、moved 项（key+added/removed paths）、`to_json()`；`write_atomic(path, text)`（临时文件+os.replace）
  - 新建 `sync/syncer.py` `CorpusSyncer(config, embedding_provider=None, zot_client=None, store=None)`：
    - 启动：开 store；校验库身份一致（不一致 SyncError，FR-3）
    - 执行 FR-5 固定顺序；V==W 且 fingerprint 未变 → no-op 成功审计（短路，除 versions/key 外无请求）
    - 内存中计算目标状态与变更分类（added/content_changed/metadata_only/moved/deleted 三子类）；集合路径解析（父链，父缺失防御性回退为集合名）
    - Phase B：收集需要的 (fingerprint, hash, text)；已有 blob 去重；按 batch_size 调 embed_texts；每批独立事务写 blob；失败抛错（水位不动）
    - Phase C：BEGIN IMMEDIATE 单事务：upsert items（version/hash/title/abstract/dateAdded）、collections、membership 全量差量替换、删除（tombstone/trash/reconcile，含链接）、旧 fingerprint 清理（模型升级）、item_embeddings 链接、写 fingerprint 与水位 V、写 success 审计行；提交前 `missing_vectors` 必须为空否则 raise 回滚
    - Phase D：GC 无引用 blob；审计 JSON 原子写 `state_dir/audit/<utc-timestamp>-<status>.json`；loguru 摘要日志
    - 失败轮：在独立事务写 failed/failed_permission 审计行（不动水位）并尝试落盘 JSON；异常照常上抛
    - `load_corpus() -> (list[CorpusPaper], np.ndarray)`：活跃、非空摘要、有当前 fingerprint 链接的 item，向量矩阵与列表严格对齐；CorpusPaper 增加带默认值的 `key: str = ""` 字段；路径由 collection 层级现算
  - 首轮（无水位）自动全量；身份不匹配不自动重置
- **Acceptance Criteria Addressed**: AC-1, AC-2, AC-3, AC-4, AC-5, AC-6, AC-8, AC-9, AC-10, AC-13, AC-14, AC-15, AC-18
- **Test Requirements**:
  - `rule` TR-4.1: 首轮全量（AC-1）：行数/水位/向量链接/审计计数；空摘要 item 不嵌入；路径父链正确
  - `rule` TR-4.2: 增量分类（AC-2）：新增/内容改/元数据改三类，嵌入调用计数精确
  - `rule` TR-4.3: 移动（AC-3）：membership/路径更新、零嵌入、审计 moved 路径
  - `rule` TR-4.4: 删除三类（AC-4/AC-5）：tombstone/trash/reconcile 分别落账，GC 后无 blob 残留
  - `rule` TR-4.5: 分页中断（AC-6）：第二页异常 → 水位 W 无部分数据；重跑成功且不重不漏
  - `rule` TR-4.6: 模型升级（AC-8）：fingerprint 变化后全量切换、旧 blob/链接清理、水位与新链接同事务
  - `rule` TR-4.7: Phase B 中断（AC-9）：批 2 失败 → 水位 W、批 1 blob 在；重跑批 1 零嵌入调用；成功后 missing_vectors 为空
  - `rule` TR-4.8: Phase C 提交注入异常（在 commit 前回调抛错）→ 全部回滚（AC-10）
  - `rule` TR-4.9: 权限失败（AC-7 同步器侧）：failed_permission 审计行落盘且业务数据不变，恢复后续跑成功
  - `rule` TR-4.10: 快照越界版本（AC-15）：version>V 对象下轮拾取
  - `rule` TR-4.11: 审计 JSON（AC-13）：字段齐全、与 DB 行一致、原子文件存在；失败轮 JSON 也存在
  - `rule` TR-4.12: 空摘要补全 + 删集合（AC-14）：语料进出正确、删集合零嵌入
  - `rule` TR-4.13: load_corpus 对齐性：CorpusPaper 顺序与矩阵行一一对应，向量等于 blob 解码
  - `rubric` TR-4.14: 审计可观测性（对应 AC-18）；scale 1-5；anchors 1/3/5 同 AC-18；threshold >= 4；evidence=审查样例 JSON 与日志
- **Notes**: 测试用 fake embedding provider（记录调用文本的可注入函数）+ Task 3 的 stub client，全部离线

## Task 5: Executor 接入与端到端测试
- **Status**: `done`
- **Priority**: high
- **Depends On**: Task 4
- **Description**:
  - `executor.py`：
    - `__init__` 构造 `CorpusSyncer(config, embedding_provider=self.reranker)`
    - `run()` 以 `syncer.sync()` + `syncer.load_corpus()` 替换 `fetch_zotero_corpus()` 的网络路径；`filter_corpus` 后按 `CorpusPaper.key` 重排向量子集
    - rerank 调用传入语料矩阵；空语料报错文案保留
    - 删除/保留 `fetch_zotero_corpus`：若无其他引用则删除（同步删除其直接测试，替换为同步器测试）
  - `tests/canned_responses.py`：stub zotero client 增补 versions/deleted/trash/key_info/分页能力（供 Task 3/4 与 E2E 共用）
  - 更新 `tests/test_executor.py`：E2E 连续两轮；断言第二轮语料侧零嵌入、候选侧嵌入、邮件正常、过滤对齐
- **Acceptance Criteria Addressed**: AC-11, AC-16
- **Test Requirements**:
  - `rule` TR-5.1: 两轮 E2E（AC-11）：首轮建缓存、二轮增量；语料嵌入调用计数第二轮为 0；候选两轮均嵌入
  - `rule` TR-5.2: include/ignore glob 过滤后向量与语料顺序对齐（抽样比对相似度矩阵行）
  - `rule` TR-5.3: 同步权限错误时 run 不上发邮件且错误上抛（不静默空邮）
  - `rule` TR-5.4: 既有 send_empty/no papers 场景测试保持通过
- **Notes**: monkeypatch 风格与现有 E2E 完全一致

## Task 6: 配置、GitHub Actions 持久化与忽略规则
- **Status**: `done`
- **Priority**: medium
- **Depends On**: Task 4
- **Description**:
  - `config/base.yaml`：新增 `zotero.state_dir: ${oc.env:ZOTERO_STATE_DIR,.zotero_sync}` 与 `reranker.embedding_model_version: null`，注释说明用法/模型升级场景
  - `.github/workflows/main.yml`：
    - env 增加 `ZOTERO_STATE_DIR: ${{ github.workspace }}/.zotero_sync`
    - Run script 前增加 actions/cache@v4 恢复（path=.zotero_sync，key 含 OS 与 ZOTERO_ID 哈希变量不可用则用固定 key+restore-keys）
    - 运行后增加 `if: always()` 的 cache 上传步骤（actions/cache 在命中已存在 key 时会自动失败上传——用单独 step 或接受既有实践；采用官方推荐：单个 cache step 在 job 级，post 运行自动保存；恢复与保存都由同一 action 完成，故只需在脚本前放置一个 actions/cache@v4 step，其 post 在 always 语义下执行）
  - `.gitignore` 增加 `.zotero_sync/`
  - conftest 测试 fixture 覆盖新配置键
- **Acceptance Criteria Addressed**: AC-12, NFR-4
- **Test Requirements**:
  - `rule` TR-6.1: Hydra compose 后 `zotero.state_dir` 与 `reranker.embedding_model_version` 有正确默认值；env 覆盖生效（monkeypatch ZOTERO_STATE_DIR）
  - `rule` TR-6.2: main.yml 为合法 YAML，包含 actions/cache step（uses: actions/cache@）与 ZOTERO_STATE_DIR env，路径一致
  - `rule` TR-6.3: .gitignore 含状态目录条目

## Task 7: 全量回归验证
- **Status**: `done`
- **Priority**: high
- **Depends On**: Task 5, Task 6
- **Description**:
  - 运行 `uv run pytest`（默认非 slow）全绿；运行 `uv run pytest -m "" --cov=src/zotero_arxiv_daily` 评估 sync 新模块覆盖率，补明显空洞
  - 自查所有 AC 证据已记录到 tasks.md Completion Evidence
- **Acceptance Criteria Addressed**: AC-16, AC-17
- **Test Requirements**:
  - `rule` TR-7.1: `uv run pytest` 退出码 0，无 warning 级失败
  - `rule` TR-7.2: 新模块分支覆盖关键路径（success/no-op/failed/permission/upgrade/interrupt 均被执行）
  - `rubric` TR-7.3: 设计质量（对应 AC-17）；scale 1-5；anchors 1/3/5 同 AC-17；threshold >= 4；evidence=独立审查

## Completion Evidence（2026-06-14 实现完成）

**回归**：`uv run pytest -q` → 196 passed, 1 deselected（slow 真实模型测试在带 `-m ""` 的全量跑中亦通过）。
**覆盖率**：`zotero_arxiv_daily.sync` 包（state/content/zotero_client/audit/syncer/__init__）**100% 行覆盖**；不可达的纯防御分支显式标注 `# pragma: no cover` 并附原因。

| 任务 | 关键产物 | TR 证据 |
|---|---|---|
| Task 1 | [state.py](file:///Users/kkcarrot/swe-project/zotero-arxiv-daily_fork1/src/zotero_arxiv_daily/sync/state.py)：7 表 schema、`transaction()` BEGIN IMMEDIATE、blob/link/missing_vectors/GC/audit DAO | tests/sync/test_state.py 全过（TR-1.1~1.5 + 上下文管理/审计查询） |
| Task 2 | [content.py](file:///Users/kkcarrot/swe-project/zotero-arxiv-daily_fork1/src/zotero_arxiv_daily/sync/content.py)、reranker base/api/local：embed_texts 分批、fingerprint base+dim、corpus_embeddings 复用 | test_content/test_base/api/local（TR-2.1~2.5） |
| Task 3 | [zotero_client.py](file:///Users/kkcarrot/swe-project/zotero-arxiv-daily_fork1/src/zotero_arxiv_daily/sync/zotero_client.py)：key 校验、versions 快照 V 取自 last-modified-version、分页、deleted/trash、items_by_keys reconcile、错误映射 | test_zotero_client.py 20 个（TR-3.1~3.4 + 畸形 versions） |
| Task 4 | [syncer.py](file:///Users/kkcarrot/swe-project/zotero-arxiv-daily_fork1/src/zotero_arxiv_daily/sync/syncer.py) + [audit.py](file:///Users/kkcarrot/swe-project/zotero-arxiv-daily_fork1/src/zotero_arxiv_daily/sync/audit.py)：FR-5 全流程、两阶段、Phase C 单事务+提交前 missing_vectors 硬门禁、drift 兜底、版本守卫、GC/审计 best-effort | test_syncer.py 38 个（TR-4.1~4.13 全覆盖，另含故障注入：缺链接拒绝提交、维度漂移、畸形 payload、崩溃零付费续跑） |
| Task 5 | [executor.py](file:///Users/kkcarrot/swe-project/zotero-arxiv-daily_fork1/src/zotero_arxiv_daily/executor.py)：删除 fetch_zotero_corpus，sync+load_corpus，过滤后按 key 对齐矩阵 | test_executor.py 15 个（TR-5.1~5.4） |
| Task 6 | [base.yaml](file:///Users/kkcarrot/swe-project/zotero-arxiv-daily_fork1/config/base.yaml) state_dir 默认值、[main.yml](file:///Users/kkcarrot/swe-project/zotero-arxiv-daily_fork1/.github/workflows/main.yml) actions/cache@v4（post 步骤 always 保存）、.gitignore | test_config_workflow.py 4 个（TR-6.1~6.3） |
| Task 7 | 全量回归 + 覆盖率 100%；修复"陈旧 payload 守卫"实现缺陷（改为永不应用 version ≤ 本地版本的 payload，与水位解耦） | TR-7.1/7.2 达成；TR-7.3 见 Task 8 review.md |

**核心不变量的测试证据**：
- 水位/链接/审计原子性：`test_transaction_*`（存储层）、`test_phase_c_callback_*_rolls_back`、`test_missing_vector_blocks_watermark_commit`（blob 在但链接缺失 → 拒绝推进水位，failed 审计）
- 中断安全：`test_paginated_interruption_resumes`、`test_phase_b_batch_failure_*`、`test_interrupted_run_reuses_blobs_with_no_prior_watermark`、`test_upgrade_crash_*`
- 删除三来源 + GC：`test_deletions_tombstone_trash_reconcile_and_gc`
- 模型升级：`test_model_upgrade_switches_fingerprint`（旧链接事务内删除、旧 blob GC、同 hash 跨 fp 复用）
- 权限/身份：`test_permission_failure_then_recovery`、`test_library_identity_mismatch_rejected`

## Completion Evidence 增补：Task 8 独立审查修复轮（2026-06-14）

详见 [review.md](file:///Users/kkcarrot/swe-project/zotero-arxiv-daily_fork1/.trae/specs/zotero-incremental-sync/review.md)。2 Critical + 3 Major + 10 Minor（6 修 / 4 接受）全部处置完毕。

**回归**：`uv run pytest -q` → **211 passed, 1 deselected**；`zotero_arxiv_daily.sync` 包保持 **100% 行覆盖**（`uv run pytest tests/sync --cov=zotero_arxiv_daily.sync`）。

**关键修复与落点**：
- **C1** rerank 矩阵随 recency 同序重排、行数断言前置：[base.py](file:///Users/kkcarrot/swe-project/zotero-arxiv-daily_fork1/src/zotero_arxiv_daily/reranker/base.py#L111-L121)
- **C2** cache key 加 `github.run_id` + 两级 restore-keys（YAML 引号/序列）：[main.yml](file:///Users/kkcarrot/swe-project/zotero-arxiv-daily_fork1/.github/workflows/main.yml#L22-L35)
- **M1** schema v2 `excluded_items` + v1→v2 迁移（[state.py](file:///Users/kkcarrot/swe-project/zotero-arxiv-daily_fork1/src/zotero_arxiv_daily/sync/state.py)）、白名单判定与 excluded drift（[syncer.py](file:///Users/kkcarrot/swe-project/zotero-arxiv-daily_fork1/src/zotero_arxiv_daily/sync/syncer.py)）、`ALLOWED_ITEM_TYPES` 常量（[zotero_client.py](file:///Users/kkcarrot/swe-project/zotero-arxiv-daily_fork1/src/zotero_arxiv_daily/sync/zotero_client.py#L28-L32)）
- **M2** 维度 abort 指引 bump version；`embedding_model_params()` 把 local `encode_kwargs` 纳入稳定指纹（[base.py](file:///Users/kkcarrot/swe-project/zotero-arxiv-daily_fork1/src/zotero_arxiv_daily/reranker/base.py#L46-L69)、[local.py](file:///Users/kkcarrot/swe-project/zotero-arxiv-daily_fork1/src/zotero_arxiv_daily/reranker/local.py#L41-L44)）
- **M3** versions 快照补集扫描（tracked drift / excluded drift / missing key，`v<=V` 守卫）合并 by-key 补取；collections 同构（新增 `collections_by_keys`）
- **m1/m3/m5/m6/m7/m8**：端点计数入审计、ROLLBACK 不掩盖原异常、Phase D 失败不写矛盾审计、身份行移入 Phase C（首轮失败零落盘）、no-op touch last_sync_at、畸形 item version → SyncProtocolError

**新增测试 13 个**：syncer +8（M1×2、M3×4、m1、m8）、state +2（schema 迁移、rollback 掩盖）、zotero_client +1（真实客户端懒构造）、reranker +2（encode_kwargs 指纹变化）；workflow 断言增强；5 个首轮失败测试更新为"身份行不存在"新语义。Rubric：TR-4.14=5/5、TR-7.3=5/5。
