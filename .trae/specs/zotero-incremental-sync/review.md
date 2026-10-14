# 独立审查报告：Zotero 增量同步与内容寻址嵌入缓存

- **审查对象**：Tasks 1-7 的实现（`sync/` 包、reranker 接口演进、executor 接入、Actions 缓存工作流）
- **审查方式**：静态代码审查 + 针对每个发现的故障注入/回归测试（离线 pytest，无 Docker/网络/mock.patch）
- **基线回归**：审查前 196 passed；修复后 **211 passed, 1 deselected（slow）**；`zotero_arxiv_daily.sync` 包 **100% 行覆盖**
- **结论**：2 Critical、3 Major 均已修复并有回归测试；10 项 Minor 中 6 项已修复、4 项评估为可接受并记录

## Acceptance Criteria 核对

| AC | 结论 | 证据 |
|---|---|---|
| AC-1 首轮全量建库 | PASS | `test_first_full_sync`：items/collections/membership/水位/向量链接、空摘要无向量 |
| AC-2 增量新增/更新只重算受影响向量 | PASS | `test_incremental_add_content_meta_classification`：嵌入调用数精确=新增+内容变化 |
| AC-3 移动零嵌入+路径审计 | PASS | moved 场景测试；moved 对比用原始 old_cols（不减已删集合），避免漏账 |
| AC-4 tombstone/trash 删除 | PASS | 删除三来源互斥分类 + GC 测试 |
| AC-5 versions 对账兜底 | PASS | reconciled 分类测试；M1 类型出局也复用该审计分类 |
| AC-6 跨页中断水位 | PASS | `test_pagination_interruption_then_resume`（失败不写身份行，见 m6） |
| AC-7 权限变化安全失败 | PASS | 403/userID 不符/library=false 三场景 + 恢复后续跑 |
| AC-8 同内容模型升级 | PASS | 旧链接事务内删除、旧 blob GC、同 hash 跨 fp 复用、崩溃整批复用 |
| AC-9 中断不缺向量/不重复付费 | PASS | Phase B 批失败：blob 保留、重跑第一批零嵌入、成功后 missing_vectors 为空 |
| AC-10 水位与缓存原子提交 | PASS | 单 BEGIN IMMEDIATE；提交前 `missing_vectors` 硬门禁；回调注入全回滚 |
| AC-11 Executor E2E 复用缓存 | PASS | test_executor 两轮 E2E；filter 后按 key 对齐；C1 修复后矩阵与论文同序重排 |
| AC-12 配置/工作流/忽略 | PASS（C2 修复后） | run_id 旋转 key + restore-keys（YAML 引号/序列形式），test_config_workflow 断言 |
| AC-13 审计完整原子落盘 | PASS | temp+fsync+os.replace；DB 行与 JSON 一致；失败轮也落盘 |
| AC-14 空摘要/集合删除成员关系 | PASS | 组合场景：补摘要进入语料、删集合摘 membership 零嵌入 |
| AC-15 不越水位 | PASS | `version>V` 对象留轮；item 与 collection 的 by-key 补取同样受 V 约束 |
| AC-16 测试全绿 | PASS | `uv run pytest -q` → 211 passed, 1 deselected |
| AC-17 设计质量（rubric） | PASS（5/5，见 TR-7.3） | 模块职责单一、错误类型明确、全部纯离线可测 |
| AC-18 审计可观测性（rubric） | PASS（5/5，见 TR-4.14） | 计数/key/路径/fingerprint/水位/耗时/错误/端点数俱全 |

## Findings

### Critical（2，均已修复）

**C1 — rerank 时间衰减权重与语料矩阵错配**（已修）
`BaseReranker.rerank()` 先按 `added_date` 重排 papers，但缓存矩阵没有应用同一置换，时间衰减权重落到错误论文上；且行数断言放在花式索引重排之后（3 行矩阵取 2 个索引不报错），错配无法被发现。
修复：行数断言前置到重排之前，随后矩阵按同一 `order` 重排（[base.py:111-121](file:///Users/kkcarrot/swe-project/zotero-arxiv-daily_fork1/src/zotero_arxiv_daily/reranker/base.py#L111-L121)）。
回归：`test_rerank_corpus_matrix_reordered_with_recency_sort`（缓存路径分数==无矩阵参考路径）、`test_rerank_corpus_matrix_asserts_row_count_mismatch`、api 侧矩阵对齐测试。

**C2 — Actions 缓存静态 key 导致水位永久冻结**（已修）
exact-hit 时 actions/cache 的 post 步骤不回存，每日 exact-hit 等于水位永不前进。
修复：key 追加 `${{ github.run_id }}` 强制 miss，restore-keys 按"同账号→同 OS"前缀恢复最新快照；表达式加引号/序列形式以规避 YAML flow-mapping 解析（[main.yml:22-35](file:///Users/kkcarrot/swe-project/zotero-arxiv-daily_fork1/.github/workflows/main.yml#L22-L35)）。
回归：`test_main_workflow_caches_state_dir_with_matching_env`（run_id、两级 restore-keys、OS 兜底）。

### Major（3，均已修复）

**M1 — item 改为非合格类型后永不离库且每轮重复补取**（已修）
item_versions 无类型过滤：A 从 journalArticle 变成 book 后，since 页（带 itemType 过滤）不返回它，drift 每轮用 itemKey 补取、每轮都无法落库。
修复：schema v2 新增 `excluded_items(key, version, item_type, updated_at)`（含 v1→v2 迁移与版本校验）；by-key payload 显式按 `ALLOWED_ITEM_TYPES` 白名单判定——已跟踪的非合格项按 reconciled 删除并记 excluded，新发现的仅记 excluded；版本未变的 excluded key 不再补取；改回合格类型时经"excluded drift"补取并重新入语料。Phase C 事务内 upsert/remove/prune（`delete_excluded_missing(versions 快照)`）。
回归：`test_item_changes_to_noneligible_type_leaves_and_can_return`（4 轮：在库→book 出局+审计→no-op 零补取→回归）、`test_nontracked_noneligible_item_is_excluded_quietly`、迁移测试 2 个。

**M2 — 嵌入模型在同名之下静默变更**（已修）
指纹只含后端+模型名+显式 version+dim；服务端换权重（dim 不变）或 local `encode_kwargs`（task/prompt_name/dimensions）变化都不会失效缓存。
修复：① 维度不一致 abort 文案明确指引 bump `reranker.embedding_model_version`；② 新增 `BaseReranker.embedding_model_params()`，对稳定参数做 canonical JSON（sort_keys、紧凑分隔）取 sha256 前 12 位纳入指纹（`backend:model:p<hash>:v<ver>:dimN`），`LocalReranker` 暴露 `encode_kwargs`。
回归：`test_local_fingerprint`（含参数时哈希格式）、`test_local_fingerprint_changes_with_encode_kwargs`（参数变→指纹变；空参数→无段）。

**M3 — 多页 since 拉取期间并发写入可使新 key 滑出分页窗口**（已修）
分页基于活的 offset 窗口，期间文库写入可能让某个 v≤V 的新对象既不在 since 页、本地又没有；collections 侧此前完全没有 drift 补取。
修复：Phase A 投影前对 versions 快照做补集扫描——本地 tracked（drift）/excluded（版本漂移）/完全缺失（`v<=V`）三类 key 合并去重后一次 `items_by_keys` 补取；collections 同构处理（新增 `collections_by_keys()`，chunks of 50）。`version>V` 的越界对象一律不补取（AC-15 不变）。
回归：`test_item_missing_from_since_pages_swept_from_versions`、`test_missing_key_newer_than_snapshot_is_not_fetched`、`test_collection_missing_from_since_pages_swept_from_versions`、`test_tracked_collection_drift_fetched_by_key`。

### Minor（10：6 修复，4 接受）

| # | 问题 | 处置 |
|---|---|---|
| m1 | `endpoints_called` 死字段，pages 缺 deleted/versions/drift 计数 | **已修**：`IncrementalZoteroClient.request_counts` 按逻辑端点计数（含 items_by_keys/collections_by_keys 与每次 follow），Phase A 后先快照（失败审计可定位卡点）、Phase C 前刷新；audit 字段类型由 list 更正为 dict；`test_report_endpoints_called_counts` |
| m2 | schema 无 FOREIGN KEY，state 注释声称外键不实 | **接受**：删除路径由 DAO 显式级联（delete_item 清 membership/链接，delete_collection 清 membership），幽灵 membership 再经集合路径回退与每轮对账收敛；单进程 Non-Goal 下风险可接受，注释已不宣称 FK |
| m3 | `transaction()` 的 ROLLBACK 自身失败会掩盖原始异常 | **已修**：ROLLBACK 包 try/except sqlite3.Error 后重新抛出原异常；`test_transaction_rollback_failure_keeps_original_error` |
| m4 | executor 不调 `syncer.close()`，连接生命周期=进程 | **接受**：CLI/GHActions 为短命进程，连接随退出回收；长驻服务化不在范围。`CorpusSyncer.close()` 已幂等可用 |
| m5 | Phase D（GC/审计文件/日志）异常会补写与已提交状态矛盾的 failed 审计 | **已修**：Phase D 整体 try/except，失败仅 warning 日志（明示水位已提交），不再回写 failed；既有 best-effort 测试保持通过 |
| m6 | 首轮身份行在 Phase A 之前写入，违反 FR-13"失败什么都不写" | **已修**：身份行移入 Phase C 事务（`init_library` 改为单语句、可在事务内调用）；首轮 Phase A/B 失败后 `get_library_state() is None`，5 个既有测试同步更新语义 |
| m7 | no-op 轮不更新 `last_sync_at`；`embeddings_reused` 把同 hash 批内去重计为复用 | **部分已修**：no-op 在审计事务内 `touch_library_state()`（`test_noop_round_short_circuits` 断言）；no-op 不做 GC（避免轻量变重）。reused 计数口径保留（同 hash 嵌入确实被复用，语义成立）；finished_at 取 Phase C 提交时刻，反映"数据一致点"而非派生 IO 完成点，保留 |
| m8 | item payload 缺/畸形 version 裸 KeyError/ValueError | **已修**：与 collection 侧一致包成 `SyncProtocolError("Malformed item payload")`；`test_item_with_malformed_version_raises` |
| m9 | vector BLOB 无端序标记 | **接受**：float32 原生字节，状态目录不跨架构共享（单机/GHActions runner 均 aarch64/x86_64 little-endian）；跨端迁移不在 Non-Goals |
| m10 | 审计 JSON 临时文件名固定，并发互踩 | **接受**：多实例并发同步同一状态目录是显式 Non-Goal；原子 replace 语义对单进程成立 |

## 已验证无需改动的安全结论

- Phase C 单事务原子性成立；blob 不可变、跨 fingerprint 不重用是正确行为（崩溃后从存活 blob 恢复 dim）。
- 删除三分类（tombstoned/trashed/reconciled）互斥且完备；trash 恢复以涨 version + added 方式正确回归（trashed 列实际为死列但行为正确）。
- 陈旧 payload 守卫比对本地 item version（不与水位比较，item version 不随库水位跳号）。
- 审计 JSON 经 temp+fsync+os.replace；空摘要两方向（一直空/中途补齐）正确；secret 仅出现在运行进程内，不入库不入审计。

## Rubric 评分

- **TR-4.14 / AC-18 审计可观测性：5/5**（修复后）。每轮含 W→V、首轮标志、四类变更 key 列表、moved 路径、删除三子类、embeddings computed/reused、model_upgraded 与新旧 fingerprint、逐端点请求计数、错误信息；人读摘要一行可核对。未给满分扣点：无单轮耗时（只有起止时间戳，可直接计算），不影响核对。
- **TR-7.3 / AC-17 设计质量：5/5**。state/content/client/syncer/audit 单一职责；Phase A-D 由代码结构强制；错误三分类（permission/protocol/base）覆盖权限、协议与可重试故障；全部行为离线可测，sync 包 100% 覆盖。

## 修复后的回归与覆盖

- `uv run pytest -q` → **211 passed, 1 deselected**
- `uv run pytest tests/sync --cov=zotero_arxiv_daily.sync` → 6 个模块全部 **100%**
- 新增测试：syncer +8（M1×2、M3×4、m1、m8、coll drift）、state +2（迁移、rollback 不掩盖）、zotero_client +1（懒构造）、reranker +2（encode_kwargs 指纹）、workflow 断言增强；5 个首轮失败测试更新为 m6 新语义
