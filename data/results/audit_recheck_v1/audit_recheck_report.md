# AUD-01 复核报告：H2_IPPO seed45/seed46 评估文件逐位重复事件

日期：2026-10-28
执行：数据取证复核（纯数据/取证，不改稿件）
冻结目录遵守：未写入任何冻结结果目录；所有产物在 `results/audit_recheck_v1/` 与 `tools/audit_recheck_*.py`。

## 一、判定

**verdict = refuted（复制事故假设被证伪）**

`results/formal_four_grid_v1/H2_IPPO_seed46/evaluation_metrics.json` **不是**被 seed45 覆盖/复制的文件：它是 seed46 checkpoint 的真实评估输出。两个权重逐位不同的独立训练策略，在 H2 的全部 5 条确定性评估轨迹上做出了**逐位相同的动作序列**，因此评估输出逐位相同。

**稿件数字无需修正**：pilot 主对比 +2.219%（5/5 正向，精确 p=0.0625）与 15 种子合并描述性效应 +1.881% 均维持原值。

## 二、取证事实

1. **重复范围**：两 `evaluation_metrics.json` sha256 相同（`c4d2aae7…b774b`）。对 60 个任务全量扫描，唯一的 U 向量重复对就是 H2_IPPO seed45/46，其余 58 个任务向量唯一。
2. **两任务确系独立训练**：
   - `checkpoint.pt` 哈希不同：seed45 `6d471bda…` vs seed46 `3bf4fe7a…`；全部 48 个 actor/critic 张量两两不同（max |diff| ≈ 0.645）。
   - `protocol_state.json` 记录独立时间线（45：06:47:40–08:04:41；46：07:57:10–09:25:55）与独立训练曲线（训练 reward 最大差 0.127）。
   - 评估配置完全一致（评估种子 1042–1046、H2 路网/路由、deterministic、temperature 1.0、IPPO 无先验）。
3. **确定性验证（重跑）**：从存量 checkpoint 重跑 5 集评估（`tools/audit_recheck_rerun_eval.py`）：
   - rerun45 与 stored45 **逐位一致**（max |diff| = 0.0 ≤ 1e-9）→ 评估管线确定性成立，stored45 为真值；
   - rerun46 与 stored46 **逐位一致**，且 == stored45。
4. **机制查明（动作轨迹记录）**：`tools/audit_recheck_trace_actions.py` 记录单集评估的完整确定性动作序列：
   - ckpt45 与 ckpt46 在 eval seed 1042 的 **1000 个决策步动作序列 sha256 完全相同**（`4652f826…160ea`）→ U 逐位相同是行为等价的必然结果；
   - 对照：ckpt42 同轨迹动作与 45 一致（对应 stored 42/44/45/46 第 1 集 U 相同 = 117.50329736211032）；ckpt43 动作序列不同（`246b6abf…49b9`，U=115.78012589928058）→ 该判别方法有效；
   - U 的"前缀重复"模式（42/44/45/46 第 1 集相同；44/45/46 前两集相同；仅 45/46 全 5 集相同）是"策略在部分轨迹段上行为等价、其他段分叉"的特征，与文件复制（要么全同要么无关）不符；
   - 权重差异真实存在：对随机高斯输入两网络 argmax 不一致率 18–32%；但在 H2 评估轨迹可达的真实状态流形上 argmax 完全一致（50 episode 短训 + 近饱和确定性环境使不同种子策略收敛到相同的确定性决策规则）。

## 三、数字核对（按"复制事故假设"的重算路径；因 rerun46 == stored46，全部不变）

| 量 | 重算值（原始任务数据） | 存储/稿件值 | 结论 |
|---|---|---|---|
| pilot seed log-ratios（42/43/44/45，未受影响） | 与存储值 diff = 0.0 | aggregation_summary | 一致 |
| seed46 H2 IPPO U（用 rerun46 替换） | 115.82679856115108（均值） | 115.82679856115108 | 无变化 |
| pilot 主对比 IPPO-P vs IPPO | **+2.2186%**，5/5 正向，p=0.0625 | 稿件 +2.219%，5/5，p=0.0625 | 无需修正 |
| 15 种子合并描述性效应 | **+1.8811%**（15/15） | 稿件 +1.881% | 无需修正 |
| 敏感性（同时剔除种子 45/46，n=13） | **+1.7733%**（13/13） | 审计参考 +1.773% | 独立复现一致 |

聚合公式独立验证：per-seed log-ratio = H1/H2/H3 等权 mean of log(U_IPPO/U_IPPO-P)；contrast pct = 100·(1−exp(−mean))；pooled = 15 个种子 log-ratio 等权 mean。复算 5 个种子 log-ratio 与存储值最大差 8.7e-19。

## 四、方法与产物

- 重跑评估：`tools/audit_recheck_rerun_eval.py`（从 checkpoint + protocol_state 环境快照直接调 `paper_protocol._evaluate`，无训练）
- 动作轨迹：`tools/audit_recheck_trace_actions.py`
- 重算：`tools/audit_recheck_recompute.py`
- 数据：`results/audit_recheck_v1/H2_IPPO_seed45_rerun/evaluation_metrics.json`、`.../H2_IPPO_seed46_rerun/evaluation_metrics.json`、`trace_H2_IPPO_seed{42,43,45,46}_ep1042.json`、`recompute_outputs.json`
- 机器可读摘要：`results/audit_recheck_v1/audit_recheck_summary.json`

## 五、说明与局限

- `paper_protocol.py` 当前版本哈希与 pilot 记录值不同（`a97e3ee1…` vs `f3fd62e7…`），其余核心文件（environment/、agents/、test_algorithms.py、路网/路由 XML）与 source_hashes 完全一致；重跑逐位复现实证确认当前评估路径对该任务等价。
- 本结论限于 H2_IPPO seed45/46 这一对任务的评估输出；不涉及训练结果本身的有效性（两训练独立、曲线不同，均正常）。
