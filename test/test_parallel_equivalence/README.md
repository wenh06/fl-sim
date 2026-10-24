# F1 等价 / golden 回归测试骨架

终审（`tmp/parallel-execution-review.md` 第五部分 **F1**）指出：设计稿 §8.1 的 14 项「CI 必跑」测试里，
**只有 5 项在仓库中落地**，最关键的 **golden 回归**（G2 算法语义零改动 / G3 sequential 逐位一致）
只有 PR 描述里的手工验证，**仓库里没有可复现的护栏**。

本目录是那四项缺失测试的可运行骨架：

| 文件 | 对应评审项 | 守什么 |
|---|---|---|
| `test_golden_regression.py` | **F1-1 golden 回归** | 每个场景的完整摘要逐位不变；基线覆盖全部 20 个注册算法；APFL `tau>1` 的既有缺陷被钉住 |
| `test_migration_equivalence.py` | **F1-2 协议迁移等价** + **F1-3 决策上移等价** | 上行载荷的键集与迁移前（`f6c3e33`）**完全一致**；上行决策序列可复现；逐算法的计数语义（FedPD 跳过不计 / ProxSkip 跳过仍计）保持 |
| `test_skipped_clients_still_train.py` | **F1-4 = 评审 M3** | 跳过上传的客户端**仍然训练、仍然写指标日志**（early-return 回归的直接守卫） |
| `golden_lib.py` | — | 场景定义 + 摘要（digest）计算；**同时兼容迁移前后的代码** |
| `make_golden.py` | — | 基线生成 / 校验 / **跨版本对照** |
| `golden/baseline.json` | — | 已提交基线（22 个场景） |

## 跑起来

```bash
# 全部测试（仓库根目录下）
pytest test/test_parallel_equivalence -q

# 校验当前代码仍复现基线
python test/test_parallel_equivalence/make_golden.py --out test/test_parallel_equivalence/golden/baseline.json --check
```

## 基线里钉住了什么

摘要包含五类观测，缺一不可：

- `messages` —— 服务器**按采样顺序**消费的上行消息（键集 + 载荷摘要）⇒ 聚合输入
- `uploads_per_round` —— 每轮实际上传的客户端集合 ⇒ 参与集合
- `decisions` —— 每个客户端每轮的上行决策（重构后才有，故跨版本对照时跳过）
- `metrics_log` / `updates` —— 指标日志行、以及每个客户端本地步前后的模型摘要 ⇒ **"跳过上行也照常训练、照常记录"**
- `final_weights` / `num_communications` —— 服务器最终权重与通信计数

## 跨版本对照：证明"迁移没有改变行为"

harness 只用**迁移前后都存在**的 API，因此可以直接指向另一个 checkout：

```bash
git worktree add /tmp/flsim-pre f6c3e33        # 重构前的 master
python test/test_parallel_equivalence/make_golden.py --repo /tmp/flsim-pre --out /tmp/pre_refactor.json
python test/test_parallel_equivalence/make_golden.py --compare /tmp/pre_refactor.json
```

**本目录实测结果：`IDENTICAL to /tmp/pre_refactor.json for 22 scenario(s)`** —— 22 个场景
（含全部 20 个注册算法）在重构前后**逐位一致**。

> 这是终审 F1 的正面答案：G2/G3 的等价性**可以被自动化验证**，不再只存在于 PR 描述里。

## 已知缺陷被钉住而不是被藏起来

`apfl-tau2` 场景在基线里记录的是一次**失败**：

```json
"outcome": {"status": "raised", "exc_type": "KeyError", "exc_msg": "'parameters'"}
```

APFL `tau>1` 时服务器与客户端的同步计数器错位（服务器第 0 轮就要聚合，而客户端晚一轮才带
`parameters`），这是**并行层之前就存在**的缺陷（在 `f6c3e33` 上同样复现）。
基线把它钉住 ⇒ 将来有人修好它时，golden 测试会失败，**逼出一次显式的、被评审的基线更新**，
而不是让行为悄悄变化。

## 落到仓库时

建议把本目录整体迁到 `test/` 下（例如 `test/test_parallel_equivalence/`），并：

1. 把 `golden/baseline.json` 提交进仓库；
2. 在 `.github/workflows/run-pytest.yml` 里加一条 job（这四个测试约 12 秒）；
3. 基线更新走**单独的 PR**，在 PR 描述里贴出 `make_golden.py --check` 的差异输出，
   让"行为变了"这件事必须被看见。

## Cross-machine behaviour (CI)

Floating-point digests are bitwise-stable only on the CPU/build that
generated the baseline (SIMD reduction order differs across processor
generations). The baseline therefore records a **machine fingerprint**
(torch build + CPU capability + CPU model):

- on the **baseline machine** (local development): the full bitwise
  comparison runs — any float change fails;
- on a **foreign machine** (e.g. a CI runner): a **structural** comparison
  runs instead — outcome, participation decisions, upload structure,
  communication counters and the metric-log structure are still guarded;
  float digests are not compared.

Baseline updates are deliberate: regenerate with `make_golden.py --out ...`
and let the diff be reviewed in the PR.
