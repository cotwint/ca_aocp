# run_baselines.py & run_real_data.py —— 逐行解读

两个脚本都做同一件事：**用不同的 conformal threshold 方法，对同一个数据流跑在线预测，比较覆盖率和区间宽度**。唯一的区别是数据来源和校准方式。

---

# 一、run_baselines.py —— 合成数据，冷启动

**环境**：合成数据（N(0,1) → N(4,1)，τ=500），无日历开销，所有 baseline 从零开始。

## 1. 导入和配置（L1~L67）

```python
# L18: __future__ annotations —— 允许 forward reference 类型标注
from __future__ import annotations

# L25: matplotlib.use("Agg") —— 非交互后端，不需要 GUI，CI/服务器可用
matplotlib.use("Agg")

# L30~31: 把项目根目录加入 sys.path[0]，
#   这样后续可以 `from ca_aocp import CAAOCP` 而不用装包
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # experiments/..  = 项目根
sys.path.insert(0, str(PROJECT_ROOT))

# L37~41: 五个 baseline，全部来自 online_conformal 包
from online_conformal.split_conformal import SplitConformal
from online_conformal.nex_conformal import NExConformal
from online_conformal.faci import FACI
from online_conformal.ogd import ScaleFreeOGD
from online_conformal.saocp import SAOCP

# L48: 合成数据路径
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_single_cp.csv"

# L50: COVERAGE=0.9 → 目标 90% 的 Y_t 落在区间内 → α=0.1
COVERAGE = 0.9

# L51: MAX_SCALE=10.0 —— 残差 |y - ŷ| 的最大预期值。
#   只有 SF-OGD 和 SAOCP 用这个，作为 scale-free 学习率的初始化。
#   合成数据: pre-change N(0,1), post-change N(4,1)
#   RollingMeanPredictor 的 ŷ 约在 0~4 之间，|y - ŷ| 最大可达 ~6-7
#   设 10.0 足够覆盖。
MAX_SCALE = 10.0

# L52: τ=500 —— 真实的 changepoint 位置。仅用于画图标记，算法不知道这个值。
TAU = 500

# L55~67: CA-AOCP 的超参，和 notebook 一致
CAAOCP_KWARGS = dict(
    alpha=1 - COVERAGE,    # = 0.1，名义误覆盖率
    eta=0.02,              # ACI 步长
    decay=0.01,            # 指数衰减 λ
    temperature=0.5,       # smooth surrogate 温度 τ
    hazard=0.005,          # BOCPD 先验 changepoint 概率 h
    bocpd_mu0=0.0,         # NIG 先验 μ
    bocpd_kappa0=1.0,      # NIG 先验 κ
    bocpd_alpha0=2.0,      # NIG 先验 α（稍强先验）
    bocpd_beta0=1.0,       # NIG 先验 β
    max_run_length=400,    # BOCPD 截断深度 K
    init_radius=1.0,       # 初始 conformal radius q_1
)
```

## 2. 共享预测器工厂（L74~L76）

```python
def make_predictor() -> RollingMeanPredictor:
    return RollingMeanPredictor(window=20)
```
所有 baseline 共用同样的点预测架构：过去 20 个观测值的滚动平均。保证了对比的公平时——区别**仅在于 conformal threshold**，不在于点预测。

## 3. 核心实验函数 `run_experiment()`（L81~L191）

### 3.1 初始化所有方法（L106~L124）

```python
# L110~L112: SplitCP、NExCP、FACI 不需要 max_scale
# model=None → 绕过 merlion 的 ForecasterBase，不训练模型
# train_data=None → 无校准数据，从零开始（cold start）
methods["SplitCP"] = SplitConformal(None, None, coverage=coverage)
methods["NExCP"]   = NExConformal(None, None, coverage=coverage)
methods["FACI"]    = FACI(None, None, coverage=coverage)

# L115~L116: SF-OGD 和 SAOCP 需要 max_scale
# 它们用 scale-free OGD，学习率 = max_scale / sqrt(3 * Σ grad²)
methods["SF-OGD"] = ScaleFreeOGD(None, None, coverage=coverage, max_scale=max_scale)
methods["SAOCP"]  = SAOCP(None, None, coverage=coverage, max_scale=max_scale)

# L119~L123: CA-AOCP 有自己的预测器（RollingMeanPredictor）
caaocp = CAAOCP(
    predictor=make_predictor(),
    **CAAOCP_KWARGS,
)
```

### 3.2 共享预测器（L126）

```python
shared_pred = make_predictor()
```
baselines 用这个预测器算 ŷ_t。注意：**CA-AOCP 不用这个**，它内部有自己的 `RollingMeanPredictor`。但因为所有 predictor 都是 `window=20` 的滚动平均，所以实际上 ŷ_t 是同步的——它们看到的 y 序列完全一样。

### 3.3 结果缓存（L128~L140）

```python
buffers = {
    name: {"covered": [], "width": [], "lower": [], "upper": [], "radius": [], "time_s": 0.0}
    for name in names
}
```
每个方法在每个时间步记录：
- `covered`: 1/0 → 后面求平均 = 覆盖率
- `width`: `2 × radius` → 区间宽度
- `lower/upper`: 区间端点
- `radius`: conformal threshold `s_hat`
- `time_s`: 累计运行时间

### 3.4 在线循环（L143~L190）—— **最核心的部分**

```python
T = len(ys)                    # 数据总长度 1000
for t_idx in range(T):         # 逐时间步处理
    y_t = float(ys[t_idx])     # 当前真实值
    x_t = None                 # 无协变量（单变量时间序列）

    # L149: shared_pred 给出 ŷ_t = 过去 20 个 y 的均值
    yhat_shared = shared_pred.predict(x_t)

    for name in names:
        method = methods[name]

        if name == "CA-AOCP":
            # ══════════════════════════════════════════════════════
            # CA-AOCP 分支：内部自行计算 score + BOCPD + 分位数
            # ══════════════════════════════════════════════════════

            # predict(x_t) 返回 (lower_bound, upper_bound)
            #   内部流程: predictor.predict(x_t) → ŷ
            #             weighted_quantile(scores, weights, alpha_t) → q_t
            #             返回 [ŷ - q_t, ŷ + q_t]
            lo, hi = method.predict(x_t)

            # update(x_t, y_t) 做以下事情:
            #   1. 用 ŷ 和 y_t 算 score s_t = |y_t - ŷ|
            #   2. 用 smooth_surrogate 算 ℓ_t = σ((s_t - q_t)/τ)
            #   3. 用 adaptive_update 更新 α_{t+1}
            #   4. 用 BOCPD 更新 run-length posterior → π_{t+1,i}
            #   5. 用 π 和 decay 算权重 w̃_{t+1,i}
            #   6. weighted_quantile(scores, w̃, α_{t+1}) → q_{t+1}
            #   7. 更新 predictor
            method.update(x_t, y_t)

            radius = (hi - lo) / 2.0          # 对称区间半径
            covered = int(lo <= y_t <= hi)    # 1 表示覆盖，0 表示未覆盖

        else:
            # ══════════════════════════════════════════════════════
            # Baseline 分支：共享 ŷ，返回预测区间
            # ══════════════════════════════════════════════════════

            # predict(horizon=1) 返回 (delta_lb, delta_ub) = (-s_hat, +s_hat)
            #   具体返回值因 baseline 而异:
            #   SplitCP / NExCP / FACI:
            #     s_hat = 历史残差数组的 (1-α) 加权分位数
            #   SF-OGD / SAOCP:
            #     s_hat = OGD 学到的阈值 delta[horizon]
            delta_lb, delta_ub = method.predict(horizon=1)

            # 构造预测区间 [ŷ_t + delta_lb, ŷ_t + delta_ub]
            lo = yhat_shared + delta_lb
            hi = yhat_shared + delta_ub
            radius = (hi - lo) / 2.0
            covered = int(lo <= y_t <= hi)

            # update 把当前 (y_t, ŷ_t) 喂给 baseline
            #   内部: residuals = |ground_truth - forecast| = |y_t - ŷ_t|
            #   SplitCP: residuals.append(|y_t - ŷ_t|)
            #   NExCP:   residuals.append(|y_t - ŷ_t|) + weighted quantile
            #   FACI:    更新多个 alpha_k + Hedge 权重
            #   SF-OGD:  grad = pinball_loss_grad(s_t, s_hat, coverage)
            #             s_hat = s_hat - lr * grad
            #   SAOCP:   多个 SF-OGD 专家各自更新 + Coin Betting 加权
            method.update(
                ground_truth=pd.Series([y_t]),
                forecast=pd.Series([yhat_shared]),
                horizon=1,
            )

        # 四个缓冲区记录当前步的结果
        buffers[name]["covered"].append(covered)
        buffers[name]["width"].append(2.0 * radius)
        buffers[name]["lower"].append(lo)
        buffers[name]["upper"].append(hi)
        buffers[name]["radius"].append(radius)
        buffers[name]["time_s"] += elapsed

    # L189: 共享预测器更新 —— 这一步之后 ŷ 会变化
    #   shared_pred 内部 buffer 追加 y_t，重新算均值
    shared_pred.update(x_t, y_t)
```

### 3.5 关键设计决策

**为什么 baseline 的 update 传 `ground_truth=Series([y_t]), forecast=Series([yhat_shared])`？**

因为 baseline 内部做的是 `abs(ground_truth - forecast)` → `|y_t - ŷ_t|` = conformal score。这和 CA-AOCP 内部的 score 计算完全一致。两个脚本（run_baselines.py 和 run_real_data.py）现在使用**相同的 update 模式**，保证可比性。

最终 residuals 数组里存的都是 `|y_t - ŷ_t|`。调用 `predict()` 时，baseline 内部用这些 residuals 的加权分位数（或 OGD 学到的阈值）作为 conformal radius，返回 `(-s_hat, +s_hat)`。

## 4. 指标计算（L198~L213）

```python
def compute_metrics(buffers, alpha=0.1):
    for name, b in buffers.items():
        cov_arr = np.array(b["covered"], dtype=float)   # [1,0,1,1,0,...]
        width_arr = np.array(b["width"])                # [2*q_1, 2*q_2, ...]
        # Coverage = covered 的均值 → 0.885 表示 88.5% 覆盖
        # Cov. Gap  = |coverage - target| → 和 0.9 的差距
        # Mean Width = 平均区间宽度 → 越小越精确
        # Time = 总耗时
```

## 5. 滚动覆盖率（L216~L219）

```python
def rolling_coverage(covered, window=50):
    c = np.array(covered, dtype=float)
    return np.convolve(c, np.ones(window) / window, mode="valid")
```
用卷积做滑动窗口平均：`[0,1,0,1,1,...] * [1/50, 1/50, ..., 1/50]` → 每个输出点是过去 50 步的覆盖率。`mode="valid"` 丢弃开头 49 个不完整的窗口。

## 6. 绘图 `plot_all()`（L237~L367）

六张图：
1. **01_rolling_coverage.png**：50 步滚动覆盖率，6 条线
2. **02_post_cp_recovery.png**：τ=500 附近的放大视图（30 步窗口），看谁恢复快
3. **03_width_over_time.png**：区间宽度随时间变化（平滑 w=20）
4. **04_ca_aocp_intervals.png**：CA-AOCP 的区间 ribbon + 数据散点
5. **05_radius_trajectory.png**：各方法的 conformal radius `s_hat` 轨迹
6. **06_radius_distribution.png**：CA-AOCP 在 changepoint 前后的 radius 分布直方图

## 7. main()（L374~L445）

```python
# 1. 读 CSV → ys 数组
df = pd.read_csv(DATA_PATH)
ys = df["value"].values

# 2. 打印数据概况（均值、标准差、changepoint 位置）
print(f"Pre-change  μ ≈ {ys[:TAU].mean():.3f} ...")

# 3. 运行实验
t0 = time.perf_counter()
buffers = run_experiment(ys, coverage=COVERAGE, max_scale=MAX_SCALE)
total_elapsed = time.perf_counter() - t0

# 4. 计算指标
metrics_df = compute_metrics(buffers, alpha=alpha)

# 5. 分 changepoint 前后的覆盖率
for name in buffers:
    pre_cov[name]  = covered[:TAU].mean()   # t < 500
    post_cov[name] = covered[TAU:].mean()   # t ≥ 500

# 6. 画图 + 存 CSV
plot_all(buffers, ys, OUTPUT_DIR)
metrics_df.to_csv(OUTPUT_DIR / "metrics.csv")
```

---

# 二、run_real_data.py —— M4 Weekly，清洁流水线

**环境**：M4 竞赛 Weekly 数据（真实时间序列），RollingMean 同时用于校准和在线阶段，所有方法共享同一校准残差。

## 和 run_baselines.py 的核心差异

| 维度 | run_baselines.py | run_real_data.py |
|------|-----------------|------------------|
| 数据 | 合成，N(0,1)→N(4,1) | M4 Weekly，10 条真实序列 |
| 校准 | 无（cold start，从 0 开始） | RollingMean walk-forward（20% train） |
| 归一化 | 不需要 | train-only min/max 归一化到约 [0,1] |
| 循环方式 | 全局 run_experiment() | 逐条序列 evaluate_one_series() |
| SF-OGD max_scale | 10.0（原始尺度） | 3.0（归一化后尺度） |
| CA-AOCP init_radius | 1.0（原始尺度） | 0.5 * std(train_fit)（自适应） |

## 1. 数据加载：绕过 M4 wrapper（L82~L105）

```python
def load_raw_m4_series(idx):
    from ts_datasets.forecast import M4 as RawM4
    ds = RawM4("Weekly")
    ts, _ = ds[idx]                            # ts 是 pd.DataFrame
    vals = ts.iloc[:, 0].values.astype(float)  # 原始值，未归一化
    T = len(vals)
    if T > 400:       n_test = 120
    elif T > 200:     n_test = 60
    else:             n_test = 26
    return vals[:-n_test], vals[-n_test:]
```

**为什么不用 `online_conformal.dataset.M4`？** 那个 wrapper 的 `__getitem__` 内部先做 `(ts - ts.min())/(ts.max() - ts.min())` 用**整个序列**（含 test）的 min/max 归一化，这会造成数据泄漏——在线预测时无法提前知道 test 的取值范围。我们只用 `ts_datasets.forecast.M4` 拿原始数据，自己用 train-only min/max 归一化。

## 2. 归一化：train-only（L108~L117）

```python
def normalize(train_raw, test_raw):
    vmin, vmax = train_raw.min(), train_raw.max()
    scale = vmax - vmin
    if scale < 1e-12: scale = 1.0
    train_norm = (train_raw - vmin) / scale
    test_norm  = (test_raw  - vmin) / scale
    return train_norm, test_norm, vmin, scale
```

test 用 train 的 min/max 归一化，test 值可能超出 [0,1]。在线场景只能这样。

## 3. 校准 walk-forward（L120~L155）

```python
def walk_forward_calibrate(calib_y, *methods, shared_pred, caaocp):
    for y_t in calib_y:
        yhat = shared_pred.predict(None)
        s_t = abs(y_t - yhat)
        # 把真实残差喂给每个 baseline
        for _, bl in methods:
            bl.update(ground_truth=pd.Series([y_t]),
                      forecast=pd.Series([yhat]), horizon=1)
        shared_pred.update(None, y_t)
        # CA-AOCP 也做 warm-start
        if caaocp is not None:
            caaocp.predict(None)
            caaocp.update(None, y_t)
```

**关键设计**：校准和在线阶段使用**完全相同的 RollingMean 预测器架构**。这确保了 calibration scores 和 online scores 来自同一个分数源——不存在不一致的问题。

## 4. 核心函数 `evaluate_one_series()`（L162~L268）

### 4.1 数据准备（L166~L188）

```python
train_raw, test_raw = load_raw_m4_series(series_idx)
train_norm, test_norm, vmin, scale = normalize(train_raw, test_raw)

# 从 train 尾部切出 20% 作为校准集
n_calib = max(1, int(len(train_norm) * CALIB_FRAC))
train_fit = train_norm[:-n_calib]
calib_y   = train_norm[-n_calib:]
```

### 4.2 Baseline 初始化（L181~L188）

```python
baselines = [
    ("SplitCP", SplitConformal(None, None, coverage=COVERAGE)),
    ("NExCP",   NExConformal(None, None, coverage=COVERAGE)),
    ("FACI",    FACI(None, None, coverage=COVERAGE)),
    ("SF-OGD",  ScaleFreeOGD(None, None, coverage=COVERAGE, max_scale=MAX_SCALE)),
    ("SAOCP",   SAOCP(None, None, coverage=COVERAGE, max_scale=MAX_SCALE)),
]
```

和 run_baselines.py 一样：`model=None, train_data=None` cold-start，然后通过 walk_forward_calibrate 收集校准残差。

### 4.3 CA-AOCP 初始化（L191~L195）

```python
init_r = float(np.std(train_fit) * 0.5)
caaocp = CAAOCP(predictor=RollingMeanPredictor(window=PRED_WINDOW), **kwargs)
```

`init_radius` 从 train_fit 的标准差自适应估计，而不是硬编码。

### 4.4 预测器预热（L198~L208）

```python
# 共享预测器：用 train_fit 尾部预热
shared_pred = RollingMeanPredictor(window=PRED_WINDOW)
for v in train_fit[-PRED_WINDOW:]:
    shared_pred.update(None, float(v))

# CA-AOCP 内部预测器也预热
for v in train_fit[-PRED_WINDOW:]:
    caaocp.predict(None)
    caaocp.update(None, float(v))
```

**这一步确保**：第一个校准数据点就有一个合理的 ŷ（基于 train 尾部），而不是冷启动的 0。

### 4.5 校准 + 在线循环（L210~L253）

校准 pass（L210）：
```python
walk_forward_calibrate(calib_y, *baselines,
                       shared_pred=shared_pred, caaocp=caaocp)
```

在线循环（L216~L253）：
```python
for t in range(len(test_y)):
    y_t = float(test_y[t])
    yhat = shared_pred.predict(None)

    for name, bl in baselines:
        delta_lb, delta_ub = bl.predict(horizon=1)
        lo = yhat + delta_lb
        hi = yhat + delta_ub
        covered = int(lo <= y_t <= hi)
        bl.update(ground_truth=pd.Series([y_t]),
                  forecast=pd.Series([yhat]), horizon=1)
        # ... 记录到 buffers

    # CA-AOCP
    lo, hi = caaocp.predict(None)
    caaocp.update(None, y_t)
    # ... 记录到 buffers

    shared_pred.update(None, y_t)
```

与 run_baselines.py 使用**完全相同**的 predict/update 模式：
- `delta_lb, delta_ub = bl.predict(horizon=1)` → `lo = yhat + delta_lb, hi = yhat + delta_ub`
- `bl.update(ground_truth=Series([y_t]), forecast=Series([yhat]), horizon=1)`

### 4.6 指标（L255~L268）

```python
metrics[name] = {
    "coverage": float(cov.mean()),
    "cov_gap": float(abs(cov.mean() - COVERAGE)),
    "mean_width_norm": float(w_norm.mean()),   # 归一化宽度
    "mean_width_raw": float(w_norm.mean() * scale),  # 原始尺度宽度
}
```

同时汇报归一化和原始尺度的宽度，方便和实际物理量纲对齐。

## 5. 聚合和绘图（L275~L370）

和 run_baselines.py 的模式类似，但按序列做聚合：
- 每个序列算一套 metrics → 10 条序列的 coverage/width 做 bar chart + error bar
- 散点图：每点 = 一条序列，Coverage vs Width

## 6. main()（L377~L428）

```python
for i in range(N):
    metrics = evaluate_one_series(series_idx=i)
    all_metrics.append(metrics)

agg_df = aggregate(all_metrics)
plot_aggregate_results(all_metrics, OUTPUT_DIR)
agg_df.to_csv(OUTPUT_DIR / "metrics.csv")
```

---
# 三、两个脚本共有的设计原则

1. **仅变 conformal threshold，不变点预测**。所有 baseline（和 CA-AOCP）使用同样架构的 `RollingMeanPredictor(window=20)`。任何覆盖率/宽度的差异都来自阈值机制的不同。

2. **在线逐步更新**。每个时间步：`predict（给区间）→ 观察真实值 → update（更新状态）`。不批量、不回看，完全是 online 设定。

3. **对称区间**。回归任务下，conformal score = 绝对残差，对称区间 `[ŷ - s_hat, ŷ + s_hat]`。

4. **统一的 predict/update 语义**。两个脚本使用完全相同的模式：`delta_lb, delta_ub = method.predict(horizon=H)` 获取区间偏移量，`method.update(ground_truth=Series([y_t]), forecast=Series([yhat]), horizon=H)` 更新状态。内部 residuals 都是 `|y_t - ŷ_t|`。

5. **统一评估指标**。覆盖率（是否达标）、Coverage Gap（偏差多少）、Mean Width（精确度）、Time（效率）。
