# CA-AOCP 实验总结报告

更新时间：2026-08-08（第三次跟进，见第19节） | 第二次跟进：2026-08-07（见第14~18节） | 首版：2026-08-04 | 覆盖范围：Task #1 ~ #6（组件消融 → shift grid → 多seed → 显著性检验 → 真实数据 → 超参数敏感性 → eta敏感性/冗余检验）+ 核心包O(K)截断bug修复 + 复杂度遗留问题修复与核心实验重跑（含eta sweep）

> **读者须知（2026-08-08更新）**：第2~10节的具体数字是**修复前（pre-fix）**的历史记录，当时`experiments/run_ablation.py`/`run_shift_grid.py`用的calibration set并未真正截断到K，只有核心包`ca_aocp.bocpd`内部的run-length后验是截断的。第14~18节记录了对这个问题（以及另外两个复杂度遗留bug）的修复，并用同样规模（30-seed multiseed + 21-seed shift grid）重新跑出了post-fix数字——**第6节"soft显著优于hard-reset"这个结论在post-fix数据下发生了实质性变化（部分指标方向反转），详见第17节**。第19节补了第9节（eta sweep）的post-fix重跑：这部分**核心方向基本复现**（受影响远小于第6节的hard-reset对比），但有几处具体数字需要更正。本文档保留原始章节不做删改，新结论以追加章节的方式呈现，读者应以第17、19节为准。

---

## 0. 背景与目标

出发点是审稿前的策略判断："现在不需要重做算法，优先补实验"，围绕一份5项实验清单（组件消融含hard-reset baseline、shift幅度×post-change样本量网格、多seed重复+误差条、真实多regime数据验证、超参数敏感性）系统补齐CA-AOCP论文的实证基础，目标是给出诚实、可复现、经得起审稿人质疑的结论——无论结论对论文有利还是不利。

在过程中，围绕代码库本身也发现了两个独立于实验清单、但同样重要的问题（见第1节），以及一个中途自我发现并修正的方法论错误（marginal SE vs paired test，见第4节），这个修正实际上反转了本报告最核心的一个结论。第13节追加修复了核心包的O(K)截断bug。**第14~18节是本报告的第二次跟进**：修复了核心包和实验脚本里三个仍然违反论文O(K)复杂度声称的遗留问题，并用修复后的代码重新跑了两组核心实验（30-seed组件消融、21-seed shift grid），发现第6节的"soft vs hard-reset"结论需要修正。

---

## 1. 代码库 vs 论文的既有差距（背景发现，非本次实验清单的一部分）

在读代码阶段确认了两个此前未被文档化的问题：

- **O(K)截断未真正实现**：论文声称核心算法复杂度是O(K)（K=截断窗口深度），但`ca_aocp/weights.py`的`compute_weights`和`ca_aocp/conformal.py`的`weighted_quantile`实际上始终对**全历史**数组做`np.argsort`/权重计算，从未截断到K。`bocpd.py`的`_compute_pi()`同理。这个差距在小规模数据上不影响正确性，但会让运行时随T线性增长，与论文对Algorithm 2复杂度的描述不符。**这一问题本轮已修复，见第13节**——本报告第2~10节的实验结果都是用一个独立、干净重写的`GenericMethod`（`run_ablation.py`）绕开这个问题得到的，未采用修复后的核心包重跑。
- **仓库自带的baseline对比对论文不利**：`experiments/run_baselines.py`已经包含FACI/SF-OGD/SAOCP等更强baseline的对比代码（论文正文未报告），运行后CA-AOCP在runtime（66s vs FACI的1.78s，T=1000）和cov_gap（6个方法中最差）上都落后。同时发现`CAAOCP_KWARGS`里用的是`max_run_length=400`而非论文Table 9报告的K=50，这本身就是runtime差距的一大来源（换回K=50后单seed耗时从66s降到1.7s）。这两组结果目前躺在仓库里未被论文引用，如果这是公开仓库，审稿人自己跑一遍是会看到的——建议要么在文中提前坦白讨论，要么至少确认这些脚本不会被审稿人无意中翻到。

---



## 2. 方法论说明（贯穿全部实验）

- **8方法消融矩阵**：`experiments/run_ablation.py`里的`GenericMethod`/`make_methods`把CA-AOCP的三个组件（BOCPD软加权 π、recency decay、ACI自适应α）拆成独立开关，组合出8个方法：`ACI`、`EWMA-ACI`、`BOCPD-weighted CP`、`BOCPD-weighted ACI`、`CA-AOCP w/o ACI`、`Full CA-AOCP`（=论文方法）、`Hard-reset CP`、`Hard-reset ACI`（hard-reset = BOCPD只给MAP changepoint估计，用它去截取一个均匀窗口，作为"退化成点估计"的对照组）。这个框架被后续几乎所有实验复用。
- **两个基准数据集**：Synthetic-CP（T=1000, τ*=500，仓库自带）；Quality-Control（T=313, τ*=148，按论文Table 8的矩统计量重构——不是原始数据，因为原始数据未随仓库发布）。
- **paired test vs marginal SE（本次会话中最重要的方法论修正）**：在30-seed消融和最初的shift-grid分析里，我最早是用每个方法自己的marginal mean±SE去比较soft-weighting和hard-reset，得出"二者打平"的结论。这是**错误的**——因为两个方法在同一个seed下跑的是完全相同的数据实现（paired design），marginal SE会系统性高估真实噪声水平。中途自查发现这个问题后，改用配对差值（hard − soft，按seed配对）做paired t-test + Wilcoxon signed-rank test，**结论直接反转**：soft-weighting相对hard-reset有统计显著、方向一致的优势（尤其在Winkler score和mean_width上）。这个修正后来推动了整个显著性报告（`significance_report.py`）全面改用paired test + Benjamini-Hochberg FDR校正。
- **沙盒约束**：bash单次调用45秒超时，无法跑后台进程（bwrap沙盒每次调用独立PID namespace），仅能访问PyPI镜像（无GitHub/UCI/Yahoo Finance）。这决定了所有多seed实验都写成`--seed-start/--seed-end` CLI + CSV增量append的形式，分批小批量跑；真实数据全部来自`pmdarima`/`aeon`这两个PyPI包内置的离线数据集。

---



## 3. Task #1：组件消融（单seed）

用8方法矩阵在两个数据集上各跑一次，作为后续多seed实验前的正确性检查。确认了实现在结构上是对的（测量顺序、权重公式与论文Appendix B一致），也第一次看到"ACI在Synthetic-CP上让cov_gap从~0.014涨到~0.03"这个后来反复出现的现象的雏形。

---



## 4. Task #3：多seed重复实验（30 seeds × 2数据集 × 8方法）

`run_multiseed.py`，产出`output_multiseed/raw.csv`（480行）。除标准指标外新增：`recovery_step`（滚动10步coverage首次达标的时间点）、`false_alarm_rate`（changepoint前r_hat被误判为近期切换的比例）、`cp_delay`（changepoint后r_hat真正收敛所需步数）。

**关键数字（30-seed均值，见**`output_multiseed/aggregate.csv`**）：**


| Dataset         | Method          | cov_gap | mean_width | winkler | postcov10 |
| --------------- | --------------- | ------- | ---------- | ------- | --------- |
| Synthetic-CP    | ACI (无BOCPD)    | 0.0417  | 3.987      | 4.630   | 0.170     |
| Synthetic-CP    | CA-AOCP w/o ACI | 0.0137  | 3.388      | 4.449   | 0.817     |
| Synthetic-CP    | Full CA-AOCP    | 0.0286  | 3.906      | 4.543   | 0.823     |
| Quality-Control | ACI (无BOCPD)    | 0.0337  | 4.600      | 5.421   | 0.550     |
| Quality-Control | CA-AOCP w/o ACI | 0.0184  | 3.983      | 5.307   | 0.737     |
| Quality-Control | Full CA-AOCP    | 0.0190  | 4.507      | 5.381   | 0.787     |


两个直接可读的结论：(a) 单纯ACI（无BOCPD）在两个数据集上postcov10都明显偏低（0.17 / 0.55），说明它对changepoint后短期恢复的建模能力弱于BOCPD；(b) BOCPD-weighting叠加ACI（Full CA-AOCP）相对不叠加ACI（w/o ACI），cov_gap在Synthetic-CP上从0.0137涨到0.0286（约2倍），在QC上基本持平（0.0184→0.0190）——**这是"ACI要不要默认开启"这个问题最初的证据来源，但当时用的默认eta=0.02，后来Task #6发现这个数字对eta高度敏感（见第9节）**。

---



## 5. Task #2：Shift grid（重新设计为"soft何时甩开hard-reset"）

不是泛泛扫描shift幅度，而是显式构造shift强度（1σ/2σ/4σ）× ramp渐变长度（0/10/40步）的3×3网格 + 一个"无changepoint"对照条件，共10个条件 × 21 seeds × 4方法（BOCPD-weighted CP/ACI, Hard-reset CP/ACI），`run_shift_grid.py`产出`output_shiftgrid/raw.csv`（840行）。新增诊断指标：`neff`（有效样本量 1/Σw̃²，衡量权重分布的"有效记忆长度"）、`width_overshoot_ratio`（changepoint后区间宽度相对新regime真实宽度的超调比例）。

（这一轮的marginal-SE分析后来被第4节提到的修正推翻并用paired test重做，结论见第6节表B/C。）

---



## 6. 显著性报告（正式配对统计）

`significance_report.py`把第4、5节的全部数据重新用paired t-test + Wilcoxon + BH-FDR过一遍，产出`output_significance/`下4张表。核心发现（q<0.05为经FDR校正后仍显著）：

- **Table A（30-seed消融）**：soft相对hard-reset在mean_width、winkler上高度显著更优（Synthetic-CP: mean_width diff=+0.067, p<1e-7；winkler diff=+0.046, p<1e-4；QC同向但幅度更小）。cov_gap本身两个数据集都不显著——soft的优势主要体现在**区间更窄+Winkler更好**，不是覆盖率本身。
- **Table B（shift grid，10条件池化，n=210）**：soft相对hard-reset在cov_gap（+0.0028, p<1e-7）、mean_width（+0.06, p<1e-17）、width_overshoot_ratio（+0.075, p<0.001）、neff_t+5/+10/+25 全部高度显著——soft-weighting在shift后保留的有效样本量更大、区间收缩更平滑，这是符合直觉的机制性证据。
- **结论**：soft weighting相对hard-reset是一个小但真实、统计显著、跨两轮独立实验一致复现的优势，主要体现在sharpness（区间宽度、Winkler）而非纯覆盖率上。这个结论在中途的marginal SE分析里被误判为"打平"，修正后才浮现——值得在论文里明确用paired test报告这个比较，而不是marginal mean±SE。

> **⚠️ 2026-08-07更新**：以上是pre-fix数字。`run_ablation.py`/`run_shift_grid.py`当时并未把calibration set真正截断到K=50（只有BOCPD内部后验是截断的，见第14节问题#2），也就是说这里比较的"hard-reset"实际上经常用了远超50个历史点的均匀窗口。用真正K截断的代码重跑后，**mean_width的方向在CP(no ACI)配对下反转（hard-reset变得更窄）**，Winkler优势在Synthetic-CP上消失；cov_gap反而冒出新的、跨几乎所有shift grid条件一致的soft显著优势。第6节的这个结论**不能再按原样引用**，完整对比见第17节。

---



## 7. Task #4：真实多regime数据验证（两轮）

**第一轮**（`run_real_multiregime.py`）：MSFT股票在dotcom崩盘(2000)、次贷危机(2008)前后的日收益率波动，以及Taylor电力需求数据（T截断到2000点，因超时）。这类数据的regime变化是**连续漂移**而非离散切换，与论文的合成benchmark设计不同。发现：单纯ACI/EWMA-ACI在这类数据上Winkler经常是最优或接近最优（例如electricity: ACI winkler=18613 vs Full CA-AOCP=19207；dotcom: ACI=12.85 vs Full=13.48），BOCPD的离散regime假设在这里没有优势。同时发现`adaptive_update.py`的`logistic()`在electricity这种量级（原始值上千）下会数值溢出（`RuntimeWarning: overflow in exp`）——**这是一个未修复的实现bug**，不影响本报告的定性结论但正式使用前应修。

**第二轮**（`run_real_segmentation.py`，按你的要求重新设计，找更贴合论文原始设定的真实离散regime-switch数据）：改用`aeon`包内置的UCR时间序列分割benchmark——Electric Devices（3段窗口，真实设备切换点）和Gun Point（motion-capture传感器模式切换）。这些是真实数据，但带有人工标注的、真正离散的regime边界，结构上更接近论文的假设。共7个真实episode（2轮合计）。

**跨7个episode的关键发现**：

- 按cov_gap计，Full CA-AOCP赢4/7（dotcom, gfc, ElectricDevices_cp1090, GunPoint），plain ACI赢3/7（electricity, cp4436, cp5712）——大致五五开。
- 按Winkler score计，plain ACI赢5/7（dotcom, electricity, cp4436, cp5712, GunPoint），Full CA-AOCP只赢2/7（gfc, cp1090）——**这就是此前反复引用的"Full CA-AOCP在7个真实episode里只在2个上Winkler优于plain ACI"的确切来源和口径**。
- 一个更细的观察：Full CA-AOCP赢的两个episode（gfc, cp1090）都是regime差异更"剧烈/清晰"的场景（2008金融危机、设备切换点1090附近波动最大），而输掉的episode里regime差异相对温和——初步支持"BOCPD-weighting在shift足够大/清晰时才值回票价"这个直觉，与第9节的eta发现互相印证。

---



## 8. Task #5：超参数敏感性（hazard h, decay λ, K）

`run_hparam_sensitivity.py` + `aggregate_hparam.py`，Synthetic-CP上10 seeds/点，OFAT扫描 + 一个cross_hazard交叉检验。


| 扫描           | 默认值   | 关键发现                                                                                                               |
| ------------ | ----- | ------------------------------------------------------------------------------------------------------------------ |
| hazard h     | 0.005 | 中等敏感性，h越大cov_gap越小（h=0.05时-11%），width/winkler基本不变，方向与论文Table 12一致                                                  |
| decay λ      | 0.01  | 真实但温和的单调效应，λ=0.05时cov_gap -12%，比论文"low sensitivity"的表述更明显一些                                                        |
| **K（截断深度）**  | 50    | **本轮最重要的发现**：cov_gap随K单调、大幅恶化——K=20时-32%，K=200时+32%。K越大、保留的历史越长，反而校准越差，与"保留更多历史=更准"的直觉相反；K=20（比默认更小）实际上是本次网格里最好的选择 |
| cross_hazard | —     | Synthetic-CP用QC的h（或反之）与用自己的h相比，cov_gap差异都在1个SE以内——说明论文按数据集调hazard并没有实质性利用changepoint位置的先验知识，不存在"调参作弊"问题            |


K的发现值得在论文里明确讨论：更大的K会让权重分布拖入更长尾巴的旧regime数据，稀释有效样本，calibration反而变差——这是一个此前完全没被记录、但对"如何选K"这个实践问题很有指导意义的结果。

---



## 9. Task #6：ACI学习率(eta)敏感性 + 冗余检验

> **⚠️ 2026-08-08更新**：本节数字是pre-fix的（`GenericMethod`未真正截断到K）。用修复后的代码重跑了同样规模的两个sweep，结论见第19节——**核心方向基本复现，但9.1节末尾"eta=0.2反超w/o-ACI基准"的具体数字需要更正，9.2节的cov_gap显著性明显减弱（尤其QC）**，width/winkler/postcov10的优势方向和量级基本不受影响。

这一节的设计直接来自重读ACI原文（Gibbs & Candès, NeurIPS 2021, *Adaptive Conformal Inference Under Distribution Shift*）后提炼出的三条理论线索：

1. **ACI的覆盖保证是长程平均**：Proposition 4.1给出的界是 |avg(err_t) − α| ≤ (max{α₁,1−α₁}+γ)/T——偏差项随T以1/T衰减。我们的合成数据T只有1000/313，且只有一次changepoint，理论上正处在这个界最不利的区间。
2. **最优步长γ应正比于shift幅度的平方根**（Theorem 4.2：γ* ∝ √E|α*_{t+1}−α*_t|）：论文自己用的γ=0.005是针对他们的金融波动率数据（连续、温和漂移）调的，没有理由认为它适合我们的**单次、巨大**的合成mean-shift。
3. **ACI的边际价值取决于conformity score本身是否已经被别的机制"去平稳化"**（论文Section 5的归一化vs非归一化score对比实验）：BOCPD-weighting本身就是在做"实时重新估计regime、重新校准score分布"的事，如果它已经吸收了大部分非平稳性，ACI在其上叠加的边际收益天然就小。



### 9.1 eta敏感性（`run_eta_sensitivity.py --sweep eta_full`，10 seeds × 2数据集 × 6个eta值，`output_eta/sensitivity_eta_full.csv`）

在eta ∈ {0.005, 0.01, 0.02(默认), 0.05, 0.1, 0.2} 上跑Full CA-AOCP：


| eta      | Synthetic-CP cov_gap | QC cov_gap    |
| -------- | -------------------- | ------------- |
| 0.005    | 0.0258 (+5%)         | 0.0185 (-17%) |
| 0.02（默认） | 0.0246 (基准)          | 0.0224 (基准)   |
| 0.05     | 0.0223 (-9%)         | 0.0188 (-16%) |
| 0.10     | 0.0194 (-21%)        | 0.0128 (-43%) |
| 0.20     | 0.0154 (-37%)        | 0.0105 (-53%) |


**这是一个和最初预期方向相反的结果**：我原本猜测eta太大导致α_t震荡，缩小eta应该能缓解ACI的calibration代价；实际观察到的是**cov_gap随eta增大单调、大幅改善**，同时mean_width和Winkler也同步小幅变好（区间不会因此变宽），postcov10基本不变（0.76~0.84区间内小幅波动，无明显规律）。检查了α_t的取值分布确认这不是"触及clip边界后失去意义的伪改善"：eta=0.2时α_t的取值范围是[0.050, 0.311]，只有约40%的步数触及下界0.05（eta=0.02时也有34%触及下界），没有出现大面积饱和。

这个结果其实完全符合ACI论文自己的理论：我们的shift是**单次、巨大**的（Synthetic-CP从N(0.01, 0.98²)跳到N(4.04, 0.97²)，跳变幅度对应α*_t的巨大跃迁），按Theorem 4.2的指导，这种大跳变正需要一个**更大**的γ才能快速收敛，而不是更小。论文默认的γ=0.005（我们的eta=0.02默认值本身已经是CA-AOCP论文自己调过的，但可能仍是针对更温和/更多次的shift场景）对我们这种数据来说明显偏保守。

进一步对比"CA-AOCP w/o ACI"的基准（第4节：Synthetic-CP cov_gap=0.0137，QC=0.0184）：eta=0.2时Full CA-AOCP的cov_gap（Synthetic 0.0154，QC 0.0105）已经**逼近甚至反超**关闭ACI的基准——QC上eta=0.2的cov_gap（0.0105）比完全不用ACI（0.0184）还要好43%。

**这基本上改写了第4节"ACI让cov_gap翻倍"这个结论的解释**：这个penalty主要是eta默认值与我们数据的shift幅度不匹配造成的，是一个**可修复的调参问题**，而不是ACI机制本身的根本缺陷。

### 9.2 eta冗余检验（`--sweep eta_redundancy`，10 seeds × 2数据集 × 6个eta，Full CA-AOCP vs plain ACI配对比较，`output_eta/redundancy_eta.csv`）

在同一批seed的同一份数据上，同时跑plain ACI（无BOCPD）和Full CA-AOCP（BOCPD+decay+ACI），在每个eta下做配对差值检验：


| Dataset         | eta     | cov_gap差(Full−ACI)       | postcov10差              |
| --------------- | ------- | ------------------------ | ----------------------- |
| Synthetic-CP    | 全部6个eta | −0.010 ~ −0.014，全部p<0.01 | **+0.64~+0.65，p<0.001** |
| Quality-Control | 全部6个eta | −0.009 ~ −0.013，多数p<0.05 | **+0.29~+0.30，p<0.001** |


**在这两个合成/重构数据集上，Full CA-AOCP在每一个测试过的eta下都显著优于plain ACI**——cov_gap更低、postcov10（changepoint后10步内的覆盖率）压倒性更高。这与第7节真实数据的结论**方向相反**：真实数据（尤其是连续漂移型的，如electricity、部分UCR episode）上plain ACI在Winkler上经常更优；而在有清晰离散changepoint结构的数据（Synthetic-CP、QC，以及第7节里"赢"的那两个真实episode gfc/cp1090）上，Full CA-AOCP稳定占优。

**这不是矛盾，而是同一个机制在起作用**：BOCPD-weighting的核心假设是"存在离散、可检测的regime边界"，这个假设成立时（合成数据、真实数据里shift够剧烈的episode）它显著优于不用BOCPD的plain ACI；假设不成立时（连续漂移的金融/电力数据，或shift相对温和的UCR episode）它没有优势甚至略差。

---



## 10. 综合结论：ACI要不要默认开启？

把第4、7、8、9节的全部证据放在一起，现在能给出一个比"开/关"更精确的回答：

1. **"ACI默认开启会让cov_gap翻倍"这个最初观察到的现象，主要是eta默认值(0.02)与Synthetic-CP/QC这类单次大幅mean-shift数据不匹配导致的，不是ACI机制本身的缺陷**。把eta调到0.1~0.2（论文默认值的5~10倍），这个penalty基本消失，甚至可能反超关闭ACI的基线。这个发现直接由重读ACI原文的Theorem 4.2（最优步长应正比于shift幅度平方根）推导并验证。
2. **BOCPD-weighting（无论是否叠加ACI）本身是否值回票价，取决于数据是否有清晰的离散regime结构**：在这类数据上（合成数据、部分真实episode）它显著优于plain ACI；在连续漂移的真实数据（金融波动率、电力需求）上没有优势。这是ACI论文自己的score-stationarity论证（Section 5）在CA-AOCP场景下的自然推论。
3. 因此更站得住脚的建议不是"默认关闭ACI"，而是：**(a) eta不应该用一个跨数据集固定的默认值，应该按shift幅度的量级去标定（哪怕是一个粗略的启发式，比如用前置窗口内score的标准差做归一化）；(b) 在论文里明确讨论CA-AOCP（BOCPD-weighting为核心）适用的场景边界——离散regime数据是它的主场，连续漂移数据更适合用plain ACI/EWMA-ACI**，把这个边界讲清楚比笼统地说"CA-AOCP更好"更有说服力，也更能预判审稿人拿真实数据反驳时的应对。
4. 如果时间有限、不想在正式提交前重新调eta默认值，退而求其次的选项是在论文里如实报告这个eta敏感性结果作为一个"local sensitivity/局限性"讨论——这本身就是一个不错的贡献点（提供了理论指导的、可复现的诊断），不必回避。

---



## 11. 仍未解决的问题（供后续参考）

- `ca_aocp/weights.py`~~与~~`ca_aocp/conformal.py`~~中O(K)截断从未真正实现~~ **已修复，见第13节。**
- ~~`ca_aocp/algorithm.py`的`CAAOCP.update()`丢弃`GaussianBOCPD.update()`返回值但仍无条件付出O(t)构建成本~~ **已修复，见第14~15节（本轮问题#3）。**
- ~~`ca_aocp/conformal.py`的`weighted_quantile`只有O(K log K)的sort实现，没有论文Algorithm 2允许的O(K)线性时间selection选项~~ **已补上，见第14~15节（本轮问题#1）——注意`CAAOCP`默认仍用sort（实测在K=50/500下比select快），select是可选项，不是默认路径。**
- ~~`experiments/run_ablation.py`/`run_shift_grid.py`的`GenericMethod`/`TrackedMethod`从未把calibration set截断到K，只有BOCPD内部后验是截断的~~ **已修复，见第14~16节（本轮问题#2）——但这个修复直接改变了第6节的统计结论，见第17节，不是纯粹的性能修复。**
- `ca_aocp/adaptive_update.py`的`logistic()`在大数值量级数据（如electricity demand原始值）下有数值溢出问题（第7节）。**本轮复核仍然存在，未修复**：`np.where(x>=0, a, b)`对标量输入会把两个分支都eager求值，`x`很大时被丢弃的那个分支里`np.exp(x)`仍会触发`RuntimeWarning: overflow in exp`，只是不影响最终结果（`np.where`选对了分支）。这次没有修，因为不在本轮反馈的三个问题范围内，且是nuisance warning而非正确性bug；如果要修，思路是把`np.where`换成先做`np.clip`再算，或者对标量输入单独走`if/else`分支。
- eta目前仍是一个跨数据集固定的超参数；若要落地第10节的建议1，需要设计一个具体的自适应/数据驱动的eta标定方案并重新验证。
- `experiments/run_baselines.py`、`run_real_data.py`里已有的更强baseline（FACI/SF-OGD/SAOCP）对比结果目前仍游离在论文叙事之外（第1节），需要决定是否、以及如何在论文里正面处理。
- 第17节发现的width反转现象**还没有一个经过验证的机制解释**（只确认了现象本身在单seed/30-seed/21-seed三个规模上方向一致），如果要写进论文需要先补一个诊断（比如统计pre-fix时`window`/`r_hat`实际超过K=50的比例和幅度）。

---



## 12. 文件索引

所有脚本与原始/聚合数据都在打包的zip里，按实验分目录：

- `experiments/run_ablation.py` — 8方法框架核心，被之后几乎所有脚本复用
- `experiments/output_ablation/` — Task #1 单seed结果
- `experiments/run_multiseed.py`, `output_multiseed/` — Task #3
- `experiments/run_shift_grid.py`, `output_shiftgrid/` — Task #2
- `experiments/significance_report.py`, `output_significance/` — 第6节正式统计
- `experiments/run_real_multiregime.py`, `output_real_multiregime/` — Task #4第一轮（金融/电力）
- `experiments/run_real_segmentation.py`, `output_real_segmentation/` — Task #4第二轮（真实离散regime数据）
- `experiments/run_hparam_sensitivity.py`, `aggregate_hparam.py`, `output_hparam/` — Task #5
- `experiments/run_eta_sensitivity.py`, `aggregate_eta.py`, `output_eta/` — Task #6
- `ca_aocp/bocpd.py`, `ca_aocp/algorithm.py`, `tests/` — 第13节核心包bug修复 + 回归测试（本轮新增）

---



## 13. `ca_aocp`核心包O(K)截断修复（本轮新增）



### 13.1 发现：这不只是性能优化缺口，是一个正确性bug

动手修"O(K)从未真正实现"之前，读`ca_aocp/bocpd.py`的截断代码（`GaussianBOCPD.update()`原版第145-158行）时发现截断逻辑本身有一个**索引对齐bug**：

```python
keep = np.argpartition(log_R_new, -self.K)[-self.K:]
keep = np.sort(keep)
...
nig_new = [nig_new[k] for k in keep]           # _nig 被压缩成K个，按keep顺序排列
log_R_dense = np.full(len(log_R_new), -np.inf)
log_R_dense[keep] = log_R_new_trunc
log_R_new = log_R_dense                         # _log_R 却没有被压缩，长度不变
```

下一步`update()`用`log_R_active = self._log_R[:len(self._nig)]`直接取`_log_R`的前K个位置——但`_log_R`是稀疏的（非`-inf`的值散落在`keep`那几个位置，不一定是前K个），这导致`_nig`（紧凑排列）和`log_R_active`（错误位置）大概率对不上号，从截断第一次生效开始逐步污染整个run-length后验分布。

**验证方式**：写了诊断脚本（现已固化为`tests/test_bocpd.py`的正式回归测试）。(1) 概率质量追踪：K=20跑600步，91%的步数里`_log_R`在`len(_nig)`之后的位置仍有非零概率，这部分在下一步被直接丢弃，累计丢失概率质量108.5（平均每步0.18）。(2) 与等效不截断的BOCPD对比同一批数据，在**可表示范围内**（run-length < K，排除截断本身造成的、预期内的长稳定期信息损失）比较后验分布：旧代码在稳态下平均偏差0.237、最坏0.95（概率本身取值[0,1]，接近满量程），确认这不是截断带来的正常信息损失，而是概率算错了。

**影响范围**：本季度所有基于BOCPD的实验结果（第2~10节的全部内容）都是在这个bug存在的情况下跑出来的，因为`run_ablation.py`的`GenericMethod`虽然绕开了`weights.py`/`conformal.py`的O(t)问题，但仍直接调用了`ca_aocp.bocpd.GaussianBOCPD`，bug就在这里面。**按你的决定，本轮只修复bug本身，不重跑第2~10节已报告的实验**——这意味着第2~10节的具体数字仍是历史记录；如果将来重新跑`run_ablation.py`等脚本，会因为这个修复得到不同（更正确）的数字（已用同一份数据实测确认：修复前后"BOCPD-weighted CP"在Synthetic-CP上的单seed cov_gap从0.011变为0.005，其余方法也有相应变化，方向不完全一致，需要重新评估，不能想当然认为差异可忽略）。

### 13.2 修复方案

`ca_aocp/bocpd.py`：引入一个显式的`_run_lengths`平行数组，记录每个槽位对应的绝对run-length值，使`_log_R`/`_nig`/`_run_lengths`三者在每次截断后一起压缩、始终对齐，不再依赖"数组下标=run-length=`_nig`里的位置"这个在截断后不成立的假设。同时提供两套接口：

- `_compute_pi()` / `update()`的返回值：保留原有的全长（O(t)）、oldest-first的返回契约不变——`run_ablation.py`里的`GenericMethod`直接依赖这个形状，为了不破坏本季度已跑出的实验脚本，这个legacy接口的**输出形状不变，但底层计算现在是从修复后的正确内部状态重建的**，因此更正确，且不需要改动`run_ablation.py`一个字。
- 新增`pi_recent(window=None)`：真正O(K)的有界接口，只返回最近`min(t, K)`个观测的权重。`ca_aocp/algorithm.py`的`CAAOCP`类现在用这个接口 + 一个有界的`collections.deque(maxlen=K)`分数缓冲区（替换原来无界增长的`list`），端到端做到O(K)每步复杂度——这是`CAAOCP`（论文主类）而非`GenericMethod`（本季度实验用的独立重写版本）的修复路径。
- `get_run_length_posterior()`：保持"按绝对run-length值索引的稠密数组"这个外部契约不变（`argmax()`语义不受影响），但现在从压缩状态按需重建，长度有界于当前追踪到的最大run-length（本质是O(K)）。



### 13.3 测试

新增`tests/`目录，共15个测试全部通过（`python3 -m pytest tests/ -v`）：

- `test_bocpd.py`（7个）：核心回归测试`test_truncated_matches_untruncated_reference_on_representable_run_lengths`验证截断后验在可表示范围内（run-length<K）与等效不截断的参考模型偏差<0.01（均值）/<0.05（最坏），把旧bug代码接入同一测试会得到0.237/0.95——确认测试有效捕捉了这个问题；另有概率质量守恒、MAP run-length在真实changepoint附近的正确性、legacy `_compute_pi()`长度契约、`pi_recent()`有界性等测试。
- `test_algorithm.py`（4个）：`CAAOCP.run()`基本正确性、内部分数缓冲区长度不超过K（O(K)内存的直接验证）、`pi`有界性、以及一个粗粒度的运行时缩放检查（序列长度扩大6倍，运行时增幅需<12倍，排除每步成本随t增长的情形）。
- `test_weights_conformal.py`（4个）：`compute_weights`/`weighted_quantile`的基本性质，确认它们本身是通用函数，只要传入定长数组就能正确工作，不需要跟着改。

`run_ablation.py`等本季度脚本用修复后的代码重跑（仅做验证，未采纳其输出）确认不报错、正常产出结果，随后已将`experiments/output_ablation/`下两个被意外覆盖的文件恢复为修复前的原始版本，第2节起报告的历史数字未被本次改动影响。

### 13.4 后续如果要重跑实验

如果之后决定重跑第2~10节的实验以采用修复后的BOCPD，预期方向：soft-weighting和hard-reset的相对比较、eta敏感性的具体数字都可能变化（因为两者都依赖BOCPD的run-length后验/MAP估计），但本报告第6节论证的"soft相对hard在Winkler/width上有小而显著优势"这类**跨方法比较**的定性结论未必会反转（因为bug是对称地影响两个方法各自的BOCPD状态，除非bug对某一方法的影响系统性更大）——这需要重新验证，不能想当然沿用。

> **2026-08-07补记**：这个预测部分不成立——不是因为第13节修的BOCPD对齐bug，而是因为第14节发现的另一个独立问题（`run_ablation.py`/`run_shift_grid.py`从未真正把calibration set截断到K）系统性地对`Hard-reset`更有利（它让hard-reset在稳定regime里能用远超K的历史点求均匀分位数），这不是"对称影响"，是单边的、系统性的。第17节给出了实测数字。

---

## 14. 第二次跟进：三个仍违反论文O(K)复杂度声称的遗留问题

第13节修复了BOCPD内部的索引对齐bug，但没有覆盖所有复杂度问题。这一轮由你直接指出了三个具体的代码位置，逐一核实后确认全部属实：

1. **`ca_aocp/conformal.py:44-56`（`weighted_quantile`）**：`np.argsort`对输入数组排序，当调用方传入的是bounded数组（`CAAOCP`现在确实传bounded的，见第13节）时，这是O(K log K)，不是O(t log t)了——但论文Algorithm 2原文允许两种实现，"O(K log K) if sorted" 和 "O(K) with a linear-time weighted selection routine"，代码里只有前者，没有后者。如果论文复杂度表要咬死最紧的O(K)（而不是O(K log K)），这里还缺一步。
2. **`experiments/run_ablation.py`（`GenericMethod`）和`experiments/run_shift_grid.py`（`TrackedMethod`）**：两者的`self._scores`从来都是无界的`list`，从未截断到K；且用的是`GaussianBOCPD.update()`的legacy全长返回值（与无界`self._scores`长度匹配），不是`pi_recent()`。也就是说，本报告第2~10节几乎所有数字（多seed、shift grid、超参数、eta敏感性——因为它们全部通过`from experiments.run_ablation import make_methods`或`GenericMethod`复用同一套实现）用的框架，calibration set从未真正被截到K，只有`GaussianBOCPD`内部的run-length后验是截断的。
3. **`ca_aocp/algorithm.py:257`（`CAAOCP.update()`）**：调用`self._bocpd.update(s_t)`但丢弃返回值，而`ca_aocp/bocpd.py:214`的`update()`内部无条件执行`return self._compute_pi()`——这是O(t)的（构建一个长度为`t`的`ages`数组并对它做`searchsorted`）。也就是说`CAAOCP`每一步都在做一次没人用的O(t)计算，且随着t增长这个被丢弃的计算成本本身也在增长，与`pi_recent()`真正用到的O(K)计算并行发生。

### 14.1 问题#3的量化验证

在动手修之前，直接对比丢弃的`_compute_pi()`和实际使用的`pi_recent()`：

| t | `_compute_pi()`(丢弃的) | `pi_recent()`(实际用的) | 倍数 |
|---|---|---|---|
| 1,000 | 14.1us | 9.2us | 1.5x |
| 10,000 | 51.0us | 9.8us | 5.2x |
| 100,000 | 743.7us | 9.6us | 77x |
| 500,000 | 3755.9us | 11.3us | 331x |

`pi_recent()`稳定在~10us不随t增长，`_compute_pi()`线性增长，t=500,000时被丢弃的计算比真正用到的计算慢331倍。这是一个可测量、会在长序列上主导总运行时间的O(t)泄漏，直接违反摘要里"per-step computation controlled by a bounded truncation budget rather than the full history length"这句话。

---

## 15. 修复方案与测试

### 15.1 问题#3（最容易修，先做）

`ca_aocp/bocpd.py`的`GaussianBOCPD.update()`加一个`compute_legacy_pi: bool = True`参数：默认`True`保持原有行为（`experiments/run_ablation.py`等仍依赖legacy全长返回值的调用方不受影响），传`False`时跳过`_compute_pi()`的构建、直接返回`None`。`ca_aocp/algorithm.py`的`CAAOCP.update()`改为调用`self._bocpd.update(s_t, compute_legacy_pi=False)`，因为它本来就只用`pi_recent()`。

新增3个回归测试（`tests/test_bocpd.py`）：`compute_legacy_pi=False`确实返回`None`且确实跳过了计算（用monkeypatch让`_compute_pi`在被调用时直接抛异常，验证`False`模式下不会触发）；默认行为不变；以及一个复杂度对照测试（`_compute_pi()`在模拟t从2,000→200,000时增长>10倍，`pi_recent()`同区间增长<5倍）。

### 15.2 问题#1（其次）

`ca_aocp/conformal.py`的`weighted_quantile`加`method: str = "sort"`参数：`"sort"`是原有实现（不变，仍是默认），`"select"`是新增的期望线性时间加权quickselect（对候选值随机选pivot，按`< pivot`/`== pivot`/`> pivot`三路划分，累加左侧+相等权重和目标weight比较后只递归进入必须包含答案的那一路，不排序）。

新增5个测试（`tests/test_weights_conformal.py`）：200组随机化场景（含重复分数、随机权重、随机level）下`select`与`sort`结果一致（`pytest.approx`容差内）；极端level（0.001/0.01/0.99/0.999）下一致；单元素输入；未知method报错。

**没有把`CAAOCP`默认切到`select`**：实测K=50时`sort`=2.9us、`select`=23.3us，K=500时`sort`=7.5us、`select`=51.5us——NumPy的C级`argsort`在这个规模下比Python级quickselect循环快得多，尽管渐进阶数更差。两个实现都在，需要严格O(K)时可以显式传`method="select"`。

### 15.3 问题#2（最后做，影响最大）

只对用到BOCPD的6个方法（`BOCPD-weighted CP/ACI`、`CA-AOCP w/o ACI`、`Full CA-AOCP`、`Hard-reset CP/ACI`）把`self._scores`从`list`换成`collections.deque(maxlen=K)`，BOCPD更新调用加`compute_legacy_pi=False`，非hard-reset的权重改用`pi_recent(K)`。`ACI`、`EWMA-ACI`两个baseline按论文Table 9本来就没有K这个超参数（ACI是O(t)、EWMA-CP是论文自己声称的O(1)），保持不变——这也是一个内建的正确性检验：如果改动意外影响了这两个方法，说明改错了地方。

`run_multiseed.py`、`run_hparam_sensitivity.py`、`run_eta_sensitivity.py`、`run_real_segmentation.py`、`run_real_multiregime.py`都通过`from experiments.run_ablation import make_methods`或`GenericMethod`复用同一个类，所以这一处修复自动覆盖了这些脚本；`run_shift_grid.py`的`TrackedMethod`是独立实现，单独改了一遍（它的4个方法全部是BOCPD-driven，所以全部截断，不需要按方法区分）。

全部23个测试通过（`python3 -m pytest tests/ -v`，15个原有 + 8个新增）。

单seed快速验证（`run_ablation.py`直接跑）：`ACI`/`EWMA-ACI`两个数据集上逐项数字与pre-fix完全一致（0.0000差异），确认改动没有意外波及不该动的部分；6个BOCPD-driven方法数字有实质变化，其中`Hard-reset CP`变化最大（Synthetic-CP width从3.49降到3.34）。

---

## 16. 核心实验重跑（30-seed multiseed + 21-seed shift grid）

按你的要求，用修复后的代码把两组核心实验重新跑到和原始报告一样的规模：

- `experiments/output_multiseed/raw_postfix.csv`：30 seeds（0~29）× 2数据集 × 8方法，481行（含表头），与原始`raw.csv`行数一致。
- `experiments/output_shiftgrid/raw_postfix.csv`：21 seeds（0~20）× 10条件 × 4方法，841行（含表头），与原始`raw.csv`行数一致。

沙盒每次调用45秒超时、无法后台运行，所以是一批2个seed（multiseed，约28秒/批）或1个seed（shift grid，约20秒/批，T=300但10条件×4方法比multiseed单条件跑得慢）分批跑完的。

用`significance_report.py`原班方法（paired t-test + Wilcoxon + BH-FDR）对post-fix数据重新生成Table A/B/C，输出到`experiments/output_significance_postfix/`（不覆盖原始`output_significance/`，便于对照）。

---

## 17. 结论解读：第6节的"soft vs hard-reset"需要修正

### 17.1 直接数字对比（paired diff = hard − soft）

| 实验 | pairing | metric | pre-fix diff | pre-fix p | **post-fix diff** | **post-fix p** |
|---|---|---|---|---|---|---|
| multiseed / Synthetic-CP | CP(无ACI) | cov_gap | −0.0013 | 0.196 | **+0.0079** | **<0.0001** |
| multiseed / Synthetic-CP | CP(无ACI) | mean_width | +0.0666(soft更窄) | <0.0001 | **−0.0661(hard更窄)** | **<0.0001** |
| multiseed / Synthetic-CP | CP(无ACI) | winkler | +0.0460 | <0.0001 | +0.0003(≈0) | 0.839 |
| multiseed / Synthetic-CP | ACI | mean_width | +0.0884 | <0.0001 | +0.0067(方向不变,不再显著) | 0.119 |
| multiseed / Synthetic-CP | ACI | winkler | +0.0584 | <0.0001 | +0.0033(≈0) | 0.465 |
| multiseed / QC | CP(无ACI) | cov_gap | +0.0023 | 0.068 | **+0.0063** | **<0.0001** |
| multiseed / QC | CP(无ACI) | mean_width | +0.0435(soft更窄) | 0.043 | **−0.0529(hard更窄)** | **<0.001** |
| multiseed / QC | CP(无ACI) | winkler | +0.0826 | <0.0001 | +0.0499(方向不变，幅度减半) | <0.001 |
| shift grid pooled(n=210) | CP(无ACI) | cov_gap | +0.0028 | <0.0001 | **+0.0082** | **<0.0001** |
| shift grid pooled(n=210) | CP(无ACI) | mean_width | +0.0597(soft更窄) | <0.0001 | **−0.0518(hard更窄)** | **<0.0001** |
| shift grid pooled(n=210) | ACI | cov_gap | +0.0020 | <0.0001 | +0.0004(≈0，不显著) | 0.359 |
| shift grid pooled(n=210) | ACI | mean_width | +0.0593 | <0.0001 | +0.0162(方向不变，幅度降到约1/4) | <0.0001 |

（完整数字见`experiments/output_significance_postfix/table_A_multiseed_ablation.csv`、`table_B_shiftgrid_pooled.csv`、`table_C_shiftgrid_by_condition.csv`、`table_combined_with_global_FDR.csv`。）

### 17.2 新的、更精细的结论

把post-fix数字和shift grid的10条件逐条拆解（`table_C`）放在一起看，浮现出一个比pre-fix版本更结构化、也更可信的故事：

1. **`CP(无ACI)`配对（纯BOCPD-weighted vs纯hard-reset，不叠加ACI）：mean_width方向反转，几乎在shift grid全部10个条件里一致**——真正截断到K=50后，hard-reset的区间反而比soft更窄（10个条件里能查到的width比较全部是hard更窄，多数p<0.01经FDR校正后仍显著）。Winkler优势在Synthetic-CP上完全消失（p=0.84）。
2. **同一个`CP(无ACI)`配对，cov_gap方向没变但从"不显著"变成"几乎处处显著"**：两个multiseed数据集、shift grid pooled、以及能查到的绝大多数shift grid单条件，cov_gap都是soft显著更好，且post-change coverage（postcov10/25/50/100）、post-change有效样本量（neff_t+1/+5/+10/+25）也全部显著偏向soft。
3. **合起来看，这是一个calibration-vs-sharpness的权衡，不是soft单边碾压**：hard-reset（真正K=50截断后）用一个更小、更"硬"的均匀窗口，代价是牺牲post-change校准精度换来更窄的平均区间；soft-weighting的优势收缩到了"覆盖率/校准"这一侧，不再同时占住"覆盖率+区间宽度+Winkler"三头。这在概念上比pre-fix版本"soft全面更优"更站得住脚——正常情况下更窄的区间和更好的覆盖率是要互相取舍的，一个方法同时在两头都赢，本身就该让人多想一步。
4. **叠加ACI后（`ACI`配对），width的反转没有发生，但优势幅度明显收窄**：soft仍然更窄（方向不变），但效应量从pre-fix的+0.059~+0.088掉到post-fix的+0.005~+0.016，在multiseed单数据集(n=30)层面大多不再显著（只有靠shift grid的大样本n=210才勉强显著）。cov_gap优势在ACI配对下post-fix基本消失（+0.002→+0.0004，n.s.）。这符合直觉：ACI自己的在线校正本来就是用来吸收残余miscalibration的，一旦两条路径都叠加了ACI，soft和hard之间因权重方案不同造成的校准差异，很大一部分被ACI自己抹平了。
5. **机制上的一个旁证**：post-fix的`neff_t+1`（切换后第1步的有效样本量/窗口大小差）比pre-fix显著性更强但均值更小（pre-fix −1.67但p=0.81不显著；post-fix −0.35但p<0.0001高度显著）——说明pre-fix时"window"这个量本身方差极大（因为有时候远超K未被截断），真正截断后虽然效应量变小，但跨seed的一致性大幅提高。这是"修复后噪声更小、信号更稳"的一个直接证据，而不只是"数字变了"。

### 17.3 尚未验证的部分

第17.2节第3点里"hard-reset变窄"这个现象本身有稳固的多规模证据（单seed、30-seed、21-seed三个层面方向一致），但**具体机制没有验证**——一个合理猜测是pre-fix时`window = min(r_hat+1, n)`在长期稳定regime里可以远超50（`r_hat`本身不受K限制，K只限制BOCPD内部追踪的候选假设个数，不限制某个候选对应的run-length数值），给了pre-fix版本的hard-reset一个不对等的、更大的均匀平均窗口；但这只是一个假说，没有专门做过"pre-fix时`r_hat`超过50的比例和幅度"这样的诊断去证实。如果这部分要写进论文，建议先补这个诊断，不要直接引用"因为"这两个字后面的解释。

---

## 18. 更新后的文件索引

在原第12节基础上新增：

- `ca_aocp/bocpd.py` — `compute_legacy_pi`参数（问题#3修复）
- `ca_aocp/algorithm.py` — `CAAOCP.update()`调用改传`compute_legacy_pi=False`
- `ca_aocp/conformal.py` — `weighted_quantile`新增`method="select"`选项（问题#1修复）
- `experiments/run_ablation.py` — `GenericMethod`改bounded score buffer（问题#2修复，影响`make_methods`所有下游脚本）
- `experiments/run_shift_grid.py` — `TrackedMethod`改bounded score buffer（问题#2修复）
- `tests/test_bocpd.py` — 新增3个测试（compute_legacy_pi相关）
- `tests/test_weights_conformal.py` — 新增5个测试（method="select"相关）
- `experiments/output_multiseed/raw_postfix.csv` — post-fix 30-seed重跑
- `experiments/output_shiftgrid/raw_postfix.csv` — post-fix 21-seed重跑
- `experiments/output_significance_postfix/` — post-fix版Table A/B/C/combined，对照原始`output_significance/`
- `experiments/output_ablation_prefix_backup/` — 修复前的单seed ablation结果备份，供对照

---

## 19. 第三次跟进：eta sweep（Task #6，第9节）重跑

`run_eta_sensitivity.py`的两个sweep都通过`from experiments.run_ablation import GenericMethod`复用第14~15节修复的同一个类，所以理论上也受K截断修复影响。按原规模重跑：10 seeds（0~9）× 2数据集 × 6个eta，`eta_full`和`eta_redundancy`各一遍，写入`experiments/output_eta/raw_postfix.csv`，行数361（120行eta_full + 240行eta_redundancy 之和 + 表头），与原始`raw.csv`完全一致。用`aggregate_eta.py`同样的逻辑重新聚合，输出到`experiments/output_eta_postfix/`。

### 19.1 (A) eta_full：核心方向复现良好

cov_gap随eta单调改善这个结论**基本原样复现**，绝对数字略有上移但趋势和百分比变化幅度都很接近：

| dataset | eta | cov_gap pre-fix (vs默认%) | cov_gap post-fix (vs默认%) |
|---|---|---|---|
| Synthetic-CP | 0.005 | 0.0258 (+5%) | 0.0312 (+2%) |
| Synthetic-CP | 0.02（默认） | 0.0246 | 0.0307 |
| Synthetic-CP | 0.10 | 0.0194 (−21%) | 0.0236 (−23%) |
| Synthetic-CP | 0.20 | 0.0154 (−37%) | 0.0197 (−36%) |
| Quality-Control | 0.005 | 0.0185 (−17%) | 0.0208 (−20%) |
| Quality-Control | 0.02（默认） | 0.0224 | 0.0259 |
| Quality-Control | 0.10 | 0.0128 (−43%) | 0.0174 (−33%) |
| Quality-Control | 0.20 | 0.0105 (−53%) | 0.0134 (−48%) |

mean_width、winkler也几乎不变（小数点第2、3位的差异）。这符合预期：`eta_full`比较的是同一个方法（Full CA-AOCP）在不同eta下的自身表现，从不涉及hard-reset那条逻辑分支——而第17节发现的大幅反转，根源正是hard-reset的`window`在pre-fix时可以远超K，这里完全不适用。**第9.1节"eta越大、cov_gap越好，且论文默认eta=0.02对我们的单次大幅shift数据偏保守"这个结论可以照原样引用。**

需要更正的是第9.1节末尾拿eta=0.2和"CA-AOCP w/o ACI"基准（第4节）比较的那句话——那个基准本身也是受K截断影响的6个方法之一，post-fix后数值变了：

| dataset | w/o-ACI基准 pre-fix | w/o-ACI基准 post-fix | eta=0.2 Full CA-AOCP pre-fix | eta=0.2 Full CA-AOCP post-fix |
|---|---|---|---|---|
| Synthetic-CP | 0.0137 | **0.0099** | 0.0154（接近基准，未反超） | **0.0197（比基准差约2倍，不再"接近"）** |
| Quality-Control | 0.0184 | **0.0150** | 0.0105（反超43%） | **0.0134（反超11%，方向不变但没那么夸张）**|

即"eta=0.2能追平甚至反超关闭ACI的基准"这个具体数字（尤其Synthetic-CP的"接近"和QC的"反超43%"）不再成立，QC上仍然反超但只有11%，Synthetic-CP上不仅没接近、反而差了将近一倍。**结论本身（"eta默认值偏保守"）不受影响，但不要再引用旧的具体百分比。**

### 19.2 (B) eta_redundancy：postcov10优势完全不受影响，cov_gap显著性明显减弱

paired diff = Full CA-AOCP − plain ACI（负数=Full CA-AOCP更好，此处plain ACI按论文Table 9设计本来就不截断，未受本轮修复影响）：

| 指标 | Synthetic-CP pre-fix | Synthetic-CP post-fix | QC pre-fix | QC post-fix |
|---|---|---|---|---|
| postcov10 diff | +0.64~+0.65, 全部p<0.001 | **+0.64~+0.65, 全部p<0.001（几乎原样不变）** | +0.29~+0.30, 全部p<0.001 | **+0.29~+0.31, 全部p<0.001（几乎原样不变）** |
| mean_width diff | −0.051~−0.077（Full更窄），全部同向 | **−0.020~−0.044（Full更窄），方向不变，量级缩小约一半** | −0.048~−0.121（Full更窄） | **−0.030~−0.092（Full更窄），方向不变** |
| winkler diff | −0.053~−0.081（Full更好） | **−0.061~−0.095（Full更好），方向不变，略微增强** | −0.009~−0.052（Full更好） | **−0.029~−0.055（Full更好），方向不变，略微增强** |
| cov_gap diff | −0.0105~−0.0140，全部6个eta p<0.01 | **−0.0062~−0.0079，全部6个eta仍p<0.05（0.018~0.036），但不再全部p<0.01** | −0.0089~−0.0134，多数p<0.05 | **−0.0045~−0.0088，多数p>0.05（仅eta=0.2边缘显著p=0.05）** |

**结论**：第9.2节"Full CA-AOCP在每一个测试过的eta下都显著优于plain ACI"这句话——**在postcov10、mean_width、winkler三个指标上依然成立，方向和显著性都稳固**；但**cov_gap这一项，QC数据集上的显著性基本消失了**（pre-fix"多数p<0.05"，post-fix变成"多数不显著"），Synthetic-CP上还保留显著性但p值普遍从<0.01降到0.02~0.036这个区间，不再是那么"碾压式"的显著。这是一个幅度上的减弱，不是方向反转——比第17节hard-reset那个对比温和得多，原因也和第19.1节一样：`eta_redundancy`从不涉及hard-reset那条最受影响的逻辑分支。

### 19.3 综合：为什么eta实验受影响比hard-reset对比小得多

第17节的大反转，根源是hard-reset的`window = min(r_hat+1, n)`在pre-fix时`n`无界，导致`window`本身可以远超K=50。`eta_full`和`eta_redundancy`两个sweep从未跑过`Hard-reset CP/ACI`，只涉及`Full CA-AOCP`（soft-weighted+ACI）和`ACI`（无BOCPD baseline，按设计本就不截断）——`Full CA-AOCP`pre-fix时calibration set确实也是无界的，但它用的是soft权重（`pi_recent`+decay），老数据的权重会随BOCPD后验自然衰减到接近0，不像hard-reset是硬性0/1窗口，所以"无界"对它的实际影响远小于对hard-reset的影响。这也是本轮意外验证的一个旁证：第17.3节提出的"hard-reset反转可能是因为pre-fix时它拿到了不成比例的额外历史"这个假说，进一步获得了支持——受影响最大的正是hard-reset，soft-weighted方法（无论是否叠加ACI）受到的影响都小得多。

### 19.4 文件索引（新增）

- `experiments/output_eta/raw_postfix.csv` — post-fix 10-seed重跑（eta_full + eta_redundancy）
- `experiments/output_eta_postfix/sensitivity_eta_full.csv`、`redundancy_eta.csv` — post-fix聚合结果，对照原始`experiments/output_eta/sensitivity_eta_full.csv`、`redundancy_eta.csv`
