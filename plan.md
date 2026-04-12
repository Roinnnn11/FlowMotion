# FlowMotion 三阶段研究计划

## 项目概览

| 阶段 | 目标会议 | Deadline | 核心方法 | 状态 |
|------|---------|----------|---------|------|
| Phase 1 | CGI 2025 | **4月20日（9天）** | Per-scene FM motion field | 🔴 进行中 |
| Phase 2 | PRCV 2025 | 4月30日（19天） | Phase1 + 更完整实验 | 待开始 |
| Phase 3 | NeurIPS 2025 | 5月中旬 | Multi-scene prior + scene flow | 待开始 |
| Future | CVPR/ICCV 2026 | — | 大规模泛化，Flow4R 方向 | 规划中 |

---

# Phase 1：CGI 冲刺（4月11日 → 4月20日，9天）

## 定位

**投稿级别**：CGI（The Visual Computer / Computers & Graphics）
**核心 claim**：用 Flow Matching 替换 SplineGS 的 Hermite spline trajectory，在 per-scene dynamic Gaussian splatting 中验证 temporal consistency 优势

## 题目（暂定）

**Flow-Matched Motion Fields for Dynamic Gaussian Splatting**

## 方法最小闭环

### 替换目标
`gaussian_model.py:54` 的 `inverse_cubic_hermite` 函数及 `control_point` 参数组
→ 替换为 FM velocity field `v_θ(x, t, τ)`

### FM 公式（最小版本）

构造相邻帧 Gaussian 位置对：
```
X_τ, X_{τ+Δτ}  （来自场景自身的 Gaussian center 轨迹）

线性插值：X_t = (1-t)·X_τ + t·X_{τ+Δτ}
Target velocity：v* = X_{τ+Δτ} - X_τ
FM loss：L_FM = ||v_θ(X_t, t, τ) - v*||²
```

### 必须包含的损失
- render reconstruction loss（保留原版）
- FM loss（新增）
- temporal smoothness regularization（新增）

### 实验组（最小）
1. SplineGS 原版（baseline）
2. Ours w/o FM（确定性 MLP motion）
3. Ours full（FM motion field）

### 数据
- Nvidia dataset（Balloon1, Balloon2, Jumping）
- 如有时间：D-NeRF 1~2 个场景

## 9天任务分配

### Day 1（今天，4月11日）
**目标：跑通 SplineGS 原版**
- [ ] 确认 data 就位（`data/nvidia_rodynrf/Balloon1/` 结构正确）
- [ ] 跑通 `python train.py -s data/nvidia_rodynrf/Balloon1/ --expname baseline_b1 --configs arguments/nvidia_rodynrf/Balloon1.py`
- [ ] 记录 loss 曲线，导出渲染视频
- [ ] 理解 `gaussian_model.py:inverse_cubic_hermite` 和 `control_point` 的调用链

**晚上 checkpoint**：能完整训练并渲染 Balloon1，知道 spline 在哪里被调用

### Day 2（4月12日）
**目标：抽象 motion 接口，写 deterministic MLP baseline**
- [ ] 把 `inverse_cubic_hermite` 封装，定义统一接口 `motion_model(x0, tau) -> delta_x`
- [ ] 新建 `scene/motion_models.py`，包含：
  - `SplineMotionModel`（原版封装）
  - `MLPMotionModel`（简单 MLP，作为 w/o FM baseline）
- [ ] 接入训练，保证原版 spline 不受影响

**晚上 checkpoint**：两个 motion model 都能切换使用，原版训练正常

### Day 3（4月13日）
**目标：实现 FM motion module**
- [ ] 新建 `FMMotionModel`：输入 `(x, t_flow, tau)`，输出 velocity
- [ ] 实现 FM loss 计算（线性插值 + velocity matching）
- [ ] 接入训练 loop（`train.py` 中加 FM loss 项）
- [ ] 在 Balloon1 上跑第一次 FM 训练，看 loss 是否下降

**晚上 checkpoint**：FM loss 正常下降，没有明显 NaN 或爆炸

### Day 4（4月14日）
**目标：加 regularization，稳定训练**
- [ ] 加 temporal smoothness loss
- [ ] 加 transport consistency loss（可选，如果 Day3 稳定）
- [ ] 调 loss 权重（lambda_fm, lambda_smooth）
- [ ] 跑 Balloon1 完整训练，渲染视频，对比 SplineGS

**晚上 checkpoint**：渲染效果不比 SplineGS 差太多，运动更自然

### Day 5（4月15日）
**目标：跑完 3 组对比实验**
- [ ] 跑 SplineGS 原版（Balloon1, Balloon2, Jumping）
- [ ] 跑 Ours w/o FM（MLPMotion）
- [ ] 跑 Ours full（FMMotion）
- [ ] 统计 PSNR/SSIM/LPIPS

**晚上 checkpoint**：主实验数字第一版，至少一个场景有提升

### Day 6（4月16日）
**目标：数字 + 可视化齐全**
- [ ] 跑 ablation：w/o smoothness，w/o FM
- [ ] 制作连续帧对比图（时序 consistency 可视化）
- [ ] 制作 loss 曲线图
- [ ] 备份所有 checkpoint 和视频

**晚上 checkpoint**：论文级图表初版完成

### Day 7（4月17日）
**目标：写论文主体**
- [ ] Method 部分（公式 + 架构图）
- [ ] Experiments 部分（表格 + 图）
- [ ] 方法总览图

**晚上 checkpoint**：论文 Method + Experiments 初稿

### Day 8（4月18日）
**目标：写完论文**
- [ ] Introduction（讲清楚 spline → FM 的转变）
- [ ] Related Work（3DGS动态, Flow Matching, trajectory-based methods）
- [ ] Conclusion + Limitations
- [ ] Abstract

**晚上 checkpoint**：完整论文初稿可读

### Day 9（4月19日）
**目标：打磨投稿**
- [ ] 全文通读，统一符号
- [ ] 检查图编号、公式编号
- [ ] 精简 claim（不过度夸大）
- [ ] 格式检查（CGI 模板）
- [ ] 导出 supplementary 视频

**提交**：4月20日 CGI deadline 前提交

---

# Phase 2：PRCV 升级版（4月20日 → 4月30日，+10天）

## 在 CGI 版本基础上的升级

### 新增内容
1. **D-NeRF 数据集完整实验**（CGI 版来不及的部分）
2. **更多 ablation**（velocity vs displacement, FM step 数影响）
3. **定性分析加强**（更多可视化，更长序列）
4. **Related Work 更完整**（加 Flow4R, GaussianFlow 等 2025 新作）

### PRCV 定位
- 中文顶会，审稿标准相对宽松
- 可以用 CGI 稿件作为基础直接投
- 重点加强中文学术界关注的工程落地细节

---

# Phase 3：NeurIPS 版本（4月30日 → 5月中旬）

## 核心升级：Multi-scene FM Motion Prior

### 方法升级
```
Phase 1/2：每个场景独立训练 FM motion field（per-scene）

Phase 3：在多场景上联合训练 FM motion prior
         新场景用 prior 初始化，加速收敛 + 更好泛化
```

### 架构变化
- FM velocity network 增加 scene context embedding
- 在 D-NeRF 全部 8 scenes + Nvidia 全部 7 scenes 上联合预训练
- 新增泛化实验：held-out scene 上验证 prior 有效性

### NeurIPS 故事
> "我们提出 scene-agnostic FM motion prior，首次把 flow matching 从 per-scene 优化提升到跨场景 motion 表示学习"

### 新增实验
- Cross-scene generalization（held-out scene PSNR）
- Few-shot adaptation（用更少帧数收敛）
- Prior 可视化（motion field 插值）

---

# Future Work：大规模泛化（CVPR/ICCV 2026 方向）

## 目标
做出接近 Flow4R 规模的工作，实现真正的 generalizable dynamic scene understanding

## 核心思路（受 Flow4R 启发）
- 以 scene flow 作为中心表示，统一 geometry + motion
- 在大规模混合数据集（Kubric, Dynamic Replica, DAVIS, ...）上联合训练
- ViT backbone 替换 per-scene MLP
- 实现 feed-forward 推理（不需要 per-scene 优化）

## 与 Phase 3 的关系
Phase 3 的 multi-scene prior 是本阶段的技术铺垫
Phase 1/2/3 的实验积累是 Future Work 的 ablation 基础

## 所需资源
- 4~8 张 A800，连续训练 1~2 周
- 大规模数据 pipeline 建设
- 时间线：NeurIPS 投稿后开始，CVPR 2026 deadline（约2025年11月）

---

# 代码核心改动位置（Phase 1 参考）

```
SplineGS/
├── scene/
│   ├── gaussian_model.py      # 核心：inverse_cubic_hermite (line 54)
│   │                          #       control_point 参数
│   ├── motion_models.py       # 新建：SplineMotionModel / MLPMotionModel / FMMotionModel
│   └── deformation.py         # 辅助：pose_network（保留不动）
├── train.py                   # 改：加 FM loss，切换 motion model
└── utils/
    └── loss_utils.py          # 改：加 fm_loss, smoothness_loss 函数
```

## 关键接口定义

```python
class FMMotionModel(nn.Module):
    """
    Flow Matching motion field.
    输入: x0 (N,3) Gaussian centers, tau (scalar) 物理时间
    输出: delta_x (N,3) 位移
    训练时额外需要: t_flow (scalar) FM 内部时间
    """
    def forward(self, x0, tau):
        # ODE 积分: x_tau = x0 + integral v_theta dt
        ...

    def fm_loss(self, x_tau, x_tau_next):
        # 线性插值 + velocity matching
        t = torch.rand(1)
        x_t = (1 - t) * x_tau + t * x_tau_next
        v_target = x_tau_next - x_tau
        v_pred = self.velocity_net(x_t, t, tau)
        return F.mse_loss(v_pred, v_target)
```

---

# 每日固定节奏

- **上午**：改代码 / 调训练（最难的事）
- **下午**：跑实验 / 统计指标 / 整理图表
- **晚上**：写论文 / 记实验日志 / 规划明天

---

# 最危险的坑（不要踩）

1. Day 1-2 就开始大改 SplineGS 结构 → 先跑通原版
2. 没有 deterministic w/o FM baseline → 审稿人会问"FM 有什么用"
3. 实验跑完才开始写论文 → Day 7 必须开始写
4. 把 per-scene 结果吹成"通用 4D 生成" → claim 要克制
5. 拖到 Day 8 还在调参 → 结果够用就够了，先写论文

---

# 三阶段验收标准

## Phase 1（CGI，4月20日）
- [ ] SplineGS baseline 跑通，有数字
- [ ] FM motion field 实现，训练稳定
- [ ] 至少 3 个场景实验，至少 1 个场景 PSNR 不低于 SplineGS
- [ ] 论文提交

## Phase 2（PRCV，4月30日）
- [ ] D-NeRF 完整数字
- [ ] 更丰富的消融实验
- [ ] 论文提交

## Phase 3（NeurIPS，5月中旬）
- [ ] Multi-scene joint training 实现
- [ ] Cross-scene generalization 实验
- [ ] 故事从"替换 spline"升级为"motion prior learning"
- [ ] 论文提交
