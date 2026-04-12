# 基于 SplineGS 的 10 天 CGI 冲刺计划

## 项目定位

### 暂定题目

**Flow-Matched Motion Fields for Dynamic Gaussian Splatting**

### 一句话目标

以 **SplineGS** 为起点，保留其成熟的 dynamic Gaussian rendering / training pipeline，把其中的**确定性 spline trajectory motion module**替换为**基于 Flow Matching 的连续时间 motion field**，验证该表示在 temporal consistency 和 motion expressiveness 上的优势。

### 核心故事

- **SplineGS**：学习的是固定的、确定性的样条运动轨迹
- **我们的方法**：学习的是一个连续时间的 motion transport process
- **关键区别**：不是拟合一条固定曲线，而是学习一个 velocity field 来驱动动态演化

### 本稿件主 claim

将 dynamic Gaussian motion 从 **spline trajectory fitting** 重写为 **flow-matched motion field learning**

用连续时间 velocity field 建模动态演化，而不是固定轨迹基函数

在 temporal consistency 上优于 deterministic trajectory/motion baseline

不强 claim 通用 4D generation，只聚焦 dynamic Gaussian motion modeling

------

## 本次投稿必须遵守的原则

### 做

复用 SplineGS 的训练、渲染、评测骨架

只替换 motion/trajectory 模块

优先保证方法闭环

优先保证 baseline、ablation、图表齐全

主打 temporal consistency 和 motion modeling formulation

### 不做

不做跨场景 foundation model

不做 text-to-4D

不做复杂大 transformer

不做超大规模实验

不做过强“多样生成”claim

不做与当前工程底座无关的大改

------

## 技术最小闭环

### 表示

使用 SplineGS 现有 dynamic Gaussian splatting pipeline

保留 Gaussian 表示和 renderer

替换其 spline trajectory module 为 FM motion field module

### 模型

目标学习一个 motion field：

$$v_\theta(x, t, \tau) $$

其中：

- $x$：Gaussian center / canonical coordinate / point state
- $t$：flow matching 内部时间
- $\tau$：物理时间

输出：

velocity

或 displacement（看实现稳定性选择）

### 必须包含的损失

render reconstruction loss

flow matching loss

temporal smoothness regularization

transport consistency loss

### 必须保留的实验组

SplineGS 原版

Ours w/o FM（deterministic MLP motion）

Ours full（FM motion field）

### 数据

主数据集：D-NeRF

可选补充：1~2 个定性场景

不额外采数据

------

# Day 1 — 跑通 SplineGS 原版，吃透代码结构

## 今日目标

确保 SplineGS 原仓库能训练、渲染，并搞清 motion trajectory 的代码入口。

## 必做

安装环境

跑通官方训练脚本

跑通 1 个最小场景

导出结果视频

找到数据加载入口

找到 Gaussian 表示定义

找到 spline trajectory / motion module

找到 rendering 调用链

找到 loss 定义位置

记一页代码结构笔记

## 今日产出

原版训练日志

原版结果视频

代码模块关系图/笔记

确认后续最该改的文件列表

## 晚上检查点

是否已经能完整训练并渲染一个场景

是否明确知道 Gaussian 在时间 $\tau$ 的位置是在哪里计算出来的

是否确认 spline motion 部分可替换

## 如果失败

换更小场景

先只跑 inference / eval

不解决创新，先解决复现

------

# Day 2 — 把 motion 模块抽象成统一接口

## 今日目标

把 SplineGS 的 trajectory module 抽象成一个可插拔接口，为后面接 deterministic / FM 两个版本做准备。

## 必做

把原 spline motion 封装成统一接口

定义类似 `motion_model(x0, tau) -> x_tau`

或定义 `velocity_model(x_t, t, tau) -> v`

新建并行模块结构：

```
SplineMotionModel
DeterministicMotionModel
FMMotionModel
```

保证原版 spline 还能正常训练

## 今日产出

motion API 统一

原版 SplineGS 不受影响

后续替换模块只改局部代码

## 晚上检查点

baseline 是否仍能跑

motion 模块是否已经独立出来

是否能在不动 renderer 的情况下替换 motion 表示

## 如果失败

少做重构，只保最小替换点

不追求代码优雅，先保证能插模块

------

# Day 3 — 先做一个 deterministic 非 FM 版本

## 今日目标

先实现一个不用 spline、也不用 FM 的普通 motion predictor，作为强 baseline 和中间过渡版本。

## 推荐形式

$\Delta x = f_\theta(x,\tau)$

或 $v_\theta(x,\tau)$

## 必做

写一个简单 MLP motion module

接入现有训练流程

保留 render loss

加入基础 smoothness regularization

跑 1 个场景

渲染短视频

## 今日产出

deterministic motion version

对比原版 spline 的初步结果

训练是否稳定的结论

## 晚上检查点

是否已经证明“不用 spline 也能动起来”

是否能作为 w/o FM baseline

motion 模块替换链路是否走通

## 如果失败

先预测 displacement，不做 velocity rollout

减少输入，只用 $x+\tau$

降低模型复杂度

------

# Day 4 — 接入 FM motion field

## 今日目标

实现论文核心创新：用 Flow Matching 学 motion field。

## 推荐最小 formulation

构造相邻状态对 $(X_\tau, X_{\tau+\Delta\tau})$

做线性插值：

$$X_t = (1-t)X_\tau + tX_{\tau+\Delta\tau} $$

target velocity:

$$v^* = X_{\tau+\Delta\tau} - X_\tau $$

FM loss:

$$\mathcal{L}_{FM} = \|v_\theta(X_t, t, \tau) - v^*\|^2 $$

## 必做

在 deterministic 版本基础上加入 flow time $t$

写 FM loss

接入训练

跑 1 个场景

保存 loss 曲线

导出短序列

## 今日产出

第一个能训练的 FM 版本

deterministic vs FM 初版对比

方法公式草稿

## 晚上检查点

FM loss 是否正常下降

是否没有明显比 deterministic 更炸

是否至少能渲染出合理动态结果

## 如果失败

简化输入变量

缩小训练范围

暂时不做 fancy sampling

先保最小 conditional FM 版本

------

# Day 5 — 加 transport consistency，稳定 full model

## 今日目标

让 FM motion 不只是“会动”，而是“动得更合理”。

## 必做

加 transport consistency loss

加 temporal smoothness

调 loss 权重

调学习率和训练策略

在 1~2 个场景上验证稳定性

保存更长一点的序列可视化

## 推荐关注现象

抖动是否减少

连续帧是否更自然

是否比 spline trajectory 更少 abrupt motion artifacts

## 今日产出

full model 初版

spline / deterministic / FM 三组初步对比

可视化结果备份

## 晚上检查点

是否已经出现至少一个明显优点

是否值得扩大到 2~3 个主场景

是否确定最终实验方向

## 如果失败

只保 FM + render loss + 一个正则

不追求长时 rollout

先保短序列时序一致性

------

# Day 6 — 跑主实验场景

## 今日目标

选 2~3 个最适合出图的场景，形成论文主结果。

## 选择原则

动作中等复杂

原版 SplineGS 本身较稳定

temporal artifact 容易观察

渲染结果视觉上清楚

## 必做

跑 SplineGS 原版

跑 Ours w/o FM

跑 Ours full

统计 PSNR / SSIM / LPIPS

准备 temporal consistency 定性图

保存全部 checkpoint

## 今日产出

主实验表格第一版

连续帧可视化第一版

对比视频素材

## 晚上检查点

是否已经有论文级主结果

是否至少在一个关键维度优于 baseline

是否已经有 1~2 张可以进正文的图

## 如果失败

减少场景数

优先保最漂亮的结果

不强求所有指标全赢

------

# Day 7 — 做最小 ablation

## 今日目标

证明你不是“只换了个模块名字”。

## 必做 ablation

w/o FM

w/o transport consistency

full model

## 可选 ablation

spline trajectory vs deterministic motion vs FM motion

velocity vs displacement parameterization

不同 regularization 权重

## 最该打的点

temporal evolution 更平滑

更少突变

不依赖固定 spline basis

motion field 表达更自然

## 今日产出

ablation 表格

ablation 图

对应结论草稿

## 晚上检查点

是否能回答“为什么一定要 FM”

是否能回答“为什么要 transport consistency”

是否已经足够支撑 method claim

## 如果失败

砍掉次要 ablation

只保最能说明问题的 2~3 项

让实验服务故事，而不是堆实验

------

# Day 8 — 开始写论文，图先定稿

## 今日目标

论文先写完整，不等实验彻底完美。

## 必做 figure

方法总览图

spline / ours w/o FM / ours 连续帧对比图

ablation 图

轨迹/运动表示对比图

## 写作顺序

Method

Experiments

Introduction

Related Work

Conclusion

Abstract

## Introduction 要写清楚

现有 dynamic Gaussian motion 多依赖显式 trajectory fitting

spline parameterization 强，但本质上是 deterministic motion curve

我们把 dynamic motion 看成 transport process

用 flow matching 学连续时间 motion field

## 今日产出

论文主体初稿

图表全部插入

标题/摘要/引言初版

## 晚上检查点

论文是否已经“能提交”

是否没有过度 claim

是否已经和 SplineGS 的区别讲清楚

## 如果失败

先写短版完整稿

related work 最后补

不要为了润色耽误成稿

------

# Day 9 — 修漏洞，收窄 claim

## 今日目标

把故事收紧，避免被审稿人一句话打死。

## 主 claim 最终版建议

我们不是完整通用 4D generative model

我们关注的是 dynamic Gaussian splatting 下的 motion reformulation

从 spline trajectory fitting 改为 FM-based motion transport

核心收益是 temporal consistency 和 motion representation flexibility

## 必做

补最缺的一项实验

重写摘要

精简引言

统一符号

写 limitations

写 conclusion

整理 supplementary 内容

## limitation 建议

当前主要验证 per-scene dynamic setting

尚未覆盖跨场景 motion prior

多样 motion sampling 仍需更大规模训练来充分验证

## 今日产出

投稿版 PDF v1

摘要终稿

limitations 和 conclusion 完成

## 晚上检查点

是否所有实验都围绕同一个主问题

是否避免把 FM 夸成“万能生成器”

是否已经准备好投稿版本

## 如果失败

删去边缘内容

只保一条最强主线

让全文围绕“FM motion vs spline trajectory”展开

------

# Day 10 — 最终检查与投稿

## 今日目标

不再改模型，只做提交准备。

## 必做

全文通读 2 遍

检查图编号/表编号

检查公式符号一致

检查所有实验数字是否准确

检查方法和实验设置是否一致

导出 supplementary 视频

整理提交材料

检查标题/摘要是否克制

## 最终核对清单

标题不夸大

摘要不夸大

claim 与实验一一对应

至少一个强 baseline

至少两个关键 ablation

至少两组清晰 qualitative

limitations 主动承认范围

主图一眼能看出你和 SplineGS 的区别

## 投稿前绝对不要做的事

不临时大改 motion module

不换数据集

不新增大实验

不临时改题成“通用 4D 生成”

------

## 每天固定节奏建议

### 上午

处理最核心、最难的问题

改代码 / 调训练

### 下午

跑实验

统计指标

整理图和视频

### 晚上

写论文

记实验日志

规划第二天唯一主目标

------

## 本稿件最重要的三个验收标准

### 1. 方法上是否有明确新意

能一句话说清：我们把 spline trajectory fitting 改成了 FM motion field learning

### 2. 实验上是否有至少一个硬结果

temporal consistency 更好

运动更自然

更少 abrupt motion artifact

至少在一个关键维度优于 deterministic baseline

### 3. 叙事上是否克制

不说“解决 4D generation”

只说“提出一种 dynamic Gaussian motion 的新 formulation”

------

## 最危险的坑

一上来就重构整个 SplineGS

不先保留 spline baseline

没有 deterministic w/o FM baseline

实验做完才开始写论文

把 per-scene 结果吹成 generative world model

没有清楚说明和 SplineGS 的本质差异

图不够直观，审稿人看不出 improvement

------

## 最终建议

这 10 天的目标不是“做出终极版 4D 模型”，而是：

做出一个小而完整的 motion reformulation paper

以 SplineGS 为强 baseline 和稳定底座

用 FM motion field 替换 deterministic spline trajectory

把论文讲成“从 trajectory fitting 到 motion transport”的转变

如果必须继续收缩，优先级如下：

保 full model

保 SplineGS baseline

保 deterministic w/o FM baseline

保 2~3 个主场景

保关键 ablation

保方法图和连续帧对比图

其他都可以砍。

[无标题](https://docs.qq.com/aio/DQ0FTV0FTU0RzeFpV?p=XTVZDmxdI7y0qBMVtLvzFQ)