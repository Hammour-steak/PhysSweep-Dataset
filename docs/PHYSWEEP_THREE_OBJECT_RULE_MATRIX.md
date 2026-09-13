# 3obj 规则与多样性

生成命令、数量和续跑见 [生成说明](GENERATION.md)，输出、身份和 sweep 共享契约见 [数据规范](PHYSWEEP_SPEC.md)。本文只补充 3obj 的能力边界。

[规则矩阵](../configs/three_object_rule_matrix.json) 声明运动、支撑和相机兼容关系，[资产范围](../configs/three_object_asset_scope.json) 声明允许资产、代理及排除原因。实际分发见 [three_object_generation.py](../tools/sampling/three_object_generation.py)：六个 generic 模式加三个特殊场景，发布为四个 family。部分 D5/D6 文件仍是运行时输入；顶层配置的历史状态文字不代表当前 CLI 分发方式。

三个对象为 `object_a`、`object_b`、`object_c`，运动角色为 P/Q/R；干预对象始终是 A，角色可轮换。当前入口按模式分配数量，每个选中模式至少一个 base；小批次不保证穷举资产、运动或视角。失败只在同一物理 cell 内重采，不跨 family/subcase 调数。

## 场景与运动

| 发布 family | 能力边界 |
|---|---|
| `generic` | 平面、浅斜面、真实静态 mesh 支撑；形状与运动兼容如下 |
| `billiards` | 三球连锁传递；禁止撞库和落袋 |
| `passive_pinball` | 被动钉板 pair control；固定夹具，禁止主动挡板、发射器和活动门 |
| `marble_run` | 三段轨道 ordered contacts；固定轨道和有限初态表 |

顶层规则中的 `pinball` 对应 CLI/输出的 `passive_pinball`。特殊场景复用原资产和 HDRI；钉板与轨道通过三色排列和背景选择变化外观。

| Generic 模板 | 布局及形状 |
|---|---|
| `chain_transfer` | 三球直线或左偏连锁，P→Q→R |
| `successive_hits` | 三球直线或左右错位，P 依次接触 Q、R |
| `converging_hits` | 三球直线汇入，P、R 向 Q 汇入 |
| `pair_control` | P–Q 接触，R 不接触另两者；直线允许三球或 mixed-R |
| `pair_control` 交叉 | 左右交叉；P/Q 为球，R 可为球、方块或圆柱 |
| `pair_control` multi-mesh | 仅平面直线；P/Q/R 可为球、方块或圆柱，最多一个球、一个真实非球资产，真实资产轮换 P/Q/R |

mixed-R 指 P/Q 为球、R 为球/方块/圆柱。平面覆盖 `world_x/world_y`、ground/raised 支撑及 P/Q/R 目标角色。浅斜面为 8–12° 的 ground/raised shallow ramp，仅允许三球直线 `pair_control`。真实 mesh 支撑仅允许三球 `chain_transfer` 或 `pair_control`。

## 资产范围

| 资产组 | 用途 |
|---|---|
| 84 个通用视觉：22 球、35 方块、27 圆柱 | 按模板允许的代理形状用于各角色 |
| 10 个真实非球 mesh：crate、box、remote、magazine、smartphone、can、plate、bowl、cup、bottle | 仅平面直线 multi-mesh，每场最多一个 |
| rubber-band ball | 球体模板 |
| 18 个真实桌面/支撑 | 静态 exact proxy；排除 rattan support |

视觉 mesh 不替换碰撞代理，物理后不重新绑定外观。候选保留 1obj 来源、代理、尺度、质量和惯量绑定。gas cylinder、baluster vase、lug wrench、toy car、mug 因偏心或非轴对称 compound proxy 排除。

## 相机与覆盖

| 场景 | 视角 |
|---|---|
| 通用平面、台球 | front/side/rear：方位 45°/90°/135°，仰角 25°/30°/35° |
| 浅斜面 | 三向斜视，仰角 35°/40°/50° |
| 被动弹珠台 | 前/左/右：方位 90°/82°/98°，仰角 12°/8°/8°，完整夹具入画 |
| 轨道弹珠 | 前/左/右 65° 高视角，用于看清轨道和接盘，不继承到 generic |

通用相机的物体投影短边占画面短边目标为 10%，允许 5–25%；检查完整 base 轨迹和关键接触帧。准入后整组固定，sweep 允许出界且不重新筛选。

请求视角允许回退到另一合格视角，因此覆盖统计读取冻结的 `selected_view_family`。公开入口不保证实际视角的精确配额；补足视角覆盖也应保持同一物理 cell。评估多样性应同时看运动、布局、形状、角色、支撑、环境、资产和实际相机，不能只看物理通过率。

## 未支持范围

三者完全独立、近同时三体碰撞、空中交互、右偏连锁、非球连锁/连续/汇入、混合形状斜面、偏心动态 compound、rattan 支撑、台球撞库/落袋、主动弹珠台及未复核轨道变体均不采样。禁止物理后重绑视觉或按 sweep 结果换组。

[规则验证器](../tools/sampling/three_object_rule_matrix.py) 检查结构和能力边界；生成入口执行 base 物理/相机准入及完整组发布检查。
