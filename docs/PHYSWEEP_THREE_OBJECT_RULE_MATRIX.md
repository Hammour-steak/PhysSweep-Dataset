# PhysSweep 3obj 正式规则矩阵

状态：规则已用于完成正式批次 `three_object_production_20260909_r7`，1,538 条 base 与 18,456 条 sweep 已发布并通过对齐审计。配额保持为独立输入，详见 [当前代码与发布说明](PHYSWEEP_CODE_STATUS.md)。

正式入口只有两份配置：

- `configs/three_object_rule_matrix.json` 定义 family、运动、支撑、相机和兼容关系。
- `configs/three_object_asset_scope.json` 保存较长的资产 allowlist、碰撞代理类型和排除原因。

历史 D5/D6 配置与试产 scene ID 只作为验证证据，不参与正式 family 分发。后续配额文件只决定各项数量，不能扩大规则范围。

## 共同契约

- 三个动态对象固定为 `object_a`、`object_b`、`object_c`，运动角色为 P、Q、R。
- 每组 1 条 base 加 12 条 sweep；只干预 `object_a` 的摩擦、恢复系数或质量，每轴 4 个值。
- 物理初态、对象与角色、几何、支撑、环境、外观、caption 和相机在组内固定。
- 相机只在 base 上准入并冻结给整组；sweep 不按运动结果重新筛选。
- 请求视角只是求解偏好；配额按 base 准入后实际选中的冻结视角统计。若实际视角不足，只在同一物理 family/subcase 内补采，不改变物理 cell。
- 缺额只在同一规则 cell 中重采，不跨 family 或 subcase 调数。
- 正式样本只发布 `metadata.json`、`trajectory.npz`、`video.mp4`，不生成 mask。

## 四个 family

| family | 运动与场景 | 资产与外观 | 相机 |
|---|---|---|---|
| `generic` | 平面、浅斜面、真实静态 mesh 支撑；四种通用运动 | 84 个 1obj 通用视觉、10 个角色安全非球 mesh、1 个球形资产、18 个真实静态支撑 | 平面 25°/30°/35° 三向斜视；斜面 35°/40°/50° 三向斜视 |
| `billiards` | 三球连锁传递，禁止撞库和落袋 | 复用球桌、球材质槽排列及 HDRI | 复用 generic 平面三向相机 |
| `pinball` | 被动钉板 pair control，禁止主动挡板、发射器和活动门 | 三色排列与 3 个背景；单一已验收夹具 | 修正后前视 12°、左右视 8°；方位角 90°/82°/98°，完整夹具入画 |
| `marble_run` | 三段轨道 ordered contacts；有限初态表 | 三色排列与 3 个背景；单一已验收轨道 | 65° 的前、左、右高视角，只用于小配额轨道分支 |

## Generic 运动与形状兼容

| 模板 | 布局 | 允许形状 | 说明 |
|---|---|---|---|
| `chain_transfer` | 直线、左偏连锁 | P/Q/R 都为球 | 任意 mesh 连锁已从正式范围删除 |
| `successive_hits` | 直线、左右错位 | P/Q/R 都为球 | P 依次接触 Q、R |
| `converging_hits` | 直线汇入 | P/Q/R 都为球 | P、R 向 Q 汇入 |
| `pair_control` | 直线三球 | P/Q/R 都为球 | P–Q 接触，R 不接触另两者 |
| `pair_control` | 直线 mixed-R | P/Q 为球，R 为球、方块或圆柱 | 保留旧 mixed-R 覆盖 |
| `pair_control` | 直线 multi-mesh | P/Q/R 均可为球、方块或圆柱；最多 1 个球；每场最多 1 个真实非球资产 | 真实资产轮换 P/Q/R；完整物理与相机准入后保留 |
| `pair_control` | 左右交叉 | P/Q 为球，R 为球、方块或圆柱 | 不把未经验证的 P/Q 非球组合扩到交叉布局 |

平面运动覆盖 `world_x` 和 `world_y`，目标角色覆盖 P/Q/R，支撑覆盖 ground/raised。正式配额还需同时覆盖模板、布局、运动轴、形状组合、目标角色、支撑、相机、环境和视觉资产，不能只按物理通过率分配。

浅斜面只使用三球直线 `pair_control`，坡度 8–12°，支撑为 ground/raised shallow ramp。它有独立材料与相机规则，不继承平面的错位、交叉或 multi-mesh 布局。

真实静态 mesh 支撑只承载三球 `chain_transfer` 或 `pair_control`；18 个支撑位于资产能力表，rattan support 明确排除。

## 资产边界

资产能力表按碰撞代理分组：

- 84 个通用视觉：22 球、35 方块、27 圆柱；只要运动模板允许对应代理形状即可用于任意角色。
- 10 个真实非球 mesh：crate、box、remote、magazine、smartphone、can、plate、bowl、cup、bottle。它们只进入平面直线 multi-mesh `pair_control`，每场最多一个，并轮换 P/Q/R。
- rubber-band ball 作为球形资产，只按球体模板使用。
- gas cylinder、baluster vase、lug wrench、toy car、mug 因偏心或非轴对称 compound proxy 排除。
- 18 个真实桌面/支撑只作静态 exact proxy，不转成动态物体。

视觉 mesh 不替换碰撞代理，物理后不重新绑定外观。每个候选必须保留 1obj 来源、代理、尺度、质量与惯量绑定。

## 相机多样性与边界

通用平面不使用俯视图。三个相机族分别为：

- `front_oblique`：45° 方位、25° 仰角。
- `side_oblique`：90° 方位、30° 仰角。
- `rear_oblique`：135° 方位、35° 仰角。

物体投影短边占画面短边的目标为 10%，允许范围 5–25%。候选相机检查完整 base 轨迹与关键接触帧；准入后整组固定。

相机求解允许从请求视角回退到另一种合格视角，因此不能用请求标签冒充实际视觉覆盖。已验收的 27 个 multi-mesh base 中，请求分布为 front/side/rear = 10/7/10，最终冻结分布为 10/9/8，其中 10 个发生回退；正式配额必须读取最终冻结的 `selected_view_family`。

斜面提高到 35°/40°/50°，用于同时看清坡面和运动。弹珠台采用修正后的前视 12°、左右视 8° 专用视图。轨道弹珠的 65° 高视角是唯一高俯视范围，因为必须显示三个轨道段和接盘；它不能被 generic 继承，正式配额应保持很小。

## 明确不支持

当前不采：三者完全独立、接近同时三体碰撞、空中交互、右偏连锁、任意 multi-mesh 连锁/连续/汇入、混合形状斜面、偏心或非轴对称动态 compound、rattan mesh 支撑、台球撞库/落袋、主动弹珠台、未复核轨道变体、物理后视觉重绑定，以及按 sweep 结果换组。

## 验证与下一关

`tools/sampling/three_object_rule_matrix.py` 只验证结构和能力边界：四个 family、三种 generic subcase、运动与形状兼容、资产清单、特殊夹具容量、相机角度、组契约和无配额状态。它不再复制历史试验 cell 或 scene ID。

multi-mesh base 探针 42 条中 31 条通过物理，27 条继续通过相机并完成 1280×720、97 帧渲染；81 个物体实例中只有 16 个球，包含 65 个不同视觉 ID。随后选择一个由真实圆柱资产、球和方体组成的 pair-control base，完成 1 条 base 加 12 条 sweep 的正式尺寸烟测：13 条物理、相机审计和渲染全部通过，整组使用同一相机，生成 13 个视频且没有 mask。验收报告位于 `outputs/three_object_v6_simplified_pair_group_smoke_20260908_r1/v6_simplified_group_smoke_report.json`。
