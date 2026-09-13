# 2obj 规则与多样性

生成命令见 [生成说明](GENERATION.md)，共同约束见 [共享规则](PHYSWEEP_RULEBOOK.md) 和 [数据规范](PHYSWEEP_SPEC.md)。动态对象为 `object_a`、`object_b`；sweep 只干预 A，保持双方初态和 base 相机。

[采样矩阵](../configs/two_object_sampling_matrix.json) 定义运动、形状配对、尺度和视角，[场景规则](../configs/two_object_scene_rules.json) 限定宿主环境，[特殊场景规则](../configs/two_object_specialized_scene_rules.json) 定义三个夹具分支。对象和宿主来自准备好的 1obj 来源池，无需先生成自己的 1obj 批次。

## 场景与运动

| 发布 family | 能力边界 |
|---|---|
| `generic` | 平面和已声明斜面上的交互，以及平面上的空中交互/独立对照 |
| `billiards` | 两球正撞、擦撞、对向撞；球间首碰前禁止撞库 |
| `passive_pinball` | 两球顶部撞击、错位撞击、斜向追碰后通过被动钉阵 |
| `marble_run` | 两球追碰、延迟追碰、逆向碰撞后通过声明的轨道段 |

Generic 有十二种意图：

| 意图 | 含义 |
|---|---|
| `surface_hit_rest_2obj` | A 撞静止 B |
| `surface_glancing_hit_rest_2obj` | A 擦撞静止 B |
| `surface_head_on_2obj` | 对向正撞 |
| `surface_glancing_opposed_2obj` | 对向擦撞 |
| `surface_crossing_2obj` | 交叉相遇 |
| `surface_catch_up_2obj` | 同向追碰 |
| `air_drop_hit_supported_2obj` | 下落 A 撞支撑面上的 B |
| `air_projectile_hit_supported_2obj` | 抛射 A 撞支撑面上的 B |
| `air_air_collision_2obj` | 两个空中对象碰撞 |
| `surface_single_independent_2obj` | 一动一静，不互相接触 |
| `surface_dual_independent_2obj` | 两者运动但互不接触 |
| `air_supported_independent_2obj` | 空中 A 与支撑面上的 B 保持独立 |

碰撞后的具体结果不预先指定；base 要满足请求的接触/独立关系和数值完整性。Generic 覆盖配置要求交互占比至少 80%，同时保留独立对照。

## 资产与场景兼容

对象可来自 generic 或 asset 来源，并按有序 A/B 角色组合。支持球、方块、圆柱及居中、直立轴对称的已准入 compound 代理；偏心和不满足姿态约束的资产不能直接加入。

- 普通表面运动允许矩阵声明的形状配对；正向对撞要求 B 为球。
- 下落/抛射撞支撑对象要求空中 A 为球；空中双体碰撞仅允许球–球。
- 独立对照使用同形状族配对，不能据此放开任意空中混合形状碰撞。
- Generic 宿主限于允许的程序化环境及支撑 collider；斜面只开放声明的表面交互，另受运动、尺度和相机兼容表限制。
- 台球、弹珠台和轨道分支均使用球与对应的固定夹具，不替换为任意 mesh。

Generic 环境覆盖 minimal、home/office、lab/studio、garage/workshop、outdoor/courtyard 等类别；宿主与对象选择分别平衡，外观保留来源绑定。特殊场景通过自己的背景、HDRI 和视角 profile 提供变化。资产数量取决于来源池的实际准入结果，不能把 1obj 的全部资产视为均可配对。

## 相机与覆盖

Generic 声明左右中位侧视、左右低位前视、左右高位后视六个视角族，并提供有界回退。斜面仅采用场景规则允许的子集。特殊场景各有左、右、高位视角，并要求夹具和双物体运动可读。

相机联合检查双方的初始包围盒、完整运动覆盖、关键接触帧、物体占比、投影分离和遮挡；不能只围绕 A 构图。当前 generic base 的中心轨迹要求完整入画，同时允许配置范围内的包围盒裁切。该 base 准入要求不延伸为 sweep 的运动筛选规则。

配额和复用预算见 [采样请求](../configs/two_object_production_sampling.json)。公开入口会按 `--count` 重新分配数量；该文件的历史总量不是每次运行的默认总量。覆盖按运动、形状、尺度、来源组合、宿主、外观及实际视角共同评估，超过有限容量的请求会报错。

## 未支持范围

未声明的形状/运动/宿主组合、任意偏心 compound、主动夹具、仿真后改初态或强制指定碰撞结果均不允许。独立对照可以没有球间碰撞，但不能绕过该意图的准入检查。

入口实现：[generate_two_object_dataset.py](../tools/cli/generate_two_object_dataset.py)。
