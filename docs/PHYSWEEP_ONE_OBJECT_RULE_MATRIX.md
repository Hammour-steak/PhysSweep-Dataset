# 1obj 规则与多样性

生成命令见 [生成说明](GENERATION.md)，共同约束见 [共享规则](PHYSWEEP_RULEBOOK.md) 和 [数据规范](PHYSWEEP_SPEC.md)。1obj 指一个动态对象；桌面、道具、墙壁和夹具均为静态环境。

[采样矩阵](../configs/one_object_sampling_matrix.json) 分别分配运动和环境，再匹配兼容组合；[通用采样规则](../configs/one_object_sampling_rules.json) 定义运动子类型、尺度、方向和相机轴，[依赖包](../configs/one_object_sampling_bundle.json) 绑定实现与资产。环境选择不改变请求的运动序列；有合法匹配时保留环境配额。

## 场景与运动

七个环境入口发布到五个 family：

| 发布 family | 环境与能力 |
|---|---|
| `generic` | ground/raised、flat/feature 支撑；按兼容关系覆盖全部通用运动 |
| `asset` | 真实支撑、带静态道具支撑、工作台；每个支撑只允许声明的 drop/push/edge-exit 等 profile |
| `billiards` | 单个标准球自由滚动或一次指定库边反弹 |
| `passive_pinball` | 单球被动下落穿过钉阵，进入接球区 |
| `marble_run` | 单球早/晚释放，沿声明的轨道链运动 |

通用矩阵包含十一种运动意图；每种还有速度、方向、初始位置或运动幅度等子类型：

| 意图 | 含义 |
|---|---|
| `slide_push_1obj` | 支撑面推滑 |
| `roll_or_slide_1obj` | 滚动或滚滑 |
| `wall_impact_1obj` | 撞击指定墙面 |
| `edge_fall_1obj` | 越过支撑边缘后下落 |
| `drop_fall_1obj` | 下落并接触支撑 |
| `projectile_1obj` | 水平抛出 |
| `arc_projectile_1obj` | 带上升段的抛物运动 |
| `slope_slide_down_1obj` | 沿斜面向下 |
| `slope_slide_up_1obj` | 沿斜面向上及声明的后续运动 |
| `ramp_to_flat_1obj` | 从斜面过渡到平面 |
| `bounce_1obj` | 可观察反弹 |

上述意图不是任意资产与场景的笛卡尔积。工作台只开放已验证的清空区域下落和长轴推动；带静态道具场景只开放声明的直线安全通道，不允许意外撞道具。具体配对由 [兼容关系](../configs/compatibility.json) 和矩阵中的 support/profile/dynamic pool 决定。

## 资产与外观

通用分支使用 84 个已准入视觉对象，按球、方块、圆柱等代理和姿态能力匹配运动；真实资产分支使用注册表中已准入的动态 compound 代理与静态支撑。视觉 mesh 不替换碰撞代理，真实支撑面以测量和代理为准。

环境、支撑和对象分别选择兼容的材质与纹理尺度，搭配 HDRI、程序化房间或已准入 mesh 场景。资产边界见 [对象配置](../configs/physassets_core_object_profiles.json)、[代理注册表](../configs/asset_proxy_registry.json) 和 [场景语义](../configs/asset_semantic_scene_rules.json)；外观选择见 [visual_sampling.json](../configs/visual_sampling.json)。

## 相机与覆盖

通用矩阵声明前左、前右、左、右、后方和高斜视六个请求视角。求解器根据运动观察窗口和结构锚点选取合格姿态；请求标签不保证成为最终视角，也不能强迫不兼容场景使用高视角。

观察意图与结构上下文放在 `camera_request.observation`，不混入物理准入的 `expected_motion`。相机检查物体占比、初始可见性、运动覆盖和遮挡；关键过程后是否允许有限出界由该运动规则决定。特殊场景使用自己的视角池或夹具相机。准入后的 base 相机沿用于 sweep。

评估覆盖应同时看运动子类型、方向、尺度、支撑、资产、外观和实际视角；小批次不保证全部覆盖。

## 未支持范围

未声明的支撑/运动/资产组合、道具意外交互、动态三角网格碰撞，以及仿真后放大物体、修改初态或补加驱动力均不允许。单物体在均匀重力和库仑接触下，单独改变质量通常不会改变轨迹；不能人为制造质量效应。

入口实现：[generate_one_object_dataset.py](../tools/cli/generate_one_object_dataset.py)。
