# multi_agent_cooperative

基于 SUMO 的双路口交通信号 **多智能体协同强化学习** 实验平台。每个路口由独立智能体控制，支持图注意力融合邻居路口状态。

## 环境要求

1. **Python** 3.9+
2. **SUMO**（需设置环境变量 `SUMO_HOME`）
   - Windows: 从 [Eclipse SUMO](https://eclipse.dev/sumo/) 安装，并将 `SUMO_HOME` 指向安装目录
   - 验证: `%SUMO_HOME%\bin\sumo.exe --version`
3. Python 依赖:

```bash
pip install -r requirements.txt
```

## 项目结构

```
environment/          # SumoEnv、TrafficSignal（支持多路口）
agents/               # DQN、REINFORCE、A3C、RPDQN、GC-MAPPO 等
nets/                 # SUMO 双路口路网
main.py               # 训练入口
test_algorithms.py    # 多算法对比 benchmark
test_multi_agent_interaction_real.py  # 协同/独立/图注意力消融
```

## 快速开始

### 单智能体训练（仅控制第一个路口 t1）

```bash
python main.py --network=dqn --num_episodes=5 --use_gui=False
```

可选算法: `dqn`, `double_dqn`, `dueling_ddqn`, `pnd_ddqn`

### 算法对比实验

```bash
python test_algorithms.py --episodes 3 --algorithms "RPDQN" "GC-MAPPO" "Double DQN"
```

### 多智能体交互消融测试

```bash
python test_multi_agent_interaction_real.py
```

## 核心改进说明

| 模块 | 说明 |RPGC-PPO""Dubl
|------|------|
| `SumoEnv(multi_agent=True)` | `step(actions)` 同时控制所有信号灯，返回状态/奖励列表 |
| `TrafficSignal._build_phase_lane_map()` | 奖励函数按相位-车道动态映射，无硬编码车道名 |
| `build_intersection_graph()` | 根据路口坐标自动生成邻接矩阵与距离矩阵 |

## 仿真场景

- 路网: `nets/2way-double-intersection/`
- 信号灯: `t1`, `t2`（双路口）
- 状态: 各进口车道车辆密度
- 动作: 绿灯相位索引

## 许可证

参见仓库根目录说明。
