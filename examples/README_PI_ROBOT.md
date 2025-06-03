# Pi Robot 双足步态演示方案

这个方案帮助你使用Pi机器人的URDF模型运行Crocoddyl双足步态演示。

## 📋 准备工作

### 1. 确保已安装依赖
```bash
# 激活虚拟环境
source crocoddyl_env/bin/activate

# 确保已安装所需包
pip install crocoddyl example-robot-data matplotlib meshcat
```

### 2. 准备Pi机器人URDF文件
确保你有Pi机器人的URDF文件，并知道文件路径。

## 🔧 使用步骤

### 步骤1: 分析你的机器人URDF
首先运行分析脚本来了解你的机器人结构：

```bash
python analyze_pi_robot.py /path/to/your/pi_robot.urdf
```

这个脚本会显示：
- 机器人的基本信息（关节数、自由度等）
- 所有关节和框架的名称
- 建议的脚部框架名称
- 推荐的配置参数

### 步骤2: 更新配置
根据分析结果，编辑 `pi_bipedal_gaits.py` 文件：

```python
# 1. 更新URDF路径
urdf_path = "/path/to/your/pi_robot.urdf"

# 2. 更新脚部框架名称
rightFoot = "your_right_foot_frame"  # 从分析结果中获取
leftFoot = "your_left_foot_frame"    # 从分析结果中获取

# 3. 如需要，自定义半坐姿态
def get_half_sitting_configuration(model):
    q0 = pinocchio.neutral(model)
    # 根据你的机器人调整关节角度
    # 例如: q0[hip_joint_index] = -0.5  # 髋关节弯曲
    # 例如: q0[knee_joint_index] = 1.0  # 膝关节弯曲
    return q0
```

### 步骤3: 运行演示
```bash
# 仅运行优化（无可视化）
python pi_bipedal_gaits.py

# 运行带图形显示的演示
python pi_bipedal_gaits.py display

# 运行带图形显示和绘图的完整演示
python pi_bipedal_gaits.py display plot
```

## 🎯 演示内容

演示包含三个阶段：
1. **行走阶段1**: 机器人执行稳定的双足行走
2. **跳跃阶段**: 机器人执行向前跳跃动作
3. **行走阶段2**: 机器人继续行走

## ⚙️ 参数调整

根据你的Pi机器人，你可能需要调整以下参数：

```python
GAITPHASES = [
    {
        "walking": {
            "stepLength": 0.2,    # 步长（米）
            "stepHeight": 0.03,   # 步高（米）
            "timeStep": 0.05,     # 时间步长
            "stepKnots": 10,      # 步态节点数
            "supportKnots": 3,    # 支撑节点数
        }
    },
    # ... 其他阶段
]
```

### 建议的调整策略：
- **小型机器人**: 减小stepLength和stepHeight
- **计算能力有限**: 减少stepKnots和supportKnots
- **稳定性问题**: 增加supportKnots，减小stepLength

## 🖥️ 可视化

### Meshcat 可视化（推荐）
- 自动启动Web界面
- 在浏览器中访问显示的URL（通常是 `http://127.0.0.1:7003/static/`）
- 支持鼠标交互（旋转、缩放、平移）

### Gepetto 可视化
- 需要额外安装Gepetto viewer
- 提供更高级的可视化功能

## 🐛 常见问题

### 1. URDF加载失败
- 检查URDF文件路径是否正确
- 确保URDF文件格式正确
- 检查依赖的mesh文件是否存在

### 2. 脚部框架未找到
- 运行 `analyze_pi_robot.py` 查看所有可用框架
- 更新 `rightFoot` 和 `leftFoot` 变量
- 确保框架名称拼写正确

### 3. 优化不收敛
- 减小步长和步高参数
- 增加迭代次数
- 调整初始姿态配置
- 检查机器人的关节限制

### 4. 可视化问题
- 确保meshcat已正确安装
- 检查防火墙设置
- 尝试不同的浏览器

## 📊 性能调优

对于不同大小的机器人，建议的参数组合：

### 小型机器人 (< 0.5m)
```python
"stepLength": 0.1,
"stepHeight": 0.02,
"timeStep": 0.03,
"stepKnots": 8,
```

### 中型机器人 (0.5-1.0m)
```python
"stepLength": 0.2,
"stepHeight": 0.05,
"timeStep": 0.05,
"stepKnots": 10,
```

### 大型机器人 (> 1.0m)
```python
"stepLength": 0.3,
"stepHeight": 0.08,
"timeStep": 0.05,
"stepKnots": 12,
```

## 🚀 扩展功能

你可以基于这个基础方案进一步扩展：

1. **添加更多步态**: 修改 `GAITPHASES` 添加新的运动模式
2. **自定义成本函数**: 修改SimpleBipedGaitProblem类
3. **添加约束**: 使用Box-FDDP处理关节限制
4. **实时控制**: 将轨迹发送到实际机器人

## 📞 支持

如果遇到问题：
1. 首先运行 `analyze_pi_robot.py` 检查机器人模型
2. 检查终端输出的错误信息
3. 尝试简化参数设置
4. 参考Crocoddyl官方文档和示例

祝你使用愉快！🤖✨ 