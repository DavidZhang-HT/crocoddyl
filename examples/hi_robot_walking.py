import os
import signal
import sys
import time

import numpy as np
import pinocchio
from pinocchio.robot_wrapper import RobotWrapper

import crocoddyl
from crocoddyl.utils.biped import SimpleBipedGaitProblem, plotSolution

WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)

print("🚶 Hi Robot 步行程序")
print("=" * 50)

# 设置正确的路径
script_dir = os.path.dirname(os.path.abspath(__file__))  # examples目录
base_dir = os.path.dirname(script_dir)  # crocoddyl目录
root_dir = os.path.dirname(base_dir)    # motionControl目录

# 使用修复后的URDF文件加载Hi机器人
urdf_path = os.path.join(root_dir, "crocoddyl", "hi_robot", "urdf", "hi_23dof_250401_rl_fixed.urdf")
mesh_dir = os.path.join(root_dir, "crocoddyl", "hi_robot")

print(f"📂 加载Hi机器人从: {urdf_path}")

try:
    print("🔄 加载完整Hi机器人模型...")
    # 使用buildModelsFromUrdf加载完整模型
    model, collision_model, visual_model = pinocchio.buildModelsFromUrdf(
        urdf_path, 
        mesh_dir,
        pinocchio.JointModelFreeFlyer()
    )
    
    # 创建机器人包装器
    hi_robot = RobotWrapper(model)
    hi_robot.visual_model = visual_model
    hi_robot.collision_model = collision_model
    hi_robot.visual_data = visual_model.createData()
    hi_robot.collision_data = collision_model.createData()
    
    print(f"✅ 成功加载Hi机器人: {hi_robot.model.name}")
    print(f"   🦴 自由度数量: {hi_robot.model.nv}")
    print(f"   🔗 关节数量: {hi_robot.model.njoints}")
    print(f"   🎨 Visual几何体数量: {len(hi_robot.visual_model.geometryObjects)}")
    
except Exception as e:
    print(f"❌ 完整模型加载失败: {e}")
    print("🔄 尝试加载简化模型...")
    try:
        # 回退到只加载运动学模型
        model = pinocchio.buildModelFromUrdf(urdf_path, pinocchio.JointModelFreeFlyer())
        hi_robot = RobotWrapper(model)
        hi_robot.visual_model = pinocchio.GeometryModel()
        hi_robot.collision_model = pinocchio.GeometryModel()
        hi_robot.visual_data = hi_robot.visual_model.createData()
        hi_robot.collision_data = hi_robot.collision_model.createData()
        print(f"✅ 使用简化模型: {hi_robot.model.name}")
    except Exception as e2:
        print(f"❌ 简化模型也失败: {e2}")
        sys.exit(1)

# 获取模型
rmodel = hi_robot.model
rdata = rmodel.createData()

# 设置力矩限制以获得更自然的运动
if hasattr(rmodel, 'effortLimit') and len(rmodel.effortLimit) > 0:
    lims = rmodel.effortLimit.copy()
    lims[7:] *= 0.6  # 降低关节力矩限制，使运动更平滑
    rmodel.effortLimit = lims

# 定义初始状态 - 自然站立姿态
q0 = pinocchio.neutral(rmodel)
q0[2] = 0.65  # 设置基座高度

# 设置自然站立姿态
try:
    # 获取关键关节并设置合理的初始角度
    joint_config = {
        "l_hip_pitch_joint": -10,  # 髋关节轻微前倾
        "r_hip_pitch_joint": -10,
        "l_calf_joint": 20,        # 膝关节轻微弯曲
        "r_calf_joint": 20,
        "l_ankle_pitch_joint": -10, # 踝关节补偿
        "r_ankle_pitch_joint": -10,
        "l_shoulder_pitch_joint": 30,  # 手臂自然下垂
        "r_shoulder_pitch_joint": 30,
    }
    
    for joint_name, angle_deg in joint_config.items():
        try:
            if joint_name in [rmodel.names[i] for i in range(rmodel.njoints)]:
                joint_id = rmodel.getJointId(joint_name)
                q0[rmodel.joints[joint_id].idx_q] = np.radians(angle_deg)
        except:
            continue
    
    print("✅ 设置自然站立姿态")
    
except Exception as e:
    print(f"⚠️  使用默认neutral配置: {e}")

# 创建初始状态
v0 = np.zeros(rmodel.nv)
x0 = np.concatenate([q0, v0])

# 添加"half_sitting"参考配置，供SimpleBipedGaitProblem使用
if not hasattr(rmodel, 'referenceConfigurations'):
    import collections
    rmodel.referenceConfigurations = collections.OrderedDict()
rmodel.referenceConfigurations["half_sitting"] = q0.copy()

print(f"   ⚙️  状态维度: {len(x0)} (nq={len(q0)}, nv={len(v0)})")

# 定义足部frame名称
rightFoot = "r_ankle_roll_link"  # Hi机器人的右足
leftFoot = "l_ankle_roll_link"   # Hi机器人的左足

# 验证frame是否存在
try:
    rightFootId = rmodel.getFrameId(rightFoot)
    leftFootId = rmodel.getFrameId(leftFoot)
    print(f"🔗 足部Frame ID: 右足={rightFootId}, 左足={leftFootId}")
except Exception as e:
    print(f"❌ 获取足部Frame ID失败: {e}")
    print("🔍 可用的frames:")
    for i, frame in enumerate(rmodel.frames):
        if "ankle" in frame.name.lower() or "foot" in frame.name.lower():
            print(f"   {i}: {frame.name}")
    sys.exit(1)

# 创建双足步态问题
print("🚶 创建双足步态问题...")
gait = SimpleBipedGaitProblem(rmodel, rightFoot, leftFoot)

# 设置步态参数 - 针对Hi机器人优化
GAITPHASES = [
    {
        "walking": {
            "stepLength": 0.15,    # 较小的步长，适合Hi机器人
            "stepHeight": 0.05,    # 较低的抬腿高度
            "timeStep": 0.04,      # 较大的时间步长，更稳定
            "stepKnots": 25,       # 减少步数以加快计算
            "supportKnots": 8,     # 减少支撑期步数
        }
    },
    {
        "walking": {
            "stepLength": 0.20,    # 逐渐增加步长
            "stepHeight": 0.06,
            "timeStep": 0.04,
            "stepKnots": 25,
            "supportKnots": 8,
        }
    },
    {
        "walking": {
            "stepLength": 0.20,    # 保持稳定步长
            "stepHeight": 0.06,
            "timeStep": 0.04,
            "stepKnots": 25,
            "supportKnots": 8,
        }
    },
    {
        "walking": {
            "stepLength": 0.15,    # 最后一步减小步长
            "stepHeight": 0.05,
            "timeStep": 0.04,
            "stepKnots": 25,
            "supportKnots": 8,
        }
    },
]

print(f"📋 步态参数:")
for i, phase in enumerate(GAITPHASES):
    params = phase["walking"]
    print(f"   步骤 {i+1}: 步长={params['stepLength']:.2f}m, 抬腿={params['stepHeight']:.2f}m")

# 求解每个步态阶段
solver = [None] * len(GAITPHASES)
for i, phase in enumerate(GAITPHASES):
    for key, value in phase.items():
        if key == "walking":
            print(f"\n🚀 求解第 {i+1} 步: {key}...")
            
            # 创建步行问题
            solver[i] = crocoddyl.SolverBoxFDDP(
                gait.createWalkingProblem(
                    x0,
                    value["stepLength"],
                    value["stepHeight"],
                    value["timeStep"],
                    value["stepKnots"],
                    value["supportKnots"],
                )
            )
            solver[i].th_stop = 1e-6  # 适中的收敛阈值

    # 设置回调函数
    if WITHPLOT:
        solver[i].setCallbacks([
            crocoddyl.CallbackVerbose(),
            crocoddyl.CallbackLogger(),
        ])
    else:
        solver[i].setCallbacks([crocoddyl.CallbackVerbose()])

    # 求解DDP问题
    xs = [x0] * (solver[i].problem.T + 1)
    us = solver[i].problem.quasiStatic([x0] * solver[i].problem.T)
    
    print(f"   🔧 问题规模: T={solver[i].problem.T}, 开始求解...")
    solver[i].solve(xs, us, 200, False, 0.1)  # 增加最大迭代次数
    
    if solver[i].isFeasible:
        print(f"   ✅ 第 {i+1} 步求解成功!")
        print(f"      🔄 迭代次数: {solver[i].iter}")
        print(f"      💰 最终成本: {solver[i].cost:.6e}")
    else:
        print(f"   ❌ 第 {i+1} 步求解失败!")
        print(f"      💰 最终成本: {solver[i].cost:.6e}")

    # 将最终状态作为下一阶段的初始状态
    x0 = solver[i].xs[-1]

print(f"\n🎉 Hi机器人步行序列求解完成!")

# 可视化完整运动
if WITHDISPLAY:
    print(f"\n🎬 启动Hi机器人步行可视化...")
    print("   🚶 观察要点:")
    print("     • 双足交替抬起和着地")
    print("     • 重心在支撑足上方保持平衡")
    print("     • 关节运动的协调性")
    print("     • 步长和步频的一致性")
    
    try:
        import gepetto
        gepetto.corbaserver.Client()
        cameraTF = [2.0, 2.0, 1.2, 0.2, 0.62, 0.72, 0.22]
        display = crocoddyl.GepettoDisplay(hi_robot, 4, 4, cameraTF)
        print("📺 使用 Gepetto 可视化")
    except Exception as e:
        print(f"Gepetto 不可用: {e}")
        try:
            display = crocoddyl.MeshcatDisplay(hi_robot)
            print("📺 使用 Meshcat 可视化")
            print("🌐 可视化URL: http://127.0.0.1:7000/static/")
        except Exception as e:
            print(f"可视化设置错误: {e}")
            display = None

    if display:
        display.rate = -1
        display.freq = 1
        print("\n🎭 播放Hi机器人步行动作...")
        print("   每个步态阶段将依次播放")
        print("   按 Ctrl+C 停止播放")
        
        try:
            while True:
                for i, phase in enumerate(GAITPHASES):
                    if solver[i].isFeasible:
                        print(f"   🚶 播放第 {i+1} 步...")
                        display.displayFromSolver(solver[i])
                        time.sleep(1.0)  # 步态间隔
                    else:
                        print(f"   ⏭️ 跳过第 {i+1} 步（求解失败）")
                print("   🔄 重复播放...")
                time.sleep(2.0)
        except KeyboardInterrupt:
            print("\n⏹️  停止可视化")

# 绘制结果
if WITHPLOT:
    print(f"\n📊 绘制步行分析图...")
    try:
        # 绘制整体运动轨迹
        feasible_solvers = [s for s in solver if s.isFeasible]
        if feasible_solvers:
            plotSolution(feasible_solvers, bounds=False, figIndex=1, show=False)
        
        # 绘制每个阶段的收敛曲线
        for i, phase in enumerate(GAITPHASES):
            if solver[i].isFeasible:
                title = f"步行阶段 {i+1} - 步长: {phase['walking']['stepLength']:.2f}m"
                if len(solver[i].getCallbacks()) > 1:
                    log = solver[i].getCallbacks()[1]
                    crocoddyl.plotConvergence(
                        log.costs,
                        log.pregs,
                        log.dregs,
                        log.grads,
                        log.stops,
                        log.steps,
                        figTitle=title,
                        figIndex=i + 3,
                        show=True if i == len(GAITPHASES) - 1 else False,
                    )
        print("📈 分析图表已生成")
    except Exception as e:
        print(f"⚠️  绘图失败: {e}")

print(f"\n🎉 Hi机器人步行程序完成!")
print("=" * 50) 