import os
import signal
import sys
import time

import numpy as np
import pinocchio
from pinocchio.robot_wrapper import RobotWrapper

import crocoddyl

WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)

print("🚶 Hi Robot 简单步行程序")
print("=" * 50)

# 设置正确的路径
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
root_dir = os.path.dirname(base_dir)

# 加载Hi机器人
urdf_path = os.path.join(root_dir, "crocoddyl", "hi_robot", "urdf", "hi_23dof_250401_rl_fixed.urdf")
mesh_dir = os.path.join(root_dir, "crocoddyl", "hi_robot")

print(f"📂 加载Hi机器人从: {urdf_path}")

try:
    model, collision_model, visual_model = pinocchio.buildModelsFromUrdf(
        urdf_path, mesh_dir, pinocchio.JointModelFreeFlyer()
    )
    hi_robot = RobotWrapper(model)
    hi_robot.visual_model = visual_model
    hi_robot.collision_model = collision_model
    hi_robot.visual_data = visual_model.createData()
    hi_robot.collision_data = collision_model.createData()
    
    print(f"✅ 成功加载Hi机器人")
    
except Exception as e:
    print(f"❌ 完整模型加载失败: {e}")
    model = pinocchio.buildModelFromUrdf(urdf_path, pinocchio.JointModelFreeFlyer())
    hi_robot = RobotWrapper(model)
    hi_robot.visual_model = pinocchio.GeometryModel()
    hi_robot.collision_model = pinocchio.GeometryModel()
    hi_robot.visual_data = hi_robot.visual_model.createData()
    hi_robot.collision_data = hi_robot.collision_model.createData()

rmodel = hi_robot.model
rdata = rmodel.createData()

# 设置力矩限制
if hasattr(rmodel, 'effortLimit') and len(rmodel.effortLimit) > 0:
    lims = rmodel.effortLimit.copy()
    lims[7:] *= 0.8
    rmodel.effortLimit = lims

# 创建状态和驱动模型
state = crocoddyl.StateMultibody(rmodel)
actuation = crocoddyl.ActuationModelFloatingBase(state)

# 设置初始状态
q0 = pinocchio.neutral(rmodel)
q0[2] = 0.65  # 基座高度

# 设置自然站立姿态
joint_config = {
    "l_hip_pitch_joint": -5,
    "r_hip_pitch_joint": -5,
    "l_calf_joint": 10,
    "r_calf_joint": 10,
    "l_ankle_pitch_joint": -5,
    "r_ankle_pitch_joint": -5,
}

for joint_name, angle_deg in joint_config.items():
    try:
        if joint_name in [rmodel.names[i] for i in range(rmodel.njoints)]:
            joint_id = rmodel.getJointId(joint_name)
            q0[rmodel.joints[joint_id].idx_q] = np.radians(angle_deg)
    except:
        continue

x0 = np.concatenate([q0, np.zeros(rmodel.nv)])

# 定义足部frames
rightFoot = "r_ankle_roll_link"
leftFoot = "l_ankle_roll_link"

try:
    rightFootId = rmodel.getFrameId(rightFoot)
    leftFootId = rmodel.getFrameId(leftFoot)
    print(f"🔗 足部Frame ID: 右足={rightFootId}, 左足={leftFootId}")
except Exception as e:
    print(f"❌ 获取足部Frame ID失败: {e}")
    sys.exit(1)

# 计算初始足部位置和重心
pinocchio.forwardKinematics(rmodel, rdata, q0)
pinocchio.updateFramePlacements(rmodel, rdata)
rfPos0 = rdata.oMf[rightFootId].translation
lfPos0 = rdata.oMf[leftFootId].translation
comRef = (rfPos0 + lfPos0) / 2
comRef[2] = pinocchio.centerOfMass(rmodel, rdata, q0)[2].item()

print(f"📍 初始位置:")
print(f"   🦶 右足: ({rfPos0[0]:.3f}, {rfPos0[1]:.3f}, {rfPos0[2]:.3f})")
print(f"   🦶 左足: ({lfPos0[0]:.3f}, {lfPos0[1]:.3f}, {lfPos0[2]:.3f})")
print(f"   🎯 重心: ({comRef[0]:.3f}, {comRef[1]:.3f}, {comRef[2]:.3f})")

# 设置步行参数
DT = 0.05  # 时间步长
stepLength = 0.08  # 很小的步长
stepHeight = 0.03  # 很低的抬腿高度
T = 40     # 时间步数

print(f"🚶 步行参数:")
print(f"   📏 步长: {stepLength:.2f}m")
print(f"   📐 抬腿高度: {stepHeight:.2f}m")
print(f"   ⏱️  时间步长: {DT}s")
print(f"   📊 总步数: {T}")

# 计算目标足部位置 - 右足向前一步
rfTarget = rfPos0.copy()
rfTarget[0] += stepLength  # X方向前进

print(f"🎯 右足目标位置: ({rfTarget[0]:.3f}, {rfTarget[1]:.3f}, {rfTarget[2]:.3f})")

# 创建接触模型
contactModel2Feet = crocoddyl.ContactModelMultiple(state, actuation.nu)
contactModel1FootLeft = crocoddyl.ContactModelMultiple(state, actuation.nu)

# 双足支撑接触
supportContactModelLeft = crocoddyl.ContactModel6D(
    state, leftFootId, pinocchio.SE3.Identity(), pinocchio.LOCAL, actuation.nu, np.array([0, 25])
)
supportContactModelRight = crocoddyl.ContactModel6D(
    state, rightFootId, pinocchio.SE3.Identity(), pinocchio.LOCAL, actuation.nu, np.array([0, 25])
)

contactModel2Feet.addContact("left_foot", supportContactModelLeft)
contactModel2Feet.addContact("right_foot", supportContactModelRight)

# 左足支撑接触（右足摆动时）
contactModel1FootLeft.addContact("left_foot", supportContactModelLeft)

# 创建成本函数
def createRunningCostModel():
    runningCostModel = crocoddyl.CostModelSum(state, actuation.nu)
    
    # 状态正则化
    xResidual = crocoddyl.ResidualModelState(state, x0, actuation.nu)
    weights = np.array([0] * 3 + [10.0] * 3 + [0.01] * (state.nv - 6) + [1] * state.nv) ** 2
    xActivation = crocoddyl.ActivationModelWeightedQuad(weights)
    xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
    
    # 控制正则化
    uResidual = crocoddyl.ResidualModelControl(state, actuation.nu)
    uRegCost = crocoddyl.CostModelResidual(state, uResidual)
    
    # 重心稳定性
    comResidual = crocoddyl.ResidualModelCoMPosition(state, comRef, actuation.nu)
    comTrackCost = crocoddyl.CostModelResidual(state, comResidual)
    
    # 添加成本
    runningCostModel.addCost("stateReg", xRegCost, 1e-2)
    runningCostModel.addCost("ctrlReg", uRegCost, 1e-4)
    runningCostModel.addCost("comStable", comTrackCost, 1e1)
    
    return runningCostModel

def createSwingCostModel(target_pos, foot_id):
    swingCostModel = createRunningCostModel()
    
    # 足部位置跟踪
    footTrackingResidual = crocoddyl.ResidualModelFramePlacement(
        state, foot_id, pinocchio.SE3(np.eye(3), target_pos), actuation.nu
    )
    footTrackingCost = crocoddyl.CostModelResidual(state, footTrackingResidual)
    swingCostModel.addCost("footTracking", footTrackingCost, 1e2)
    
    return swingCostModel

# 创建动作模型列表
print("🔧 创建步行动作模型...")

models = []

# 阶段1: 双足支撑 (准备阶段)
dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
    state, actuation, contactModel2Feet, createRunningCostModel()
)
models.extend([crocoddyl.IntegratedActionModelEuler(dmodel, DT) for _ in range(8)])

# 阶段2: 右足摆动 (抬腿阶段)
# 创建一个渐进的轨迹
for i in range(T//2):
    progress = (i + 1) / (T//2)
    
    # 计算中间目标位置 (抛物线轨迹)
    intermediate_pos = rfPos0.copy()
    intermediate_pos[0] = rfPos0[0] + stepLength * progress  # X方向插值
    intermediate_pos[2] = rfPos0[2] + stepHeight * np.sin(np.pi * progress)  # Z方向抛物线
    
    swingCostModel = createSwingCostModel(intermediate_pos, rightFootId)
    dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, contactModel1FootLeft, swingCostModel
    )
    models.append(crocoddyl.IntegratedActionModelEuler(dmodel, DT))

# 阶段3: 右足着地 (双足支撑)
dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
    state, actuation, contactModel2Feet, createRunningCostModel()
)
models.extend([crocoddyl.IntegratedActionModelEuler(dmodel, DT) for _ in range(8)])

# 创建终端模型
terminalCostModel = createRunningCostModel()
# 右足位置终端成本
rfFinalResidual = crocoddyl.ResidualModelFramePlacement(
    state, rightFootId, pinocchio.SE3(np.eye(3), rfTarget), actuation.nu
)
rfFinalCost = crocoddyl.CostModelResidual(state, rfFinalResidual)
terminalCostModel.addCost("rfFinal", rfFinalCost, 1e3)

terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelContactFwdDynamics(
        state, actuation, contactModel2Feet, terminalCostModel
    ), 0
)

# 创建射击问题
print(f"🎯 创建射击问题 (T={len(models)})...")
problem = crocoddyl.ShootingProblem(x0, models, terminalModel)

# 创建DDP求解器
solver = crocoddyl.SolverBoxFDDP(problem)
solver.setCallbacks([crocoddyl.CallbackVerbose()])
solver.th_stop = 1e-6

# 求解
print("🚀 开始求解Hi机器人单步运动...")
xs = [x0] * (solver.problem.T + 1)
us = solver.problem.quasiStatic([x0] * solver.problem.T)
solver.solve(xs, us, 300, False, 1e-8)

if solver.isFeasible:
    print("✅ 单步运动求解成功!")
    print(f"   🔄 迭代次数: {solver.iter}")
    print(f"   💰 最终成本: {solver.cost:.6e}")
    
    # 分析最终结果
    xT = solver.xs[-1]
    pinocchio.forwardKinematics(rmodel, rdata, xT[:state.nq])
    pinocchio.updateFramePlacements(rmodel, rdata)
    finalRfPos = np.array(rdata.oMf[rightFootId].translation.T.flat)
    
    print(f"\n📊 步行结果分析:")
    print(f"   🎯 目标位置: ({rfTarget[0]:.3f}, {rfTarget[1]:.3f}, {rfTarget[2]:.3f})")
    print(f"   🦶 右足最终位置: ({finalRfPos[0]:.3f}, {finalRfPos[1]:.3f}, {finalRfPos[2]:.3f})")
    print(f"   📏 步行距离: {np.linalg.norm(finalRfPos - rfPos0):.3f}m")
    print(f"   📐 X方向误差: {abs(finalRfPos[0] - rfTarget[0]):.3f}m")
    
else:
    print("❌ 单步运动求解失败!")
    print(f"   💰 最终成本: {solver.cost:.6e}")

# 可视化
if WITHDISPLAY and solver.isFeasible:
    print(f"\n🎬 启动Hi机器人单步运动可视化...")
    
    try:
        import gepetto
        gepetto.corbaserver.Client()
        display = crocoddyl.GepettoDisplay(hi_robot)
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
        print("\n🎭 播放Hi机器人单步运动...")
        print("   按 Ctrl+C 停止播放")
        
        try:
            # 播放一次完整动画
            display.displayFromSolver(solver)
            print("✅ 动画播放完毕!")
            
            # 保持服务器运行
            print("\n⏸️  按 Ctrl+C 停止可视化服务器")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️  停止可视化")

print(f"\n🎉 Hi机器人单步运动程序完成!")
print("=" * 50) 