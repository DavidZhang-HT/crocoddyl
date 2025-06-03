import os
import signal
import sys
import time

import numpy as np
import pinocchio
from pinocchio.robot_wrapper import RobotWrapper

import crocoddyl
from crocoddyl.utils.biped import plotSolution

WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)

print("🥋 Hi Robot Taichi 完整测试程序")
print("=" * 50)

# 加载hi机器人 - 使用完整URDF和mesh
script_dir = os.path.dirname(os.path.abspath(__file__))  # examples目录
base_dir = os.path.dirname(script_dir)  # crocoddyl目录
root_dir = os.path.dirname(base_dir)    # motionControl目录

# 使用修复后的URDF文件
urdf_path = os.path.join(root_dir, "crocoddyl", "hi_robot", "urdf", "hi_23dof_250401_rl_fixed.urdf")
mesh_dir = os.path.join(root_dir, "crocoddyl", "hi_robot")

print(f"📂 脚本目录: {script_dir}")
print(f"📂 根目录: {root_dir}")
print(f"📂 加载Hi机器人从: {urdf_path}")  
print(f"📂 Mesh目录: {mesh_dir}")

# 验证文件和目录存在
if not os.path.exists(urdf_path):
    print(f"❌ 修复后的URDF文件不存在: {urdf_path}")
    print("🔄 回退到原始URDF...")
    urdf_path = os.path.join(root_dir, "crocoddyl", "hi_robot", "urdf", "hi_23dof_250401_rl.urdf")

if not os.path.exists(mesh_dir):
    print(f"❌ Mesh目录不存在: {mesh_dir}")
    sys.exit(1)

mesh_check_dir = os.path.join(mesh_dir, "meshes")
if os.path.exists(mesh_check_dir):
    mesh_files = os.listdir(mesh_check_dir)
    print(f"📁 发现 {len(mesh_files)} 个mesh文件")
    
    # 验证几个关键mesh文件存在
    key_meshes = ["base_link.STL", "waist_link.STL", "l_ankle_roll_link.STL", "r_ankle_roll_link.STL"]
    for mesh in key_meshes:
        mesh_path = os.path.join(mesh_check_dir, mesh)
        if os.path.exists(mesh_path):
            print(f"✅ {mesh}: 存在")
        else:
            print(f"❌ {mesh}: 不存在")
else:
    print(f"❌ Meshes子目录不存在: {mesh_check_dir}")

mesh_load_success = False
try:
    print("🔄 尝试加载完整机器人模型（包含mesh）...")
    
    # 使用buildModelsFromUrdf加载完整模型
    model, collision_model, visual_model = pinocchio.buildModelsFromUrdf(
        urdf_path, 
        mesh_dir,  # mesh根目录
        pinocchio.JointModelFreeFlyer()
    )
    
    # 创建机器人包装器
    hi_robot = RobotWrapper(model)
    hi_robot.visual_model = visual_model
    hi_robot.collision_model = collision_model
    hi_robot.visual_data = visual_model.createData()
    hi_robot.collision_data = collision_model.createData()
    
    print(f"✅ 成功加载完整模型!")
    print(f"   🎨 Visual几何体数量: {len(visual_model.geometryObjects)}")
    print(f"   🔥 Collision几何体数量: {len(collision_model.geometryObjects)}")
    
    # 打印visual geometry信息用于调试
    if len(visual_model.geometryObjects) > 0:
        print("🎨 Visual几何体信息:")
        for i, geom in enumerate(visual_model.geometryObjects):
            if i < 5:  # 显示前5个
                print(f"   {i}: {geom.name} - {geom.meshPath}")
            elif i == 5:
                print(f"   ... 还有 {len(visual_model.geometryObjects) - 5} 个几何体")
                break
    
    mesh_load_success = True
    
except Exception as e:
    print(f"❌ 完整模型加载失败: {e}")
    print("🔄 尝试加载简化模型...")
    mesh_load_success = False
    
    try:
        # 回退到只加载运动学模型
        model = pinocchio.buildModelFromUrdf(urdf_path, pinocchio.JointModelFreeFlyer())
        hi_robot = RobotWrapper(model)
        
        # 手动创建一些基本几何体用于可视化
        visual_model = pinocchio.GeometryModel()
        collision_model = pinocchio.GeometryModel()
        
        # 为主要关节添加简单几何体
        import hppfcl
        
        # 为基座添加一个盒子
        base_geom = pinocchio.GeometryObject("base_visual", 
                                           model.getFrameId("base_link"), 
                                           hppfcl.Box(0.2, 0.3, 0.15),
                                           pinocchio.SE3.Identity())
        base_geom.meshColor = np.array([0.8, 0.8, 0.8, 1.0])
        visual_model.addGeometryObject(base_geom)
        
        # 为手部添加一个小球
        try:
            hand_frame_id = model.getFrameId("l_wrist_link")
            hand_geom = pinocchio.GeometryObject("hand_visual", 
                                               hand_frame_id,
                                               hppfcl.Sphere(0.03),
                                               pinocchio.SE3.Identity())
            hand_geom.meshColor = np.array([1.0, 0.0, 0.0, 1.0])  # 红色
            visual_model.addGeometryObject(hand_geom)
        except:
            print("⚠️  无法添加手部几何体")
        
        # 为脚部添加几何体
        try:
            for foot_name, color in [("l_ankle_roll_link", [0.0, 1.0, 0.0, 1.0]), 
                                   ("r_ankle_roll_link", [0.0, 0.0, 1.0, 1.0])]:
                foot_frame_id = model.getFrameId(foot_name)
                foot_geom = pinocchio.GeometryObject(f"{foot_name}_visual", 
                                                   foot_frame_id,
                                                   hppfcl.Box(0.2, 0.1, 0.05),
                                                   pinocchio.SE3.Identity())
                foot_geom.meshColor = np.array(color)
                visual_model.addGeometryObject(foot_geom)
        except Exception as e:
            print(f"⚠️  无法添加脚部几何体: {e}")
        
        hi_robot.visual_model = visual_model
        hi_robot.collision_model = collision_model
        hi_robot.visual_data = visual_model.createData()
        hi_robot.collision_data = collision_model.createData()
        
        print(f"✅ 使用带简单几何体的模型")
        print(f"   🎨 添加了 {len(visual_model.geometryObjects)} 个基本几何体")
        
    except Exception as e2:
        print(f"❌ 简化模型也失败: {e2}")
        sys.exit(1)

# 获取机器人模型和数据
rmodel = hi_robot.model
rdata = rmodel.createData()

# 设置力矩限制
if hasattr(rmodel, 'effortLimit') and len(rmodel.effortLimit) > 0:
    lims = rmodel.effortLimit.copy()
    # 适当降低力矩限制以使运动更平滑
    lims[7:] *= 0.7  # 降低关节力矩限制
    rmodel.effortLimit = lims

# 创建状态和驱动模型
state = crocoddyl.StateMultibody(rmodel)
actuation = crocoddyl.ActuationModelFloatingBase(state)

print(f"   ⚙️  状态维度: {state.nx} (nq={state.nq}, nv={state.nv})")
print(f"   🎮 控制维度: {actuation.nu}")

# 设置积分时间和总时间步数
DT = 3e-2  # 较小的时间步长以获得更平滑的运动
T = 50     # 总时间步数
target = np.array([0.3, 0.3, 1.1])  # 目标位置 (x, y, z)

print(f"🎯 太极运动目标:")
print(f"   📍 目标位置: ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})")
print(f"   ⏱️  时间步长: {DT}s")
print(f"   📊 总步数: {T}")

# 定义关键frame名称
rightFoot = "r_ankle_roll_link"  # hi_robot的右足
leftFoot = "l_ankle_roll_link"   # hi_robot的左足
leftHand = "l_wrist_link"        # hi_robot的左手

try:
    # 获取frame ID
    rightFootId = rmodel.getFrameId(rightFoot)
    leftFootId = rmodel.getFrameId(leftFoot)
    leftHandId = rmodel.getFrameId(leftHand)
    
    print(f"🔗 关键Frame ID:")
    print(f"   🦶 右足: {rightFoot} (ID: {rightFootId})")
    print(f"   🦶 左足: {leftFoot} (ID: {leftFootId})")
    print(f"   ✋ 左手: {leftHand} (ID: {leftHandId})")
    
except Exception as e:
    print(f"❌ 获取Frame ID失败: {e}")
    print("🔍 可用的frames:")
    for i, frame in enumerate(rmodel.frames):
        if i < 20:  # 只显示前20个
            print(f"   {i}: {frame.name}")
    sys.exit(1)

# 设置初始配置
q0 = pinocchio.neutral(rmodel)
q0[2] = 0.65  # 设置基座高度，让机器人站立

# 创建一个自然的站立姿态
try:
    # 获取关节索引并设置初始角度
    joint_names = [joint.shortname() for joint in rmodel.joints]
    print(f"🦴 关节类型: {joint_names[:10]}...")  # 显示前10个关节类型
    
    # 设置自然站立姿态
    if "l_hip_pitch_joint" in [rmodel.names[i] for i in range(rmodel.njoints)]:
        l_hip_pitch_id = rmodel.getJointId("l_hip_pitch_joint")
        q0[rmodel.joints[l_hip_pitch_id].idx_q] = np.radians(-10)  # 髋关节轻微前倾
    
    if "r_hip_pitch_joint" in [rmodel.names[i] for i in range(rmodel.njoints)]:
        r_hip_pitch_id = rmodel.getJointId("r_hip_pitch_joint")
        q0[rmodel.joints[r_hip_pitch_id].idx_q] = np.radians(-10)
    
    if "l_calf_joint" in [rmodel.names[i] for i in range(rmodel.njoints)]:
        l_calf_id = rmodel.getJointId("l_calf_joint")
        q0[rmodel.joints[l_calf_id].idx_q] = np.radians(20)       # 膝关节轻微弯曲
    
    if "r_calf_joint" in [rmodel.names[i] for i in range(rmodel.njoints)]:
        r_calf_id = rmodel.getJointId("r_calf_joint")
        q0[rmodel.joints[r_calf_id].idx_q] = np.radians(20)
    
    if "l_shoulder_pitch_joint" in [rmodel.names[i] for i in range(rmodel.njoints)]:
        l_shoulder_pitch_id = rmodel.getJointId("l_shoulder_pitch_joint")
        q0[rmodel.joints[l_shoulder_pitch_id].idx_q] = np.radians(30)  # 手臂自然下垂
    
    if "r_shoulder_pitch_joint" in [rmodel.names[i] for i in range(rmodel.njoints)]:
        r_shoulder_pitch_id = rmodel.getJointId("r_shoulder_pitch_joint")
        q0[rmodel.joints[r_shoulder_pitch_id].idx_q] = np.radians(30)
    
    print("✅ 设置自然站立姿态")
    
except Exception as e:
    print(f"⚠️  使用默认neutral配置: {e}")

# 设置初始状态
x0 = np.concatenate([q0, np.zeros(rmodel.nv)])

# 计算初始位置
pinocchio.forwardKinematics(rmodel, rdata, q0)
pinocchio.updateFramePlacements(rmodel, rdata)
rfPos0 = rdata.oMf[rightFootId].translation
lfPos0 = rdata.oMf[leftFootId].translation
leftHandPos0 = rdata.oMf[leftHandId].translation

# 计算重心参考
comRef = (rfPos0 + lfPos0) / 2
comRef[2] = pinocchio.centerOfMass(rmodel, rdata, q0)[2].item()

print(f"📍 初始位置:")
print(f"   🦶 右足: ({rfPos0[0]:.3f}, {rfPos0[1]:.3f}, {rfPos0[2]:.3f})")
print(f"   🦶 左足: ({lfPos0[0]:.3f}, {lfPos0[1]:.3f}, {lfPos0[2]:.3f})")
print(f"   ✋ 左手: ({leftHandPos0[0]:.3f}, {leftHandPos0[1]:.3f}, {leftHandPos0[2]:.3f})")
print(f"   🎯 重心: ({comRef[0]:.3f}, {comRef[1]:.3f}, {comRef[2]:.3f})")

# 创建接触模型
contactModel2Feet = crocoddyl.ContactModelMultiple(state, actuation.nu)

# 双足接触支撑
supportContactModelLeft = crocoddyl.ContactModel6D(
    state,
    leftFootId,
    pinocchio.SE3.Identity(),
    pinocchio.LOCAL,
    actuation.nu,
    np.array([0, 30]),  # 适当降低接触力限制
)
supportContactModelRight = crocoddyl.ContactModel6D(
    state,
    rightFootId,
    pinocchio.SE3.Identity(),
    pinocchio.LOCAL,
    actuation.nu,
    np.array([0, 30]),
)

contactModel2Feet.addContact(leftFoot + "_contact", supportContactModelLeft)
contactModel2Feet.addContact(rightFoot + "_contact", supportContactModelRight)

print("🔗 创建双足接触模型")

# 快速太极动作设置 - 简化成本函数
print("💰 创建简化成本函数")

# 1. 状态正则化成本
xResidual = crocoddyl.ResidualModelState(state, x0, actuation.nu)
state_weights = np.array([0] * 3 + [5.0] * 3 + [0.01] * (state.nv - 6) + [1] * state.nv) ** 2
xActivation = crocoddyl.ActivationModelWeightedQuad(state_weights)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)

# 2. 控制正则化成本
uResidual = crocoddyl.ResidualModelControl(state, actuation.nu)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)

# 3. 终端状态成本
terminal_weights = np.array([0] * 3 + [10.0] * 3 + [0.01] * (state.nv - 6) + [50] * state.nv) ** 2
xTActivation = crocoddyl.ActivationModelWeightedQuad(terminal_weights)
xRegTermCost = crocoddyl.CostModelResidual(state, xTActivation, xResidual)

# 4. 左手跟踪成本 - 太极动作的关键
handTrackingResidual = crocoddyl.ResidualModelFramePlacement(
    state, leftHandId, pinocchio.SE3(np.eye(3), target), actuation.nu
)
handTrackingActivation = crocoddyl.ActivationModelWeightedQuad(
    np.array([1] * 3 + [0.1] * 3) ** 2  # 位置权重高，姿态权重低
)
handTrackingCost = crocoddyl.CostModelResidual(
    state, handTrackingActivation, handTrackingResidual
)

# 5. 重心稳定性成本
comResidual = crocoddyl.ResidualModelCoMPosition(state, comRef, actuation.nu)
comTrackCost = crocoddyl.CostModelResidual(state, comResidual)

# 创建运行成本模型
runningCostModel = crocoddyl.CostModelSum(state, actuation.nu)
terminalCostModel = crocoddyl.CostModelSum(state, actuation.nu)

# 添加成本 - 简化权重
runningCostModel.addCost("handPose", handTrackingCost, 1e2)
runningCostModel.addCost("stateReg", xRegCost, 1e-2)
runningCostModel.addCost("ctrlReg", uRegCost, 1e-3)
runningCostModel.addCost("comStable", comTrackCost, 5e1)

# 终端成本模型
terminalCostModel.addCost("handPose", handTrackingCost, 1e2)
terminalCostModel.addCost("stateReg", xRegTermCost, 1e-2)
terminalCostModel.addCost("comStable", comTrackCost, 1e2)

print("🎭 创建简化太极动作模型")

# 创建微分动作模型
dmodelRunning = crocoddyl.DifferentialActionModelContactFwdDynamics(
    state, actuation, contactModel2Feet, runningCostModel
)
dmodelTerminal = crocoddyl.DifferentialActionModelContactFwdDynamics(
    state, actuation, contactModel2Feet, terminalCostModel
)

# 创建积分动作模型
runningModel = crocoddyl.IntegratedActionModelEuler(dmodelRunning, DT)
terminalModel = crocoddyl.IntegratedActionModelEuler(dmodelTerminal, 0)

print("⚙️  创建积分动作模型")

# 定义问题
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

print("🎯 创建射击问题")

# 创建DDP求解器
solver = crocoddyl.SolverBoxFDDP(problem)

if WITHPLOT:
    solver.setCallbacks([
        crocoddyl.CallbackVerbose(),
        crocoddyl.CallbackLogger(),
    ])
else:
    solver.setCallbacks([crocoddyl.CallbackVerbose()])

print("🔧 配置求解器")

# 求解优化问题
print("\n🚀 开始求解Hi机器人太极动作...")
print("=" * 50)

xs = [x0] * (solver.problem.T + 1)
us = solver.problem.quasiStatic([x0] * solver.problem.T)
solver.th_stop = 1e-6
solver.solve(xs, us, 300, False, 1e-8)

print("\n" + "=" * 50)
if solver.isFeasible:
    print("✅ 太极动作求解成功!")
    print(f"   🔄 迭代次数: {solver.iter}")
    print(f"   💰 最终成本: {solver.cost:.6e}")
    
    # 分析最终结果
    xT = solver.xs[-1]
    pinocchio.forwardKinematics(rmodel, rdata, xT[:state.nq])
    pinocchio.updateFramePlacements(rmodel, rdata)
    com = pinocchio.centerOfMass(rmodel, rdata, xT[:state.nq])
    finalLeftHandPos = np.array(rdata.oMf[leftHandId].translation.T.flat)
    
    print(f"\n📊 最终结果分析:")
    print(f"   🎯 目标位置: ({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})")
    print(f"   ✋ 左手最终位置: ({finalLeftHandPos[0]:.3f}, {finalLeftHandPos[1]:.3f}, {finalLeftHandPos[2]:.3f})")
    print(f"   📏 手部到目标距离: {np.linalg.norm(finalLeftHandPos - target):.3f}m")
    print(f"   ⚖️  重心XY距离: {np.linalg.norm(com[:2] - comRef[:2]):.3f}m")
    print(f"   📐 与初始状态距离: {np.linalg.norm(x0 - np.array(xT.flat)):.3f}")
    
else:
    print("❌ 太极动作求解失败!")
    print(f"   💰 最终成本: {solver.cost:.6e}")

# 可视化
display = None
if WITHDISPLAY:
    print(f"\n🎬 启动Hi机器人太极动作可视化...")
    print("   🥋 观察要点:")
    print("     • 左手从初始位置到目标位置的平滑运动")
    print("     • 双足保持稳定接触")
    print("     • 重心保持平衡")
    print("     • 关节运动的协调性")
    
    try:
        import gepetto
        gepetto.corbaserver.Client()
        display = crocoddyl.GepettoDisplay(hi_robot)
        # 添加目标点标记
        display.robot.viewer.gui.addSphere("world/target", 0.03, [1.0, 0.0, 0.0, 1.0])
        display.robot.viewer.gui.applyConfiguration(
            "world/target", [*target.tolist(), 0.0, 0.0, 0.0, 1.0]
        )
        print("📺 使用 Gepetto 可视化")
    except Exception as e:
        print(f"Gepetto 不可用: {e}")
        try:
            display = crocoddyl.MeshcatDisplay(hi_robot)
            if mesh_load_success:
                print("📺 使用 Meshcat 可视化 - 完整mesh模型")
                print("🌐 可视化URL: http://127.0.0.1:7000/static/")
            else:
                print("📺 使用 Meshcat 可视化 - 简化几何体")
                print("🌐 可视化URL: http://127.0.0.1:7000/static/")
            
            # 添加目标点标记到meshcat
            try:
                import meshcat.geometry as g
                display.robot.viewer["target"].set_object(g.Sphere(0.03), g.MeshLambertMaterial(color=0xff0000))
                display.robot.viewer["target"].set_transform(
                    np.array([[1, 0, 0, target[0]],
                             [0, 1, 0, target[1]],
                             [0, 0, 1, target[2]],
                             [0, 0, 0, 1]])
                )
                print("🎯 添加目标点标记")
            except Exception as e:
                print(f"⚠️  无法添加目标点: {e}")
                
        except Exception as e:
            print(f"可视化设置错误: {e}")
            display = None

    if display and solver.isFeasible:
        display.rate = -1
        display.freq = 1
        print("\n🎭 播放Hi机器人太极动作...")
        print("   按 Ctrl+C 停止播放")
        
        try:
            # 播放一次完整动画
            display.displayFromSolver(solver)
            print("✅ 动画播放完毕!")
            if mesh_load_success:
                print("🌐 现在您应该能看到完整的Hi机器人模型进行太极动作")
                print("   包含所有关节的详细STL mesh文件")
            else:
                print("🌐 使用简化几何体显示机器人太极动作")
            
            # 保持服务器运行
            print("\n⏸️  按 Ctrl+C 停止可视化服务器")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️  停止可视化")

print(f"\n🎉 Hi机器人太极测试完成!")
print("=" * 50) 