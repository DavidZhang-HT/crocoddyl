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

def create_half_sitting_config(model):
    """创建一个更适合双足行走的半蹲姿态，确保质心在合理高度"""
    q0 = pinocchio.neutral(model)
    
    # 调整髋关节和膝关节角度以获得更好的初始姿态
    # 关节顺序: [free_flyer(7), r_hip_pitch, r_hip_roll, r_thigh, r_calf, r_ankle_pitch, r_ankle_roll,
    #                            l_hip_pitch, l_hip_roll, l_thigh, l_calf, l_ankle_pitch, l_ankle_roll]
    
    # 右腿关节角度 (轻微弯曲)
    q0[7] = -0.2   # r_hip_pitch: 髋关节前倾
    q0[8] = 0.0    # r_hip_roll: 髋关节侧摆
    q0[9] = 0.0    # r_thigh: 大腿旋转
    q0[10] = 0.4   # r_calf: 膝关节弯曲
    q0[11] = -0.2  # r_ankle_pitch: 踝关节俯仰
    q0[12] = 0.0   # r_ankle_roll: 踝关节侧摆
    
    # 左腿关节角度 (对称)
    q0[13] = -0.2  # l_hip_pitch
    q0[14] = 0.0   # l_hip_roll
    q0[15] = 0.0   # l_thigh
    q0[16] = 0.4   # l_calf
    q0[17] = -0.2  # l_ankle_pitch
    q0[18] = 0.0   # l_ankle_roll
    
    # 调整基座高度以确保脚部接触地面
    q0[2] = 0.25   # 基座Z坐标，确保脚部在地面附近
    
    return q0

def load_pi_robot_advanced():
    """Load Pi robot with the fixed URDF for advanced bipedal motion"""
    # Get the path to the fixed URDF file
    urdf_path = os.path.join(os.path.dirname(__file__), "..", "pi_robot", "urdf", "pi_robot_fixed.urdf")
    mesh_dir = os.path.join(os.path.dirname(__file__), "..", "pi_robot")
    
    print(f"Loading Pi robot from: {urdf_path}")
    print(f"Mesh directory: {mesh_dir}")
    
    try:
        # Load with mesh files
        model, collision_model, visual_model = pinocchio.buildModelsFromUrdf(
            urdf_path, mesh_dir, pinocchio.JointModelFreeFlyer()
        )
        
        # Create robot wrapper
        pi_robot = RobotWrapper(model)
        pi_robot.visual_model = visual_model
        pi_robot.collision_model = collision_model
        pi_robot.visual_data = visual_model.createData()
        pi_robot.collision_data = collision_model.createData()
        
        print(f"Successfully loaded Pi robot with visual model: {len(visual_model.geometryObjects)} visual objects")
        
    except Exception as e:
        print(f"Warning: Could not load with visual models: {e}")
        # Fallback: load without mesh files
        model = pinocchio.buildModelFromUrdf(urdf_path, pinocchio.JointModelFreeFlyer())
        pi_robot = RobotWrapper(model)
        # Create empty visual/collision models
        pi_robot.visual_model = pinocchio.GeometryModel()
        pi_robot.collision_model = pinocchio.GeometryModel()
        pi_robot.visual_data = pi_robot.visual_model.createData()
        pi_robot.collision_data = pi_robot.collision_model.createData()
    
    # 创建改进的半蹲姿态，确保质心在合理位置
    q0 = create_half_sitting_config(pi_robot.model)
    
    # 验证重心位置
    data = pi_robot.model.createData()
    com = pinocchio.centerOfMass(pi_robot.model, data, q0)
    print(f"✅ 改进后重心位置: x={com[0]:.3f}, y={com[1]:.3f}, z={com[2]:.3f}")
    
    # Initialize referenceConfigurations properly
    if not hasattr(pi_robot.model, 'referenceConfigurations'):
        pi_robot.model.referenceConfigurations = {}
    pi_robot.model.referenceConfigurations["half_sitting"] = q0.copy()
    
    # Set initial configuration
    pi_robot.q0 = q0
    pi_robot.v0 = pinocchio.utils.zero(pi_robot.model.nv)
    
    # Set joint limits for free flyer
    ub = pi_robot.model.upperPositionLimit
    ub[:7] = 1
    pi_robot.model.upperPositionLimit = ub
    lb = pi_robot.model.lowerPositionLimit
    lb[:7] = -1
    pi_robot.model.lowerPositionLimit = lb
    
    # 更保守的力矩限制
    lims = pi_robot.model.effortLimit.copy()
    lims[7:] *= 0.5  # 减少50%力矩限制，更保守
    pi_robot.model.effortLimit = lims
    
    return pi_robot

# Load Pi robot
try:
    pi_robot = load_pi_robot_advanced()
    print(f"Successfully loaded Pi robot: {pi_robot.model.name}")
    print(f"DOF: {pi_robot.model.nq}, joints: {pi_robot.model.njoints}")
    
    # Print available frames
    print("Available frames:")
    for i, frame in enumerate(pi_robot.model.frames):
        if 'sole' in frame.name or 'foot' in frame.name:
            print(f"  {i}: {frame.name} ⭐")
        else:
            print(f"  {i}: {frame.name}")
        
except Exception as e:
    print(f"Error loading Pi robot: {e}")
    print("Falling back to Talos legs example...")
    import example_robot_data
    pi_robot = example_robot_data.load("talos_legs")

# Defining the initial state of the robot
q0 = pi_robot.model.referenceConfigurations["half_sitting"].copy()
v0 = pinocchio.utils.zero(pi_robot.model.nv)
x0 = np.concatenate([q0, v0])

# Setting up the 3d walking problem
if "pi_12dof" in pi_robot.model.name:
    # Use the new sole links for better contact detection
    rightFoot = "r_sole_link"
    leftFoot = "l_sole_link"
    print("Using Pi robot sole links for contact")
else:
    rightFoot = "right_sole_link"
    leftFoot = "left_sole_link"
    print("Using Talos robot sole links for contact")

print(f"Foot frames: left={leftFoot}, right={rightFoot}")

# Verify the frames exist
try:
    left_frame_id = pi_robot.model.getFrameId(leftFoot)
    right_frame_id = pi_robot.model.getFrameId(rightFoot)
    print(f"Frame IDs: left={left_frame_id}, right={right_frame_id}")
except:
    print(f"Warning: Could not find foot frames {leftFoot}, {rightFoot}")
    # Fallback to ankle roll links
    rightFoot = "r_ankle_roll_link"
    leftFoot = "l_ankle_roll_link"
    print(f"Fallback foot frames: left={leftFoot}, right={rightFoot}")

gait = SimpleBipedGaitProblem(pi_robot.model, rightFoot, leftFoot)

# 从走路到跑步的完整步态序列 - 包含飞行阶段的跑步动作
GAITPHASES = [
    {
        "warm_up_walk": {
            "stepLength": 0.05,  # 5cm起始步长
            "stepHeight": 0.01,  # 1cm起始步高
            "timeStep": 0.05,
            "stepKnots": 10,
            "supportKnots": 6,
        }
    },
    {
        "fast_walk": {
            "stepLength": 0.08,  # 8cm快步走
            "stepHeight": 0.03,  # 3cm步高
            "timeStep": 0.04,    # 更快节奏
            "stepKnots": 10,
            "supportKnots": 3,   # 更短支撑时间
        }
    },
    {
        "slow_jog": {
            "jumpHeight": 0.015, # 1.5cm轻微飞行
            "jumpLength": [0.08, 0.0, 0.0],  # 8cm前进
            "timeStep": 0.035,   # 更快的节奏
            "groundKnots": 4,    # 较短地面接触
            "flyingKnots": 2,    # 短暂飞行
        }
    },
    {
        "medium_jog": {
            "jumpHeight": 0.025, # 2.5cm飞行高度
            "jumpLength": [0.10, 0.0, 0.0],  # 10cm前进
            "timeStep": 0.03,    # 更快节奏
            "groundKnots": 3,    # 短地面接触
            "flyingKnots": 2,    # 飞行阶段
        }
    },
    {
        "fast_run": {
            "jumpHeight": 0.04,  # 4cm飞行高度
            "jumpLength": [0.12, 0.0, 0.0],  # 12cm前进
            "timeStep": 0.025,   # 快速节奏
            "groundKnots": 2,    # 很短地面接触
            "flyingKnots": 3,    # 明显飞行阶段
        }
    },
    {
        "sprint": {
            "jumpHeight": 0.05,  # 5cm最大飞行高度
            "jumpLength": [0.15, 0.0, 0.0],  # 15cm大步冲刺
            "timeStep": 0.02,    # 最快节奏
            "groundKnots": 2,    # 最短地面接触
            "flyingKnots": 4,    # 长飞行阶段
        }
    },
    {
        "decelerate": {
            "jumpHeight": 0.02,  # 减速跑
            "jumpLength": [0.08, 0.0, 0.0],  # 8cm步长
            "timeStep": 0.04,
            "groundKnots": 4,
            "flyingKnots": 2,
        }
    },
    {
        "cool_down": {
            "stepLength": 0.05,  # 恢复慢走
            "stepHeight": 0.02,
            "timeStep": 0.06,
            "stepKnots": 10,
            "supportKnots": 6,
        }
    },
]

solver = [None] * len(GAITPHASES)
phase_names = []

for i, phase in enumerate(GAITPHASES):
    for key, value in phase.items():
        phase_names.append(key)
        
        # 区分跑步（跳跃）和走路问题
        if key in ["slow_jog", "medium_jog", "fast_run", "sprint", "decelerate"]:
            # 跑步阶段 - 使用跳跃问题（有飞行阶段）
            solver[i] = crocoddyl.SolverBoxFDDP(
                gait.createJumpingProblem(
                    x0,
                    value["jumpHeight"],
                    value["jumpLength"],
                    value["timeStep"],
                    value["groundKnots"],
                    value["flyingKnots"],
                )
            )
        else:
            # 步行阶段 - 使用步行问题（始终有一脚着地）
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
        
        # 设置收敛条件 - 跑步阶段需要更宽松的条件
        if key in ["warm_up_walk", "fast_walk", "cool_down"]:
            solver[i].th_stop = 1e-3  # 步行阶段 - 较宽松
        elif key in ["slow_jog", "medium_jog"]:
            solver[i].th_stop = 5e-4  # 慢跑阶段 - 中等
        else:
            solver[i].th_stop = 1e-3  # 快跑/冲刺阶段 - 宽松（更难收敛）

    # Added the callback functions
    print(f"\n🎯 SOLVING PHASE {i+1}: {key.upper().replace('_', ' ')}")
    print(f"Parameters: {value}")
    
    if WITHPLOT:
        solver[i].setCallbacks(
            [
                crocoddyl.CallbackVerbose(),
                crocoddyl.CallbackLogger(),
            ]
        )
    else:
        solver[i].setCallbacks([crocoddyl.CallbackVerbose()])

    # Solving the problem with the BoxFDDP solver
    xs = [x0] * (solver[i].problem.T + 1)
    us = solver[i].problem.quasiStatic([x0] * solver[i].problem.T)
    
    # 根据阶段复杂度调整最大迭代次数
    if key in ["warm_up_walk", "fast_walk", "cool_down"]:
        max_iter = 30  # 步行阶段
    elif key in ["slow_jog", "medium_jog"]:
        max_iter = 60  # 慢跑中跑阶段
    else:
        max_iter = 80  # 快跑/冲刺阶段（最具挑战性）
    
    solver[i].solve(xs, us, max_iter, False)
    
    # Check convergence and update initial state
    if solver[i].isFeasible:
        x0 = solver[i].xs[-1]
        print(f"✅ Phase {i+1} ({key}) solved successfully in {solver[i].iter} iterations!")
        print(f"   Final cost: {solver[i].cost:.6e}")
        
        # Print some motion statistics
        if len(solver[i].xs) > 0:
            com_motion = []
            data = pi_robot.model.createData()
            for x in solver[i].xs:
                q_state = x[:pi_robot.model.nq]
                # Use proper CoM computation
                com_pos = pinocchio.centerOfMass(pi_robot.model, data, q_state)
                com_motion.append(com_pos.flatten())
            
            com_motion = np.array(com_motion)
            if len(com_motion) > 1:
                total_distance = np.sum(np.linalg.norm(np.diff(com_motion, axis=0), axis=1))
                print(f"   Total distance traveled: {total_distance:.3f}m")
                max_height = np.max(com_motion[:, 2])
                min_height = np.min(com_motion[:, 2])
                print(f"   Height variation: {min_height:.3f}m to {max_height:.3f}m")
    else:
        print(f"⚠️  WARNING: Phase {i+1} ({key}) did not converge!")
        print(f"   Continuing with current state...")

# Display the entire motion sequence
if WITHDISPLAY:
    print("\n🤖 Starting Pi Robot Advanced Bipedal Motion Visualization...")
    try:
        import gepetto
        gepetto.corbaserver.Client()
        cameraTF = [2.5, 2.5, 1.0, 0.3, 0.3, 0.6, 0.3]
        display = crocoddyl.GepettoDisplay(pi_robot, 4, 4, cameraTF)
        print("📺 Using Gepetto viewer")
    except Exception as e:
        print(f"Gepetto not available: {e}")
        try:
            display = crocoddyl.MeshcatDisplay(pi_robot)
            print("📺 Using Meshcat viewer")
            print("🌐 You can open the visualizer by visiting:")
            print("   http://127.0.0.1:7007/static/")
        except Exception as e:
            print(f"Error setting up visualization: {e}")
            display = None

    if display:
        display.rate = -1
        display.freq = 1
        
        print("\n🎭 Playing Pi Robot Advanced Motion Sequence...")
        print("   Phases: " + " → ".join([name.replace('_', ' ').title() for name in phase_names]))
        print("   Press Ctrl+C to stop")
        
        try:
            phase_delay = 1.0  # Delay between phases
            while True:
                for i, phase_name in enumerate(phase_names):
                    if solver[i] and solver[i].isFeasible:
                        print(f"🎬 Playing Phase {i+1}: {phase_name.replace('_', ' ').title()}")
                        display.displayFromSolver(solver[i])
                        if i < len(phase_names) - 1:  # Don't wait after last phase
                            time.sleep(phase_delay)
                print("🔄 Restarting sequence...\n")
                time.sleep(2.0)
        except KeyboardInterrupt:
            print("\n⏹️  Visualization stopped by user")

# Plotting the entire motion
if WITHPLOT:
    print("\n📊 Generating motion analysis plots...")
    
    # Only plot feasible solutions
    feasible_solvers = [s for s in solver if s and s.isFeasible]
    feasible_names = [name for i, name in enumerate(phase_names) if solver[i] and solver[i].isFeasible]
    
    if feasible_solvers:
        plotSolution(feasible_solvers, bounds=False, figIndex=1, show=False)

        for i, (s, name) in enumerate(zip(feasible_solvers, feasible_names)):
            if len(s.getCallbacks()) > 1 and hasattr(s.getCallbacks()[1], 'costs'):
                title = f"Pi Robot - {name.replace('_', ' ').title()}"
                log = s.getCallbacks()[1]
                crocoddyl.plotConvergence(
                    log.costs,
                    log.pregs,
                    log.dregs,
                    log.grads,
                    log.stops,
                    log.steps,
                    figTitle=title,
                    figIndex=i + 3,
                    show=True if i == len(feasible_solvers) - 1 else False,
                )

# Print summary
print("\n🎉 Pi Robot Advanced Bipedal Motion Demo Completed!")
print("📈 Motion Sequence Summary:")
for i, (name, s) in enumerate(zip(phase_names, solver)):
    status = "✅ Converged" if s and s.isFeasible else "❌ Failed"
    iterations = s.iter if s else "N/A"
    cost = f"{s.cost:.3e}" if s and s.isFeasible else "N/A"
    print(f"   {i+1}. {name.replace('_', ' ').title():<15} {status} ({iterations} iter, cost: {cost})")

print(f"\n🔧 Successfully completed {sum(1 for s in solver if s and s.isFeasible)}/{len(solver)} phases")
print("📁 Files used:")
print("   • URDF: pi_robot_fixed.urdf")
print("   • Script: pi_advanced_bipedal.py")
print("💡 This demo showcases:")
print("   • Progressive step complexity")
print("   • Multiple gait patterns")
print("   • Side stepping (simplified)")
print("   • Small jumping motion")
print("   • Motion recovery") 