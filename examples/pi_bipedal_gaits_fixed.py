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
    """创建一个更适合双足行走的半蹲姿态，调整质心高度"""
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
    
    # 调整基座高度以确保脚部接触地面，质心在合适高度
    q0[2] = 0.25   # 基座Z坐标，确保重心高度合理
    
    return q0

def load_pi_robot_fixed():
    """Load Pi robot with the fixed URDF"""
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
    
    # 创建改进的半蹲姿态
    q0 = create_half_sitting_config(pi_robot.model)
    
    # 验证重心位置
    data = pi_robot.model.createData()
    com = pinocchio.centerOfMass(pi_robot.model, data, q0)
    print(f"调整后重心位置: x={com[0]:.3f}, y={com[1]:.3f}, z={com[2]:.3f}")
    
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
    
    return pi_robot

# Load Pi robot
try:
    pi_robot = load_pi_robot_fixed()
    print(f"Successfully loaded Pi robot: {pi_robot.model.name}")
    print(f"DOF: {pi_robot.model.nq}, joints: {pi_robot.model.njoints}")
    
    # Print available frames
    print("Available frames:")
    for i, frame in enumerate(pi_robot.model.frames):
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

# 调整后的步态参数 - 步长5-10cm，步高1-10cm
GAITPHASES = [
    {
        "walking": {
            "stepLength": 0.05,  # 5cm步长
            "stepHeight": 0.01,  # 1cm步高
            "timeStep": 0.05,
            "stepKnots": 10,
            "supportKnots": 4,
        }
    },
    {
        "jumping": {
            "jumpHeight": 0.02,  # 2cm跳跃高度 (更保守)
            "jumpLength": [0.0, 0.03, 0.0],  # 3cm前向跳跃 (更保守)
            "timeStep": 0.05,
            "groundKnots": 6,    # 更多地面支撑时间
            "flyingKnots": 3,    # 更短飞行时间
        }
    },
    {
        "walking": {
            "stepLength": 0.10,  # 10cm步长
            "stepHeight": 0.10,  # 10cm步高
            "timeStep": 0.04,
            "stepKnots": 12,
            "supportKnots": 3,
        }
    },
]

solver = [None] * len(GAITPHASES)
for i, phase in enumerate(GAITPHASES):
    for key, value in phase.items():
        if key == "walking":
            # Creating a walking problem
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
        elif key == "jumping":
            # Creating a jumping problem
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
        solver[i].th_stop = 1e-4

    # Added the callback functions
    print(f"SOLVING {key.upper()} PHASE #{i}")
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
    
    solver[i].solve(xs, us, 50, False)
    
    # Defining the final state as initial one for the next phase
    if solver[i].isFeasible:
        x0 = solver[i].xs[-1]
        print(f"✓ Phase {i} ({key}) solved successfully in {solver[i].iter} iterations!")
        print(f"  Final cost: {solver[i].cost:.6e}")
    else:
        print(f"✗ WARNING: Phase {i} ({key}) did not converge!")

# Display the entire motion
if WITHDISPLAY:
    print("\n🤖 Starting Pi Robot visualization...")
    try:
        import gepetto
        gepetto.corbaserver.Client()
        cameraTF = [1.0, 1.0, 0.5, 0.2, 0.3, 0.5, 0.2]
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
        
        print("\n🎬 Playing Pi robot gait sequence...")
        print("   Press Ctrl+C to stop")
        
        try:
            while True:
                for i, phase in enumerate(GAITPHASES):
                    if solver[i].isFeasible:
                        phase_name = next(iter(phase.keys()))
                        print(f"🎭 Playing phase {i+1}: {phase_name}")
                        display.displayFromSolver(solver[i])
                time.sleep(2.0)
        except KeyboardInterrupt:
            print("\n⏹️  Visualization stopped by user")

# Plotting the entire motion
if WITHPLOT:
    print("\n📊 Generating motion plots...")
    plotSolution(solver, bounds=False, figIndex=1, show=False)

    for i, phase in enumerate(GAITPHASES):
        if solver[i].isFeasible:
            phase_name = next(iter(phase.keys()))
            title = f"Pi Robot - {phase_name.title()} Phase {i+1}"
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

print("\n🎉 Pi Robot bipedal gait demo completed!")
print("📁 URDF file: pi_robot_fixed.urdf")
print("🔧 Key improvements:")
print("   • Fixed mesh file paths (removed package:// references)")
print("   • Added sole links for better contact detection")
print("   • Improved collision geometries")
print("   • Added material definitions") 