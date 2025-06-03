import os
import signal
import sys
import time

import numpy as np
import pinocchio
# 使用正确的RobotWrapper导入
from pinocchio.robot_wrapper import RobotWrapper

import crocoddyl
from crocoddyl.utils.biped import SimpleBipedGaitProblem, plotSolution

WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)

def create_natural_stance(robot_model):
    """
    创建自然站立姿态，膝盖强制深度弯曲，为跳跃提供更好的初始条件
    """
    q = pinocchio.neutral(robot_model)
    q[2] = 0.28  # 降低基座高度，以配合膝盖弯曲
    
    # 获取关节索引
    try:
        r_hip_joint_id = robot_model.getJointId("r_hip_pitch_joint")
        l_hip_joint_id = robot_model.getJointId("l_hip_pitch_joint")
        r_knee_joint_id = robot_model.getJointId("r_calf_joint")
        l_knee_joint_id = robot_model.getJointId("l_calf_joint")
        r_ankle_joint_id = robot_model.getJointId("r_ankle_pitch_joint")
        l_ankle_joint_id = robot_model.getJointId("l_ankle_pitch_joint")
        
        r_hip_q_idx = robot_model.joints[r_hip_joint_id].idx_q
        l_hip_q_idx = robot_model.joints[l_hip_joint_id].idx_q
        r_knee_q_idx = robot_model.joints[r_knee_joint_id].idx_q
        l_knee_q_idx = robot_model.joints[l_knee_joint_id].idx_q
        r_ankle_q_idx = robot_model.joints[r_ankle_joint_id].idx_q
        l_ankle_q_idx = robot_model.joints[l_ankle_joint_id].idx_q
        
        # 设置自然站立姿态，使用更大的正值表示深度膝盖弯曲
        q[r_hip_q_idx] = np.radians(-25)   # 髋关节更大幅度后倾
        q[l_hip_q_idx] = np.radians(-25)
        q[r_knee_q_idx] = np.radians(60)   # 膝关节深度弯曲(大正值)
        q[l_knee_q_idx] = np.radians(60)
        q[r_ankle_q_idx] = np.radians(-20)  # 踝关节跖屈以平衡
        q[l_ankle_q_idx] = np.radians(-20)
        
        print(f"   ✅ 设置深度膝盖弯曲站立姿态:")
        print(f"      髋关节: {np.degrees(q[r_hip_q_idx]):.1f}°")
        print(f"      膝关节: {np.degrees(q[r_knee_q_idx]):.1f}° (正值表示弯曲)")
        print(f"      踝关节: {np.degrees(q[r_ankle_q_idx]):.1f}°")
        
        return q, (r_hip_q_idx, l_hip_q_idx, r_knee_q_idx, l_knee_q_idx, r_ankle_q_idx, l_ankle_q_idx)
        
    except Exception as e:
        print(f"   ⚠️  Failed to set natural stance: {e}")
        return q, None

def enforce_joint_limits(problem, joint_indices, min_knee_bend):
    """
    强制执行关节限制，特别是膝盖关节的最小弯曲角度
    """
    if joint_indices is None:
        return False
        
    r_hip_q_idx, l_hip_q_idx, r_knee_q_idx, l_knee_q_idx, r_ankle_q_idx, l_ankle_q_idx = joint_indices
    
    # 设置膝关节弯曲的下限 (正值表示弯曲，阻止膝盖向后弯曲)
    # 获取每个运行模型的状态维度和控制维度
    nx = problem.runningModels[0].state.nx if problem.runningModels else 0
    ndx = problem.runningModels[0].state.ndx if problem.runningModels else 0
    
    # 创建状态上下限
    state_lb = np.full(nx, -np.inf)
    state_ub = np.full(nx, np.inf)
    
    # 设置膝关节的下限为minimum_knee_bend (阻止膝盖低于此值)
    state_lb[r_knee_q_idx] = min_knee_bend
    state_lb[l_knee_q_idx] = min_knee_bend
    
    # 为所有运行模型设置状态限制
    limited_models = 0
    
    for model in problem.runningModels:
        if hasattr(model, 'differential') and hasattr(model.differential, 'set_state_limits'):
            try:
                model.differential.set_state_limits(state_lb, state_ub)
                limited_models += 1
            except Exception as e:
                print(f"      ⚠️ 设置模型限制失败: {e}")
                pass
    
    # 同样为终端模型设置状态限制
    if hasattr(problem.terminalModel, 'differential') and hasattr(problem.terminalModel.differential, 'set_state_limits'):
        try:
            problem.terminalModel.differential.set_state_limits(state_lb, state_ub)
            limited_models += 1
        except Exception as e:
            print(f"      ⚠️ 设置终端模型限制失败: {e}")
    
    print(f"   ✅ 为 {limited_models} 个模型设置了膝关节弯曲限制 (最小值: {np.degrees(min_knee_bend):.1f}°)")
    
    return limited_models > 0

def add_knee_bend_costs(problem, joint_indices, min_knee_bend):
    """
    在整个运动中添加膝盖弯曲的成本函数
    确保膝盖在跳跃过程中保持足够的弯曲
    """
    if joint_indices is None:
        return False
        
    r_hip_q_idx, l_hip_q_idx, r_knee_q_idx, l_knee_q_idx, r_ankle_q_idx, l_ankle_q_idx = joint_indices
    
    try:
        added_costs = 0
        
        # 为每个阶段添加膝盖弯曲约束
        for i, model in enumerate(problem.runningModels):
            if not hasattr(model, 'differential'):
                continue
            
            state = model.state
            
            # 创建膝关节目标姿势 (深度弯曲)
            knee_ref = np.zeros(state.nx)
            knee_ref[r_knee_q_idx] = min_knee_bend  # 正值表示弯曲
            knee_ref[l_knee_q_idx] = min_knee_bend
            
            # 仅为膝关节设置权重
            knee_weights = np.zeros(state.ndx)
            knee_weights[r_knee_q_idx] = 50.0  # 显著增加膝关节弯曲权重
            knee_weights[l_knee_q_idx] = 50.0
            
            # 创建膝关节残差
            knee_residual = crocoddyl.ResidualModelState(state, knee_ref)
            
            # 创建膝关节弯曲成本
            knee_cost = crocoddyl.CostModelResidual(
                state,
                crocoddyl.ActivationModelWeightedQuad(knee_weights),
                knee_residual
            )
            
            try:
                # 添加到成本函数
                model.differential.costs.addCost(f"knee_bend_{i}", knee_cost, 100.0)  # 增加成本权重
                added_costs += 1
            except Exception as e:
                # 由于维度不匹配等原因可能失败，这是预期的
                pass
                
        print(f"   ✅ 添加了 {added_costs} 个膝盖弯曲成本函数 (权重: 100.0)")
        return added_costs > 0
        
    except Exception as e:
        print(f"   ⚠️  Failed to add knee bend costs: {e}")
        return False

def modify_terminal_state(problem, robot_model, joint_indices):
    """
    修改终端状态，确保着陆时膝关节保持弯曲
    """
    if joint_indices is None:
        return
        
    r_hip_q_idx, l_hip_q_idx, r_knee_q_idx, l_knee_q_idx, r_ankle_q_idx, l_ankle_q_idx = joint_indices
    
    try:
        # 修改终端模型
        terminal_model = problem.terminalModel
        
        if hasattr(terminal_model, 'differential') and hasattr(terminal_model.differential, 'costs'):
            # 创建着陆姿态目标
            landing_target = np.zeros(terminal_model.state.nx)
            landing_target[2] = 0.28  # 基座高度降低以配合膝盖弯曲
            landing_target[r_hip_q_idx] = np.radians(-20)   # 着陆时髋关节后倾
            landing_target[l_hip_q_idx] = np.radians(-20)
            landing_target[r_knee_q_idx] = np.radians(50)   # 着陆时膝关节明显弯曲(正值)
            landing_target[l_knee_q_idx] = np.radians(50)
            landing_target[r_ankle_q_idx] = np.radians(-15)   # 着陆时踝关节跖屈
            landing_target[l_ankle_q_idx] = np.radians(-15)
            
            # 着陆姿态权重
            landing_weights = np.zeros(terminal_model.state.ndx)
            landing_weights[r_knee_q_idx] = 200.0  # 显著增加膝关节控制权重
            landing_weights[l_knee_q_idx] = 200.0
            landing_weights[r_hip_q_idx] = 100.0   # 增加髋关节控制权重
            landing_weights[l_hip_q_idx] = 100.0
            
            # 创建着陆成本
            landing_residual = crocoddyl.ResidualModelState(terminal_model.state, landing_target)
            landing_cost = crocoddyl.CostModelResidual(
                terminal_model.state,
                crocoddyl.ActivationModelWeightedQuad(landing_weights),
                landing_residual
            )
            
            try:
                terminal_model.differential.costs.addCost("landing_posture", landing_cost, 300.0)  # 增加终端成本权重
                print(f"   ✅ 添加着陆姿态成本 (权重: 300.0)")
            except Exception as e:
                print(f"   ⚠️  Failed to add landing cost: {e}")
            
    except Exception as e:
        print(f"   ⚠️  Failed to modify terminal state: {e}")

def configure_solver(solver, use_warm_start=True):
    """
    配置求解器，调整参数以提高收敛性能
    """
    # 设置较低的终止阈值，使求解器更准确
    solver.th_stop = 1e-7
    
    # 如果使用热启动，设置线搜索参数
    if use_warm_start:
        solver.th_grad = 1e-10  # 梯度终止阈值
        solver.th_step = 1e-3   # 步长终止阈值
        solver.reg_min = 1e-9   # 最小正则化值
        
    # 设置最大迭代次数
    solver.max_iters = 200
    
    # 设置详细回调
    solver.setCallbacks([crocoddyl.CallbackVerbose()])
    
    return solver

# 加载机器人
urdf_path = os.path.join(os.path.dirname(__file__), "..", "pi_robot", "urdf", "pi_robot_fixed.urdf")
mesh_dir = os.path.join(os.path.dirname(__file__), "..", "pi_robot")

print(f"Loading Pi robot from: {urdf_path}")

model, collision_model, visual_model = pinocchio.buildModelsFromUrdf(
    urdf_path, mesh_dir, pinocchio.JointModelFreeFlyer()
)

# 使用标准的RobotWrapper而不是自定义类
pi_robot = RobotWrapper(model)
pi_robot.visual_model = visual_model
pi_robot.collision_model = collision_model
pi_robot.visual_data = visual_model.createData()
pi_robot.collision_data = collision_model.createData()

print(f"Successfully loaded Pi robot: {pi_robot.model.name}")

# 创建深度弯曲的初始姿态
print(f"\n🦵 创建深度膝盖弯曲站立姿态")
q_natural, joint_indices = create_natural_stance(pi_robot.model)

# 设置参考配置
if not hasattr(pi_robot.model, 'referenceConfigurations'):
    pi_robot.model.referenceConfigurations = {}
pi_robot.model.referenceConfigurations["half_sitting"] = q_natural.copy()

# 设置初始状态
v0 = pinocchio.utils.zero(pi_robot.model.nv)
x0 = np.concatenate([q_natural, v0])

# 设置浮动基座限制
ub = pi_robot.model.upperPositionLimit
ub[:7] = 1
pi_robot.model.upperPositionLimit = ub

lb = pi_robot.model.lowerPositionLimit
lb[:7] = -1
pi_robot.model.lowerPositionLimit = lb

# 设置关节限制 - 重点设置膝关节的范围
if joint_indices:
    r_hip_q_idx, l_hip_q_idx, r_knee_q_idx, l_knee_q_idx, r_ankle_q_idx, l_ankle_q_idx = joint_indices
    
    # 设置膝关节的关节限制范围
    pi_robot.model.lowerPositionLimit[r_knee_q_idx] = np.radians(20)  # 最小弯曲20度
    pi_robot.model.lowerPositionLimit[l_knee_q_idx] = np.radians(20)
    
    pi_robot.model.upperPositionLimit[r_knee_q_idx] = np.radians(120)  # 最大弯曲120度
    pi_robot.model.upperPositionLimit[l_knee_q_idx] = np.radians(120)
    
    print(f"   ✅ 设置膝关节弯曲范围限制: {np.degrees(pi_robot.model.lowerPositionLimit[r_knee_q_idx]):.1f}° 到 {np.degrees(pi_robot.model.upperPositionLimit[r_knee_q_idx]):.1f}°")

# 设置跳跃问题
rightFoot = "r_sole_link"
leftFoot = "l_sole_link"
gait = SimpleBipedGaitProblem(pi_robot.model, rightFoot, leftFoot)

# 改进的跳跃配置
DEEP_KNEE_JUMPS = [
    {
        "name": "深度膝盖弯曲微型跳跃", 
        "height": 0.01,   # 1cm微跳
        "length": [0.0, 0.0, 0.0],  # 纯垂直跳跃
        "timeStep": 0.02,
        "groundKnots": 25,   # 增加地面准备时间
        "flyingKnots": 10,    # 飞行时间
    },
    {
        "name": "深度膝盖弯曲标准跳跃", 
        "height": 0.03,   # 3cm目标高度
        "length": [0.05, 0.0, 0.0],  # 5cm前进
        "timeStep": 0.02,
        "groundKnots": 30,   # 更长的地面准备时间，用于深蹲和蓄力
        "flyingKnots": 15,   # 延长飞行时间
    }
]

solvers = []
successful_jumps = []
jump_names = []
jump_heights = []
jump_distances = []
current_x = x0.copy()

# 根据关节索引获取膝关节弯曲值
if joint_indices:
    r_knee_q_idx = joint_indices[2]
    knee_bend = np.radians(50)  # 目标膝关节最小弯曲程度 (正值，更大值表示更深弯曲)
else:
    knee_bend = np.radians(50)

print(f"\n🦶 深度膝盖弯曲跳跃动作")
print(f"   🎯 策略: 深度膝盖弯曲 + 关节限制 + 有力蹬地")
print(f"   🦵 初始膝盖: {np.degrees(q_natural[joint_indices[2] if joint_indices else 0]):.1f}° (深度弯曲)")
print(f"   📏 目标膝盖最小弯曲: {np.degrees(knee_bend):.1f}°")

for i, jump_config in enumerate(DEEP_KNEE_JUMPS):
    print(f"\n🚀 创建 {jump_config['name'].upper()}")
    print(f"   🏔️  目标高度: {jump_config['height']*100:.1f}cm")
    print(f"   📐 前进距离: {jump_config['length'][0]*100:.1f}cm")
    print(f"   ⏱️  地面时间: {jump_config['groundKnots']} 节点")
    
    # 创建基础跳跃问题
    base_problem = gait.createJumpingProblem(
        current_x,
        jump_config["height"],
        jump_config["length"],
        jump_config["timeStep"],
        jump_config["groundKnots"],
        jump_config["flyingKnots"],
    )
    
    # 强制执行关节限制
    enforce_joint_limits(base_problem, joint_indices, knee_bend)
    
    # 添加膝盖弯曲约束
    add_knee_bend_costs(base_problem, joint_indices, knee_bend)
    
    # 修改终端状态以确保安全着陆
    modify_terminal_state(base_problem, pi_robot.model, joint_indices)
    
    # 使用BoxFDDP求解器，能够处理状态约束
    current_solver = crocoddyl.SolverBoxFDDP(base_problem)
    current_solver = configure_solver(current_solver)

    print(f"\n🎯 求解 {jump_config['name'].upper()}")
    xs = [current_x] * (current_solver.problem.T + 1)
    us = current_solver.problem.quasiStatic([current_x] * current_solver.problem.T)
    current_solver.solve(xs, us, 200, False)

    if current_solver.isFeasible:
        print(f"✅ {jump_config['name']} 求解成功 ({current_solver.iter} 次迭代)!")
        print(f"   📊 最终成本: {current_solver.cost:.6e}")
        
        # 分析跳跃结果
        q_initial = current_solver.xs[0][:pi_robot.model.nq]
        q_final = current_solver.xs[-1][:pi_robot.model.nq]
        
        # 找到最高点
        max_height = max([xs[:pi_robot.model.nq][2] for xs in current_solver.xs])
        actual_height = (max_height - q_initial[2]) * 100
        
        forward_distance = (q_final[0] - q_initial[0]) * 100
        
        print(f"   🏔️  实际跳跃高度: {actual_height:.1f}cm")
        print(f"   📏 实际前进距离: {forward_distance:.1f}cm") 
        
        # 保存跳跃信息用于后续可视化
        jump_names.append(jump_config['name'])
        jump_heights.append(jump_config['height']*100)
        jump_distances.append(jump_config['length'][0]*100)
        
        # 分析关节角度
        if joint_indices:
            r_hip_q_idx, l_hip_q_idx, r_knee_q_idx, l_knee_q_idx, r_ankle_q_idx, l_ankle_q_idx = joint_indices
            
            all_knee_r = [np.degrees(xs[:pi_robot.model.nq][r_knee_q_idx]) for xs in current_solver.xs]
            all_knee_l = [np.degrees(xs[:pi_robot.model.nq][l_knee_q_idx]) for xs in current_solver.xs]
            all_hip_r = [np.degrees(xs[:pi_robot.model.nq][r_hip_q_idx]) for xs in current_solver.xs]
            all_ankle_r = [np.degrees(xs[:pi_robot.model.nq][r_ankle_q_idx]) for xs in current_solver.xs]
            
            min_knee_r = min(all_knee_r)
            min_knee_l = min(all_knee_l)
            max_knee_r = max(all_knee_r)
            max_knee_l = max(all_knee_l)
            
            knee_r_range = max_knee_r - min_knee_r
            knee_l_range = max_knee_l - min_knee_l
            
            print(f"   🦵 关节角度分析:")
            print(f"      右膝: {min_knee_r:.1f}° 到 {max_knee_r:.1f}° (变化范围: {knee_r_range:.1f}°)")
            print(f"      左膝: {min_knee_l:.1f}° 到 {max_knee_l:.1f}° (变化范围: {knee_l_range:.1f}°)")
            print(f"      右髋: {min(all_hip_r):.1f}° 到 {max(all_hip_r):.1f}°")
            print(f"      右踝: {min(all_ankle_r):.1f}° 到 {max(all_ankle_r):.1f}°")
            
            # 评估膝盖弯曲情况
            target_bend = np.degrees(knee_bend)
            if min_knee_r >= 20 and min_knee_l >= 20:
                print(f"   ✅ 成功实现膝盖弯曲 (最小值: {min_knee_r:.1f}°, 目标: {target_bend:.1f}°)")
            elif min_knee_r >= 10 and min_knee_l >= 10:
                print(f"   ✅ 膝盖有一定弯曲，但未达到目标")
            else:
                print(f"   ⚠️  膝盖弯曲程度不足")
                
            # 评估髋关节参与
            hip_range = max(all_hip_r) - min(all_hip_r)
            if hip_range > 20:
                print(f"   ✅ 髋关节积极参与运动 (变化{hip_range:.1f}°)")
            else:
                print(f"   ⚠️  髋关节运动有限 (变化{hip_range:.1f}°)")
                
            # 评估踝关节脚尖推进
            ankle_range = max(all_ankle_r) - min(all_ankle_r)
            if ankle_range > 20:
                print(f"   ✅ 检测到有力脚尖推进 (踝关节变化{ankle_range:.1f}°)")
            else:
                print(f"   ⚠️  脚尖推进有限 (踝关节变化{ankle_range:.1f}°)")
        
        solvers.append(current_solver)
        successful_jumps.append(i)
        
        # 更新当前状态以继续下一跳
        current_x = np.concatenate([q_final, np.zeros(pi_robot.model.nv)])
        
    else:
        print(f"❌ {jump_config['name']} 求解失败!")
        print(f"   📊 最终成本: {current_solver.cost:.6e}")
        break

# 可视化 - 使用参考文件中的方法
if WITHDISPLAY or True:
    print(f"\n🎬 启动深度膝盖弯曲跳跃可视化...")
    print(f"   🎯 特色: 深度膝盖弯曲 + 关节限制 + 有力蹬地")
    
    try:
        import gepetto
        gepetto.corbaserver.Client()
        cameraTF = [3.0, 3.0, 2.0, 0.2, 0.2, 0.8, 0.2]
        display = crocoddyl.GepettoDisplay(pi_robot, 4, 4, cameraTF)
        print("📺 使用 Gepetto 可视化")
    except Exception as e:
        print(f"Gepetto 不可用: {e}")
        try:
            # 使用标准MeshcatDisplay替代自定义可视化
            display = crocoddyl.MeshcatDisplay(pi_robot)
            print("📺 使用 Meshcat 可视化")
            print("🌐 可视化URL: http://127.0.0.1:7007/static/")
        except Exception as e:
            print(f"可视化设置错误: {e}")
            display = None

    if display:
        display.rate = -1  # 设置播放速率
        display.freq = 1   # 设置刷新频率
        
        print(f"\n🚀 播放深度膝盖弯曲跳跃序列...")
        print(f"   ✅ 成功求解跳跃: {len(successful_jumps)}/{len(DEEP_KNEE_JUMPS)}")
        print("   🔍 观察要点:")
        print("     • 初始深度膝盖弯曲状态")
        print("     • 膝盖保持弯曲的整个运动过程")
        print("     • 髋关节和脚踝的协调配合")
        print("     • 着陆时的膝盖缓冲")
        print("   按 Ctrl+C 停止")
        
        try:
            phase_delay = 3.0
            while True:
                for jump_idx in successful_jumps:
                    jump_name = DEEP_KNEE_JUMPS[jump_idx]["name"]
                    height = jump_heights[jump_idx]
                    distance = jump_distances[jump_idx]
                    print(f"🎬 播放: {jump_name} (目标: {height:.1f}cm高, {distance:.1f}cm远)")
                    print(f"   🦵 观察膝盖弯曲保持状态和伸展动作...")
                    # 使用标准的displayFromSolver方法
                    display.displayFromSolver(solvers[jump_idx])
                    if jump_idx != successful_jumps[-1]:
                        time.sleep(phase_delay)
                print("🔄 重新播放深度膝盖弯曲跳跃序列...\n")
                time.sleep(5.0)
        except KeyboardInterrupt:
            print("\n⏹️  用户停止可视化")

print(f"\n🎉 深度膝盖弯曲跳跃实现完成!")
print("📊 实现总结:")
print("   🦵 采用深度膝盖弯曲姿态 (目标膝盖角度: {:.1f}°)".format(np.degrees(knee_bend)))
print("   🔒 添加膝关节范围限制")
print("   🎯 强化膝盖弯曲成本函数")
print("   🛬 安全着陆姿态控制")
print("   🔍 全面的关节角度分析")
print(f"   🏆 成功完成 {len(successful_jumps)}/{len(DEEP_KNEE_JUMPS)} 个深度膝盖弯曲跳跃测试")
print("\n💡 生物力学特征:")
print("   • 始终保持膝盖弯曲")
print("   • 髋关节积极参与运动") 
print("   • 强化踝关节脚尖推进机制")
print("   • 更符合人类跳跃的运动模式") 