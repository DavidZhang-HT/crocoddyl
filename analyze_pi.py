import pinocchio
import numpy as np

# Load Pi robot
urdf_path = 'pi_robot/urdf/pi_robot_fixed.urdf'
model = pinocchio.buildModelFromUrdf(urdf_path, pinocchio.JointModelFreeFlyer())

print('🤖 Pi机器人参数分析:')
total_mass = sum([model.inertias[i].mass for i in range(1, model.njoints)])
print(f'总质量: {total_mass:.3f} kg')
print(f'自由度: {model.nq}')

# 计算重心
q0 = pinocchio.neutral(model)
data = model.createData()
com = pinocchio.centerOfMass(model, data, q0)
print(f'重心位置: x={com[0]:.3f}, y={com[1]:.3f}, z={com[2]:.3f}')

# 腿长估算
hip_to_thigh = 0.06925
thigh_to_calf = 0.07025  
calf_to_ankle = 0.14
total_leg_length = hip_to_thigh + thigh_to_calf + calf_to_ankle
print(f'腿长估算: {total_leg_length:.3f} m')

# 当前步态参数
current_step_lengths = [0.02, 0.04, 0.06, 0.08]
current_step_heights = [0.005, 0.01, 0.015, 0.02]

print('\n🚶 当前步态参数分析:')
print('步长相对腿长比例:')
for i, step_len in enumerate(current_step_lengths):
    ratio = step_len / total_leg_length
    status = "✅" if ratio < 0.2 else "⚠️"
    print(f'  阶段{i+1}: {step_len}m ({ratio:.1%} 腿长) {status}')

# 问题诊断
print('\n🩺 可能的问题:')
problems = []

if abs(com[0]) > 0.01:
    problems.append(f"重心X偏移: {com[0]:.3f}m")
if abs(com[1]) > 0.01:
    problems.append(f"重心Y偏移: {com[1]:.3f}m")
if com[2] < -0.05:
    problems.append(f"重心过低: {com[2]:.3f}m")
if max(current_step_lengths) / total_leg_length > 0.25:
    problems.append("步长过大")

for problem in problems:
    print(f'  ❌ {problem}')

if not problems:
    print('  ✅ 未发现明显问题')

# 建议改进
print('\n💡 改善建议:')
print('1. 调整初始姿态 - 设置半蹲位置')
print('2. 减小步长和步高')
print('3. 增加支撑时间')
print('4. 检查足部接触点')

# 建议的新参数
print('\n🔧 建议的新步态参数:')
suggested_lengths = [0.005, 0.01, 0.015, 0.025]
suggested_heights = [0.003, 0.005, 0.008, 0.012]

print('建议步长:')
for i, length in enumerate(suggested_lengths):
    print(f'  阶段{i+1}: {length:.3f}m')

print('建议步高:')
for i, height in enumerate(suggested_heights):
    print(f'  阶段{i+1}: {height:.3f}m') 