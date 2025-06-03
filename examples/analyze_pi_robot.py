#!/usr/bin/env python3
"""
Pi Robot URDF Analysis Tool

This script helps analyze your Pi robot URDF to identify:
- Joint names and types
- Frame names (especially foot frames)
- Robot structure and DOF
- Suggested configuration for the gait demo

Usage: python analyze_pi_robot.py path/to/your/pi_robot.urdf
"""

import sys
import os
import numpy as np
import pinocchio

def analyze_pi_robot(urdf_path):
    """Analyze Pi robot URDF structure"""
    
    if not os.path.exists(urdf_path):
        print(f"Error: URDF file not found at {urdf_path}")
        return None
    
    try:
        # Load the robot model
        model = pinocchio.buildModelFromUrdf(urdf_path, pinocchio.JointModelFreeFlyer())
        data = model.createData()
        
        print("="*60)
        print(f"🤖 Pi Robot Analysis: {os.path.basename(urdf_path)}")
        print("="*60)
        
        # Basic robot information
        print(f"\n📊 Basic Information:")
        print(f"   Robot name: {model.name}")
        print(f"   Number of joints: {model.njoints}")
        print(f"   Number of DOFs: {model.nq}")
        print(f"   Number of velocity DOFs: {model.nv}")
        
        # Joint analysis
        print(f"\n🔗 Joint Analysis:")
        for i, joint_name in enumerate(model.names):
            if i == 0:  # Skip universe joint
                continue
            joint = model.joints[i]
            print(f"   {i}: {joint_name}")
            print(f"      Type: {joint}")
            print(f"      DOF: {joint.nq}")
        
        # Frame analysis
        print(f"\n🎯 Frame Analysis ({len(model.frames)} frames):")
        foot_candidates = []
        for i, frame in enumerate(model.frames):
            frame_name = frame.name.lower()
            print(f"   {i}: {frame.name}")
            
            # Look for potential foot frames
            if any(keyword in frame_name for keyword in ['foot', 'ankle', 'sole', 'end_effector']):
                foot_candidates.append(frame.name)
                print(f"      👟 POTENTIAL FOOT FRAME")
        
        # Suggested foot frames
        print(f"\n👟 Suggested Foot Frames:")
        if foot_candidates:
            left_foot = None
            right_foot = None
            
            for candidate in foot_candidates:
                candidate_lower = candidate.lower()
                if 'left' in candidate_lower or 'l_' in candidate_lower:
                    left_foot = candidate
                elif 'right' in candidate_lower or 'r_' in candidate_lower:
                    right_foot = candidate
            
            print(f"   Left foot: {left_foot or 'Not found automatically'}")
            print(f"   Right foot: {right_foot or 'Not found automatically'}")
            
            if not left_foot or not right_foot:
                print(f"   All candidates: {foot_candidates}")
        else:
            print("   No obvious foot frames found. Check frame names manually.")
        
        # Configuration analysis
        print(f"\n⚙️  Configuration Analysis:")
        q_neutral = pinocchio.neutral(model)
        print(f"   Neutral configuration shape: {q_neutral.shape}")
        print(f"   Configuration bounds:")
        print(f"      Lower: {model.lowerPositionLimit}")
        print(f"      Upper: {model.upperPositionLimit}")
        
        # Generate suggested code
        print(f"\n💡 Suggested Code Configuration:")
        print("="*40)
        
        suggested_left = left_foot if foot_candidates and left_foot else "left_foot_frame_name"
        suggested_right = right_foot if foot_candidates and right_foot else "right_foot_frame_name"
        
        print(f"""
# In your pi_bipedal_gaits.py file, update these variables:
urdf_path = "{urdf_path}"
rightFoot = "{suggested_right}"
leftFoot = "{suggested_left}"

# Suggested gait parameters for your robot:
GAITPHASES = [
    {{
        "walking": {{
            "stepLength": 0.2,      # Conservative step length
            "stepHeight": 0.03,     # Low step height
            "timeStep": 0.05,       # Reasonable time step
            "stepKnots": 10,        # Fewer knots for faster computation
            "supportKnots": 3,
        }}
    }},
    # Add more phases as needed...
]

# For half-sitting configuration, you may need to customize:
def get_half_sitting_configuration(model):
    q0 = pinocchio.neutral(model)
    # Customize joint angles here based on your robot's structure
    # Example:
    # q0[joint_index] = desired_angle_in_radians
    return q0
""")
        
        # 计算重心
        com = pinocchio.centerOfMass(model, data, q_neutral)
        print(f"\n📏 重心位置: x={com[0]:.3f}, y={com[1]:.3f}, z={com[2]:.3f}")
        
        # 检查关节限制
        print("\n🔍 关节限制检查:")
        for i, name in enumerate(model.names[1:]):  # 跳过universe
            if i < len(model.lowerPositionLimit) - 7:  # 排除free flyer
                joint_idx = i + 7  # free flyer占用前7个
                if joint_idx < len(model.lowerPositionLimit):
                    lower = model.lowerPositionLimit[joint_idx]
                    upper = model.upperPositionLimit[joint_idx]
                    effort = model.effortLimit[joint_idx]
                    print(f"   {name}: [{lower:.2f}, {upper:.2f}] rad, effort: {effort:.1f} Nm")
        
        # 检查腿长和机器人尺寸
        print("\n🦵 腿部尺寸估算:")
        # 从URDF中提取的关键尺寸
        hip_to_thigh = 0.06925  # r_thigh_joint origin z
        thigh_to_calf = 0.07025  # r_calf_joint origin z  
        calf_to_ankle = 0.14     # r_ankle_pitch_joint origin z
        total_leg_length = hip_to_thigh + thigh_to_calf + calf_to_ankle
        print(f"腿长估算: {total_leg_length:.3f} m")
        
        # 检查足部尺寸
        print("\n🦶 足部尺寸(从碰撞几何):")
        print("  踝关节碰撞盒: 0.12 x 0.08 x 0.03 m")
        
        # 分析步态参数合理性
        print("\n🚶 当前步态参数分析:")
        current_step_lengths = [0.02, 0.04, 0.06, 0.08]  # 来自我们的GAITPHASES
        current_step_heights = [0.005, 0.01, 0.015, 0.02]
        
        print("步长相对腿长比例:")
        for i, step_len in enumerate(current_step_lengths):
            ratio = step_len / total_leg_length
            status = "✅" if ratio < 0.3 else "⚠️"
            print(f"  阶段{i+1}: {step_len}m ({ratio:.1%} 腿长) {status}")
        
        print("步高相对腿长比例:")
        for i, step_height in enumerate(current_step_heights):
            ratio = step_height / total_leg_length
            status = "✅" if ratio < 0.1 else "⚠️"
            print(f"  阶段{i+1}: {step_height}m ({ratio:.1%} 腿长) {status}")
        
        # 建议的改进参数
        print("\n💡 建议的步态参数改进:")
        leg_length = total_leg_length
        # 更保守的步态参数
        suggested_step_lengths = [leg_length*0.02, leg_length*0.04, leg_length*0.06, leg_length*0.10]
        suggested_step_heights = [leg_length*0.01, leg_length*0.02, leg_length*0.03, leg_length*0.04]
        
        print("建议步长:")
        for i, step_len in enumerate(suggested_step_lengths):
            print(f"  阶段{i+1}: {step_len:.3f}m")
        
        print("建议步高:")
        for i, step_height in enumerate(suggested_step_heights):
            print(f"  阶段{i+1}: {step_height:.3f}m")
        
        # 检查初始姿态
        print("\n🤸 初始姿态分析:")
        print("中性姿态关节角度:")
        joint_names = ['r_hip_pitch', 'r_hip_roll', 'r_thigh', 'r_calf', 'r_ankle_pitch', 'r_ankle_roll',
                       'l_hip_pitch', 'l_hip_roll', 'l_thigh', 'l_calf', 'l_ankle_pitch', 'l_ankle_roll']
        
        for i, name in enumerate(joint_names):
            if i + 7 < len(q_neutral):  # 确保索引不越界
                angle = q_neutral[i + 7]  # 跳过free flyer的7个DOF
                print(f"   {name}: {angle:.3f} rad ({np.degrees(angle):.1f}°)")
        
        # 质量分布分析
        print("\n💪 质量分布分析:")
        total_mass = model.totalMass
        base_mass = 2.1609  # 从URDF读取
        leg_mass_estimate = (total_mass - base_mass) / 2
        print(f"基座质量: {base_mass:.3f} kg ({base_mass/total_mass:.1%})")
        print(f"单腿质量估算: {leg_mass_estimate:.3f} kg ({leg_mass_estimate/total_mass:.1%})")
        
        print(f"\n重心高度相对腿长: {com[2]/leg_length:.1%}")
        if com[2] < 0:
            print("⚠️  警告: 重心在地面以下，可能导致不稳定的步态")
        
        # 步态问题诊断
        print("\n🩺 步态问题诊断:")
        problems = []
        
        if abs(com[0]) > 0.01:
            problems.append(f"重心X偏移过大: {com[0]:.3f}m")
        if abs(com[1]) > 0.01:
            problems.append(f"重心Y偏移过大: {com[1]:.3f}m")
        if com[2] < 0:
            problems.append("重心过低，在地面以下")
        if max(current_step_lengths) / leg_length > 0.3:
            problems.append("步长过大，超过腿长30%")
        if max(current_step_heights) / leg_length > 0.1:
            problems.append("步高过大，超过腿长10%")
        
        if problems:
            print("可能的问题:")
            for problem in problems:
                print(f"  ❌ {problem}")
        else:
            print("  ✅ 未发现明显问题")
        
        print("\n🔧 改善建议:")
        print("1. 调整初始姿态到更稳定的半蹲位置")
        print("2. 减小步长和步高参数")
        print("3. 增加支撑相时间比例")
        print("4. 调整重心位置到合理高度")
        print("5. 检查足部接触点是否正确")
        
        return model
        
    except Exception as e:
        print(f"Error analyzing URDF: {e}")
        return None

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_pi_robot.py <path_to_urdf>")
        print("Example: python analyze_pi_robot.py /path/to/pi_robot.urdf")
        return
    
    urdf_path = sys.argv[1]
    model = analyze_pi_robot(urdf_path)
    
    if model:
        print("\n✅ Analysis complete!")
        print("Now you can update the pi_bipedal_gaits.py file with the suggested configuration.")
    else:
        print("\n❌ Analysis failed. Please check your URDF file.")

if __name__ == "__main__":
    main() 