from typing import Union
import numpy as np
import time
from collections import deque
from scipy.spatial.transform import Rotation as R

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.signal_processing import high_pass_filter, low_pass_filter, rk4_integrate

def logger(msg):
    # write to a log file
    pass

class SafetyLayer:
    def __init__(self, robot: str):
        self.robot = robot
        
        if self.robot == "g1":
            self.leg_joint2motor_idx = [0, 1, 2, 3, 4, 5, 
                                        6, 7, 8, 9, 10, 11]
            self.arm_waist_joint2motor_idx = [12, 13, 14, 
                                              15, 16, 17, 18, 19, 20, 21, 
                                              22, 23, 24, 25, 26, 27, 28]
            
            # TODO: some torque, velocity, position limits for g1?
            # max_speed 2m/s
            # 

        elif self.robot == "h1_2":
            self.leg_joint2motor_idx = [0, 1, 2, 3, 4, 5, 
                                        6, 7, 8, 9, 10, 11]
            self.arm_waist_joint2motor_idx = [12, 
                                              13, 14, 15, 16, 17, 18, 19,
                                              20, 21, 22, 23, 24, 25, 26]

            # TODO: some torque, velocity, position limits for h12?
        self.init_safety_threshold()

    def init_safety_threshold(self):
        self.max_tilt_angle = np.radians(120)  # 30 degrees
        self.max_ang_vel = np.array([3.0, 3.0, 3.0])  # rad/s
        self.max_temp = 60.0  # Celsius
        self.max_acc_g = 2.0 
        self.max_joint_torque = 120 #Nm
        self.max_joint_vel = 10
        self.max_joint_position = np.pi
        self.max_up_z = 0.45

    def is_falling(self) -> bool:
        if self.quat is None:
            return False
        rot = R.from_quat(self.quat)
        body_z_world = rot.apply([0, 0, 1])  
        up_z = body_z_world[2] 
        gravity = [0, 0, -1]  
        cos_theta = np.clip(np.dot(body_z_world, gravity) / np.linalg.norm(body_z_world), -1.0, 1.0)
        tilt_angle = np.arccos(cos_theta)
        falling_due_to_tilt = tilt_angle > self.max_tilt_angle
        falling_due_to_z = np.abs(up_z) < self.max_up_z
        is_falling = falling_due_to_tilt | falling_due_to_z
        print(f"Is_falling: {is_falling} (tilt_angle: {np.degrees(tilt_angle):.2f} deg, up_z: {up_z:.2f})")
        return is_falling
    
    def is_over_ang_vel_limit(self) -> bool:
        if self.omega is None:
            return False
        print(f"Omega: {self.omega} ")
        return np.any(np.abs(self.omega) > self.max_ang_vel)
    
    def is_over_joint_vel_limit(self) -> bool:
        pass

    def is_over_joint_torque_limit(self) -> bool:
        if self.tauj is None:
            return False
        print(f"Tauq: {self.tauj}")
        return np.any(np.abs(self.tauj) > self.max_joint_torque)
        
    def is_over_joint_position_limit(self) -> bool:
        pass
        # return np.any((self.qj < self.joint_position_limits_min) | (self.qj > self.joint_position_limits_max))

    def is_overheating(self) -> bool:
        pass


    def read_low_state(self, low_state: LowStateHG):
        self.quat = low_state.imu_state.quaternion # quaternion
        self.omega = np.array(low_state.imu_state.gyroscope, dtype=np.float32) # angular velocity
        self.acc = np.array(low_state.imu_state.accelerometer, dtype=np.float32) # acceleration

        dof_idx = self.leg_joint2motor_idx + self.arm_waist_joint2motor_idx
        dof_size = len(dof_idx)
        # record the current pos
        self.qj = np.zeros(dof_size)
        self.dqj = np.zeros(dof_size)
        self.tauj = np.zeros(dof_size)
        self.motor_temps = np.zeros(dof_size, dtype=object)
        for i in range(dof_size):
            self.qj[i] = low_state.motor_state[dof_idx[i]].q
            self.dqj[i] = low_state.motor_state[dof_idx[i]].dq
            self.tauj[i] = low_state.motor_state[dof_idx[i]].tau_est
            self.motor_temps[i] = low_state.motor_state[dof_idx[i]].temperature

        print(f"Motor temperatures: {self.motor_temps}")
    def run(self, low_state: LowStateHG, low_cmd: LowCmdHG):
        # PLACE THIS RIGHT BEFORE SEND_CMD()
        # TODO: Check for safety conditions
        # Implement safety checks 
        # input low_state 
        # output low_cmd
        # check for joint limits, torque limits, orientation, etc.
        # Example: Check if the robot is falling
        safe = True

        self.read_low_state(low_state)
        

        if self.is_falling():
            print("Robot is falling!")            
            safe = False
        if self.is_over_ang_vel_limit():
            print("Robot is over angular velocity limit!")
            safe = False
        if self.is_over_joint_vel_limit():
            print("Robot is over joint velocity limit!")
            safe = False
        if self.is_over_joint_torque_limit():
            print("Robot is over joint torque limit!")
            safe = False
        if self.is_over_joint_position_limit():
            print("Robot is over joint position limit!")
            safe = False
        if self.is_overheating():
            print("Robot is overheating!")
            safe = False

        # TODO: MAYBE MORE?
        
        if not safe:
            # If not safe, set low_cmd to zero or damping command, assuming init_cmd done
            create_damping_cmd(low_cmd)
        return low_cmd