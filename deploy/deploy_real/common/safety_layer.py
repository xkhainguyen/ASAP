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

# Safety Layer for Unitree Robots

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
        elif self.robot == "h1_2":
            self.leg_joint2motor_idx = [0, 1, 2, 3, 4, 5, 
                                        6, 7, 8, 9, 10, 11]
            self.arm_waist_joint2motor_idx = [12, 
                                              13, 14, 15, 16, 17, 18, 19,
                                              20, 21, 22, 23, 24, 25, 26]

            # TODO: some torque, velocity, position limits for h12?

    def is_falling(self) -> bool:
        # TODO: Check if the robot is falling based on IMU data
        # Implement logic to determine if the robot is falling
        return False
    
    def is_over_ang_vel_limit(self) -> bool:
        # TODO: Check if the robot's angular velocity exceeds a certain limit
        # Implement logic to check angular velocity limits
        return False
    
    def is_over_joint_vel_limit(self) -> bool:
        # TODO: Check if the robot's velocity exceeds a certain limit
        # Implement logic to check velocity limits
        return False

    def is_over_joint_torque_limit(self) -> bool:
        # TODO: Check if the robot's torque exceeds a certain limit
        # Implement logic to check torque limits
        return False
    
    def is_over_joint_position_limit(self) -> bool:
        # TODO: Check if the robot's position exceeds a certain limit
        # Implement logic to check position limits
        return False

    def is_overheating(self) -> bool:
        # TODO: Check if the robot is overheating
        # Implement logic to check temperature limits
        return False

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
        for i in range(dof_size):
            self.qj[i] = low_state.motor_state[dof_idx[i]].q
            self.dqj[i] = low_state.motor_state[dof_idx[i]].dq
            self.tauj[i] = low_state.motor_state[dof_idx[i]].tau_est
        
        # TODO: may read more?

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
        