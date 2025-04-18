from typing import Union
import numpy as np
import time
import torch
import ipdb
import subprocess
import onnxruntime as rt
from collections import deque

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import RemoteController, KeyMap, KeyListener
from common.signal_processing import high_pass_filter, low_pass_filter, rk4_integrate
from config import Config

np.set_printoptions(precision=4, suppress=True)

SINGLE_FRAME = False
LINER_VELOCITY = False

position_limit = {
    "left_hip_pitch_joint": [-2.5307, 2.8798],
    "left_hip_roll_joint": [-0.5236, 2.9671],
    "left_hip_yaw_joint": [-2,7576, 2.7576],
    "left_knee_joint": [-0.087267, 2.8798],
    "left_ankle_pitch_joint": [-0.87267, 0.5236],
    "left_ankle_roll_joint": [-0.2618, 0.2618],

    "right_hip_pitch_joint": [-2.5307, 2.8798],
    "right_hip_roll_joint": [-2.9671, 0.5236],
    "right_hip_yaw_joint": [-2.7576, 2.7576],
    "right_knee_joint": [-0.087267, 2.8798],
    "right_ankle_pitch_joint": [-0.87267, 0.5236],
    "right_ankle_roll_joint": [-0.2618, 0.2618],
    
    "waist_yaw_joint": [-2.618, 2.618],
    "waist_roll_joint": [-0.52, 0.52],
    "waist_pitch_joint": [-0.52, 0.52],
    
    "left_shoulder_pitch_joint": [-3.0892, 2.6704],
    "left_shoulder_roll_joint": [-1.5882, 2.2515],
    "left_shoulder_yaw_joint": [-2.618, 2.618],
    "left_elbow_joint": [-1.0472, 2.0944],

    "left_wrist_roll_joint": [-1.97222, 1.97222],
    "left_wrist_pitch_joint": [-1.61443, 1.61443],
    "left_wrist_yaw_joint": [-1.61443, 1.61443],

    "right_shoulder_pitch_joint": [-3.0892, 2.6704],
    "right_shoulder_roll_joint": [-2.2515, 1.5882],
    "right_shoulder_yaw_joint": [-2.618, 2.618],
    "right_elbow_joint": [-1.0472, 2.0944],

    "right_wrist_roll_joint": [-1.97222, 1.97222],
    "right_wrist_pitch_joint": [-1.61443, 1.61443],
    "right_wrist_yaw_joint": [-1.61443, 1.61443]
}
dof_upper_limit = np.array(
    [position_limit[joint_name][1] for joint_name in position_limit.keys()]
)
dof_lower_limit = np.array(
    [position_limit[joint_name][0] for joint_name in position_limit.keys()]
)
class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()
        self.key_listener = KeyListener()

        # Initialize the policy network
        self.policy = rt.InferenceSession(
            config.policy_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.policy.get_inputs()[0].name
        print(f"policy: {config.policy_path}")
        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.tauj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.ref_motion_phase = 0
        self.config.num_curr_obs = config.num_actions + 3 + config.num_actions*2 + 3 + 1

        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0
        self.update_mode_machine_ = False
        self.real_deploy = False
        self.act_buffer = deque(maxlen=6)
        self.last_time = 0.0

        self.history_length = 4  
        self.lin_vel_buf = np.zeros(3 * self.history_length, dtype=np.float32)
        self.ang_vel_buf = np.zeros(3 * self.history_length, dtype=np.float32)
        self.proj_g_buf = np.zeros(3 * self.history_length, dtype=np.float32)
        self.dof_pos_buf = np.zeros(23 * self.history_length, dtype=np.float32)
        self.dof_vel_buf = np.zeros(23 * self.history_length, dtype=np.float32)
        self.action_buf = np.zeros(23 * self.history_length, dtype=np.float32)
        self.ref_motion_phase_buf = np.zeros(1 * self.history_length, dtype=np.float32)

        self.config.default_angles23 = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, -0.0, 
                                            -0.1, 0.0, 0.0, 0.3, -0.2, -0.0, 
                                            0.0, 0.0, 0.0, 
                                            0.0, 0.0, 0.0, 0.0, 
                                            0.0, 0.0, 0.0, 0.0 ], dtype=np.float32)  #TODO: offset the ankles instead of delta action
        
        self.config.action_joint2motor_idx = np.array([0, 1, 2, 3, 4, 5, 
                                                    6, 7, 8, 9, 10, 11,
                                                    12, 13, 14,
                                                    15, 16, 17, 18, 
                                                    22, 23, 24, 25])
                
        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)
        else:
            raise ValueError("Invalid msg_type")

        # wait for the subscriber to receive data
        print("Waiting for the robot to connect...")
        self.wait_for_low_state()

        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
            
    def LowStateHgHandler(self, msg: LowStateHG):
        if self.update_mode_machine_ == False:
            self.mode_machine_ = self.low_state.mode_machine
            self.update_mode_machine_ = True
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while not self.update_mode_machine_:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")

        # while (self.remote_controller.button[KeyMap.start] != 1):
        create_zero_cmd(self.low_cmd)
        self.send_cmd(self.low_cmd)
        time.sleep(self.config.control_dt)

    def start_default_state(self):
        print("Moving to default state.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)
        
        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        kps = self.config.kps + self.config.arm_waist_kps
        kds = self.config.kds + self.config.arm_waist_kds
        default_pos = np.concatenate((self.config.default_angles, self.config.arm_waist_target), axis=0)
        dof_size = len(dof_idx)
        
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        # move to default pos state
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def keep_default_state(self):
        print("Enter default state.")
        # print("Waiting for the Button A signal...")

        # while self.remote_controller.button[KeyMap.A] != 1:
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i] 
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i] 
            self.low_cmd.motor_cmd[motor_idx].tau = 0
        for i in range(len(self.config.arm_waist_joint2motor_idx)):
            motor_idx = self.config.arm_waist_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i] 
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i] 
            self.low_cmd.motor_cmd[motor_idx].tau = 0
        self.send_cmd(self.low_cmd)
        time.sleep(self.config.control_dt)

    def filter_action(self, action, cutoff=5.0):
        """ Updates the FIFO buffer and filters acceleration if enough data is available. """
        # Add new acceleration sample to FIFO buffer
        self.act_buffer.append(action)

        # Only filter if there are at least 6 samples
        if len(self.act_buffer) > 6:
            act_array = np.array(self.act_buffer)
            filtered_act = low_pass_filter(act_array, cutoff, fs=1/self.config.control_dt)
            return filtered_act[-1]  # Return the most recent filtered value
        else:
            return action  # Return raw value if not enough data
        
    def run(self):
        # OBSERVATION
        self.counter += 1
        # Get the current joint position and velocity
        for i in range(len(self.config.action_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.action_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.action_joint2motor_idx[i]].dq
            self.tauj[i] = self.low_state.motor_state[self.config.action_joint2motor_idx[i]].tau_est

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        from scipy.spatial.transform import Rotation as R
        rotation_quaternion = R.from_euler('y', 0.0).as_quat()  # ('x', angle) creates a rotation quaternion
        rotated_quaternion = R.from_quat(rotation_quaternion) * R.from_quat(quat)
        quat = rotated_quaternion.as_quat()

        ang_vel = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)
           
        # create observation
        gravity_orientation = get_gravity_orientation(quat)
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        qj_obs = (qj_obs - self.config.default_angles23) * self.config.dof_pos_scale
        dqj_obs = dqj_obs * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale
        lin_vel = ang_vel * 0.0

        motion_length = 4.067
        if (self.ref_motion_phase < 1.0): # always in [0, 1]
            # ref_motion_phase += 0.0315  #TODO: compute the phase based on motion length and episode length
            self.ref_motion_phase += 2.0 * self.config.control_dt / motion_length
        else:
            self.ref_motion_phase = 1.0

        if SINGLE_FRAME:
            # 1 single frame  
            current_obs = np.concatenate((
                                        ang_vel,
                                        lin_vel,
                                        qj_obs,
                                        dqj_obs,
                                        gravity_orientation,
                                        np.array([self.ref_motion_phase])
                        ), axis=-1, dtype=np.float32)
            obs_buf = current_obs
        
        elif not SINGLE_FRAME:
            if LINER_VELOCITY:
                # 2 history frames with liner velocity
                history_obs_buf = np.concatenate((self.action_buf, self.ang_vel_buf, self.lin_vel_buf, self.dof_pos_buf, self.dof_vel_buf, self.proj_g_buf, self.ref_motion_phase_buf), axis=-1, dtype=np.float32)
                obs_buf = np.concatenate((self.action, ang_vel, lin_vel, qj_obs, dqj_obs, history_obs_buf, gravity_orientation, np.array([self.ref_motion_phase])), axis=-1, dtype=np.float32)
            elif not LINER_VELOCITY:
                # USING 3 history frames without liner velocity
                history_obs_buf = np.concatenate((self.action_buf, self.ang_vel_buf, self.dof_pos_buf, self.dof_vel_buf, self.proj_g_buf, self.ref_motion_phase_buf), axis=-1, dtype=np.float32)
                obs_buf = np.concatenate((self.action, ang_vel, qj_obs, dqj_obs, history_obs_buf, gravity_orientation, np.array([self.ref_motion_phase])), axis=-1, dtype=np.float32)
            else:
                assert False
        else: assert False

        # ACTION
        # Get the action from the policy network
        onnx_pred = self.policy.run(None, {self.input_name: [obs_buf]})[0][0]
        self.action = onnx_pred
        self.action = self.filter_action(self.action, cutoff=5.0)
        
        # transform action to target_dof_pos
        all_dof_actions = np.zeros(29) # hardware order
        all_dof_actions[self.config.action_joint2motor_idx] = self.config.default_angles23 + self.action * self.config.action_scale

        all_dof_actions = np.clip(all_dof_actions.copy(), dof_lower_limit*0.98,  dof_upper_limit*0.98)

        self.target_leg_pos = all_dof_actions[self.config.leg_joint2motor_idx]
        self.target_upper_pos = all_dof_actions[self.config.arm_waist_joint2motor_idx]

        # Build low cmd
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = self.target_leg_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i] * 1.05
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i] * 1.0
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        for i in range(len(self.config.arm_waist_joint2motor_idx)):
            motor_idx = self.config.arm_waist_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = self.target_upper_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i] * 1.05
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i] * 1.0
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # send the command
        self.send_cmd(self.low_cmd)

        # update history, push the latest to the front, drop the oldest from the end
        self.ang_vel_buf = np.concatenate((ang_vel, self.ang_vel_buf[:-3]), axis=-1, dtype=np.float32)
        self.lin_vel_buf = np.concatenate((lin_vel, self.lin_vel_buf[:-3]), axis=-1, dtype=np.float32)
        self.proj_g_buf = np.concatenate((gravity_orientation, self.proj_g_buf[:-3] ), axis=-1, dtype=np.float32)
        self.dof_pos_buf = np.concatenate((qj_obs, self.dof_pos_buf[:-23] ), axis=-1, dtype=np.float32)
        self.dof_vel_buf = np.concatenate((dqj_obs, self.dof_vel_buf[:-23] ), axis=-1, dtype=np.float32)
        self.action_buf = np.concatenate((self.action, self.action_buf[:-23] ), axis=-1, dtype=np.float32)
        self.ref_motion_phase_buf = np.concatenate((np.array([self.ref_motion_phase]), self.ref_motion_phase_buf[:-1]), axis=-1, dtype=np.float32)   
        
        time.sleep(self.config.control_dt * 0.94) # run every control_dt seconds

        # """Callback function to handle incoming messages."""
        # current_time = time.time()
        # read_dt = current_time - self.last_time
        # self.last_time = current_time

        # if read_dt > 0:
        #     print(f"Published message at {1/read_dt:.2f} Hz")  # Print the actual receiving fr
            
    def handle_key_cmd(self):
        # self.cmd[0] = self.remote_controller.ly
        # self.cmd[1] = self.remote_controller.lx * -1
        # # # self.cmd[0] = -1
        # self.cmd[2] = self.remote_controller.rx * -1
        
        if (self.key_listener.is_pressed('w')):
            self.cmd[0] += 0.1
            print("[MODE] New command: ", self.cmd)
        if (self.key_listener.is_pressed('s')):
            self.cmd[0] -= 0.1
            print("[MODE] New command: ", self.cmd)
        if (self.key_listener.is_pressed('a')):
            self.cmd[1] += 0.1   
            print("[MODE] New command: ", self.cmd)
        if (self.key_listener.is_pressed('d')):
            self.cmd[1] -= 0.1
            print("[MODE] New command: ", self.cmd)
        if (self.key_listener.is_pressed('q')):
            self.cmd[2] += 0.1   
            print("[MODE] New command: ", self.cmd)
        if (self.key_listener.is_pressed('e')):
            self.cmd[2] -= 0.1
            print("[MODE] New command: ", self.cmd)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="g1.yaml")
    args = parser.parse_args()

    # Load config
    config_path = f"./deploy/deploy_real/configs/{args.config}"
    config = Config(config_path)

    real_deploy = False if args.net == "lo" else True

    # Initialize DDS communication
    print("Initializing DDS communication...")
    channel = 1 if real_deploy is False else 0
    ChannelFactoryInitialize(channel, args.net)
    print("DDS communication initialized.")

    controller = Controller(config)

    if real_deploy:
        # Enter the zero torque state, press the start key to continue executing
        controller.zero_torque_state()

        # Gradually move to the default state
        controller.start_default_state()

        # Keep the robot at default state, press the A key to continue executing
        controller.keep_default_state()

    while True:
        try:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break

    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
