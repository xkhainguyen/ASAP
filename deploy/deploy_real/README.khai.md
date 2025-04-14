# VinRobotics Reinforcement Learning Gym Full Pipeline

## Policy Description

Please check `deploy_real.py` for more details

### Observation

Some are subtracted by the nominal/default position (deviation) and are scaled

We should also notice this `default_angles`

Be careful of the **motor order** of the policy and that of the SDK:

    H1_2_JointIndex.LeftHipYaw,
    H1_2_JointIndex.LeftHipPitch,
    H1_2_JointIndex.LeftHipRoll,
    H1_2_JointIndex.LeftKnee,
    H1_2_JointIndex.LeftAnklePitch,
    H1_2_JointIndex.LeftAnkleRoll,
    H1_2_JointIndex.RightHipYaw,
    H1_2_JointIndex.RightHipPitch,
    H1_2_JointIndex.RightHipRoll,
    H1_2_JointIndex.RightKnee,
    H1_2_JointIndex.RightAnklePitch,
    H1_2_JointIndex.RightAnkleRoll

```python
# (3) pelvis angular velocity in the (global) Initial frame 
self.obs[:3] = ang_vel
# (3) gravity orientation in pelvis frame 
self.obs[3:6] = gravity_orientation     
# (3) user pelvis vel cmd, currently assume m/s and rad/s, mapping from -1->+1 in joystick to max_cmd, max_cmd might be absolute in m/s
self.obs[6:9] = self.cmd * self.config.cmd_scale * self.config.max_cmd 
# (12) scaled joint pos deviation (rad) 
self.obs[9 : 9 + num_actions] = qj_obs              
# (12) scaled joint vel (rad/s) 
self.obs[9 + num_actions : 9 + num_actions * 2] = dqj_obs    
# (12) previous action (rad) 
self.obs[9 + num_actions * 2 : 9 + num_actions * 3] = self.action
# (1) sin phase, check code for details
self.obs[9 + num_actions * 3] = sin_phase         
# (1) cos phase                      
self.obs[9 + num_actions * 3 + 1] = cos_phase                          
```

Might want to check `legged_gym/envs/base/legged_robot_config.py`

### Action

Output of the policy is some scaled action delta

```python
# Get the action from the policy network  (12)
obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
self.action = self.policy(obs_tensor).detach().numpy().squeeze()
```

Final output is the absolute joint position

```python
# transform action to target_dof_pos (12)
target_dof_pos = self.config.default_angles + self.action * self.config.action_scale
```

They also fix some Kps and Kds for the low-level PD controllers.

## Software-in-the-loop deployment (SDK Mujoco)

First, start the RL controller process

```python
khai@rtx8:~/vr_rl_gym$ python deploy/deploy_real/deploy_real_khai.py lo h1_2.yaml
```

Then, run the SDK Mujoco simulator process

```python
khai@rtx8:~/unitree_mujoco_si/simulate_python$ python unitree_mujoco.py
```

## Real world deployment

This code can deploy the trained network on physical robots. Currently supported robots include Unitree G1, H1, H1_2.

### Khai's Notes

* This policy runs at 50 Hz.
* Make sure to put the robot into Dev Mode `L2+R2` so the default controller doesn't run.

### Startup Usage

```bash
python deploy_real.py {net_interface} {config_name}
```

- `net_interface`: is the name of the network interface connected to the robot, such as `enp3s0`
- `config_name`: is the file name of the configuration file. The configuration file will be found under `deploy/deploy_real/configs/`, such as `g1.yaml`, `h1.yaml`, `h1_2.yaml`.

### Startup Process

#### 1. Start the robot

Start the robot in the lifted state (with crane) and wait for the robot to enter `zero torque mode` (state right after turned on)

#### 2. Enter the debugging mode

Make sure the robot is in `zero torque mode`, press the `L2+R2` key combination of the remote control; the robot will enter the `debugging mode`, and the robot joints are in the damping state (hard joint) in the `debugging mode`.

#### 3. Connect the robot

Use an Ethernet cable to connect your computer to the network port on the robot. Modify the network configuration as follows

<img src="https://doc-cdn.unitree.com/static/2023/9/6/0f51cb9b12f94f0cb75070d05118c00a_980x816.jpg" width="400px">

Then use the `ifconfig` command to view the name of the network interface connected to the robot. Record the network interface name, which will be used as a parameter of the startup command later

<img src="https://oss-global-cdn.unitree.com/static/b84485f386994ef08b0ccfa928ab3830_825x484.png" width="400px">

#### 4. Start the program

Assume that the network card currently connected to the physical robot is named `enp3s0`. Take the G1 robot as an example, execute the following command to start

```bash
python deploy_real.py enp3s0 g1.yaml
```

##### 4.1 Zero torque state

After starting, the robot joints will be in the zero torque state. You can shake the robot joints by hand to feel and confirm.

##### 4.2 Default position state

In the zero torque state, press the `start` button on the remote control, and the robot will move to the default joint position state.

After the robot moves to the default joint position, you can slowly lower the lifting mechanism to let the robot's feet touch the ground.

##### 4.3 Motion control mode

After the preparation is completed, press the `A` button on the remote control, and the robot will step on the spot. After the robot is stable, you can gradually lower the rope to give the robot a certain amount of space to move.

At this time, you can use the joystick on the remote control to control the movement of the robot.
The front and back of the left joystick controls the movement speed of the robot in the x direction
The left and right of the left joystick controls the movement speed of the robot in the y direction
The left and right of the right joystick controls the movement speed of the robot's yaw angle

##### 4.4 Exit control

In motion control mode, press the `select` button on the remote control, the robot will enter the damping mode and fall down, and the program will exit. Or use `ctrl+c` in the terminal to close the program.

> Note:
>
> Since this example deployment is not a stable control program and is only used for demonstration purposes, please try not to disturb the robot during the control process. If any unexpected situation occurs during the control process, please exit the control in time to avoid danger.

### Video tutorial

deploy on G1 robot: [https://oss-global-cdn.unitree.com/static/ea12b7fb3bb641b3ada9c91d44d348db.mp4](https://oss-global-cdn.unitree.com/static/ea12b7fb3bb641b3ada9c91d44d348db.mp4)

deploy on H1 robot: [https://oss-global-cdn.unitree.com/static/d4e0cdd8ce0d477fb5bffbb1ac5ef69d.mp4](https://oss-global-cdn.unitree.com/static/d4e0cdd8ce0d477fb5bffbb1ac5ef69d.mp4)

deploy on H1_2 robot: [https://oss-global-cdn.unitree.com/static/8120e35452404923b854d9e98bdac951.mp4](https://oss-global-cdn.unitree.com/static/8120e35452404923b854d9e98bdac951.mp4)
