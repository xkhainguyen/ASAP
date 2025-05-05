from urdfpy import URDF

path="/home/cookie/VR/unitree_mujoco_si/unitree_robots/g1/g1_29dof.urdf"
robot = URDF.load(path)


for joint in robot.joints:
    if joint.limit:
        with open("joint_limits.tsv", "a") as f:
            f.write(f"{joint.name}\t{joint.limit.lower}\t{joint.limit.upper}\t{joint.limit.effort}\t{joint.limit.velocity}\n")
    else:
        continue