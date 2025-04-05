import pybullet as p
p.connect(p.DIRECT)
robot = p.loadURDF("/home/king/Isaac/xArm_Ctrl/urdf/xarm/xarm6_robot_white.urdf")
for i in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, i)
    print(f"Joint {i}: {info[1].decode('utf-8')}, Type={info[2]}, Limits={info[8:10]}")
p.disconnect()