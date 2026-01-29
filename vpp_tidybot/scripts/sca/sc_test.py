import pybullet as p
import pybullet_data

# 1. Connect to PyBullet (use DIRECT to avoid opening a GUI)
physicsClient = p.connect(p.DIRECT)

# 2. (Optional) Tell PyBullet where to look for common URDFs
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 3. Load your Franka URDF with self‐collision enabled
#    - useFixedBase=True keeps the robot base stationary
#    - flags=p.URDF_USE_SELF_COLLISION tells PyBullet to check collisions between all non‐adjacent links
franka_id = p.loadURDF(
    "panda/panda.urdf",
    useFixedBase=True,
    flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
)
p.setCollisionFilterPair(franka_id, franka_id, 4, 6, enableCollision=0)
# 4. (Optional) Move joints into some configuration that might cause self‐contact.
#    For example, set joint 2 to 90 degrees (indexing joints starts at 0):
joints = [0, 1, 2, 3, 4, 5, 6]  # adjust indices if needed
target_positions = [-0.756, 1.19, 2.35, -2.09, -1.99, 0.149, -2.22]
for i, q in zip(joints, target_positions):
    p.resetJointState(franka_id, i, q)

# 5. Step the simulation once so contact checks are updated
p.stepSimulation()

# 6. Query for any contact points where bodyA == bodyB == franka_id
contacts = p.getContactPoints(bodyA=franka_id, bodyB=franka_id)

if len(contacts) > 0:
    print("❌ Self‐collision detected:")
    for c in contacts:
        linkA = c[3]  # link index on A
        linkB = c[4]  # link index on B
        # c[5], c[6], c[7] etc. contain contact position/normal, if you need them
        print(f"    Link {linkA} ↔ Link {linkB}")
else:
    print("✅ No self‐collision.")

# 7. Disconnect when done
p.disconnect()
