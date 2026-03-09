# Lab 3 - Task 2: Trotting Gait Implementation (Explained)

This document explains the complete trotting gait implementation for the quadruped robot, covering every code section with detailed commentary.

---

## Overview

The trotting gait makes the robot walk by moving **diagonal leg pairs in sync**:
- **Pair A**: Right-Front (RF) + Left-Back (LB) — in phase
- **Pair B**: Left-Front (LF) + Right-Back (RB) — offset by half a cycle (0.5)

Each foot traces a **closed triangular path** in 3D space: flat along the ground (stance phase) and a swing up through the air (swing phase).

---

## Step 1: Define the 6 Waypoints (Triangle Shape)

These are the **relative** waypoint positions that define the triangular foot path before any leg offsets are applied. All Y-values are 0 because the triangle lies in the X-Z plane.

```python
touch_down_position = np.array([0.05, 0, -0.14])     # Right edge of base (foot lands)
stand_position_1   = np.array([0.025, 0, -0.14])      # 1/4 across base
stand_position_2   = np.array([0, 0, -0.14])           # Center of base
stand_position_3   = np.array([-0.025, 0, -0.14])     # 3/4 across base
liftoff_position   = np.array([-0.05, 0, -0.14])      # Left edge of base (foot lifts)
mid_swing_position = np.array([0, 0, -0.05])           # Apex of triangle (foot in air)
```

### Triangle geometry:
- **Base** runs along the X-axis at Z = -0.14 (ground level), from X = -0.05 to X = +0.05 → **width = 0.10 m**
- **Apex** is at Z = -0.05 → **height = 0.09 m** above the base
- The foot moves: `touch_down → stand_1 → stand_2 → stand_3 → liftoff → mid_swing → (back to touch_down)`

```
        mid_swing (0, -0.05)
           /\
          /  \
         /    \
        /      \
       /________\
 liftoff        touch_down
(-0.05, -0.14)  (0.05, -0.14)
      ← stand phases →
```

---

## Step 2: Apply Leg Offsets (4 Legs)

Each leg has a different mounting position on the robot body. An offset vector shifts the entire triangle to the correct location for each leg.

```python
# Right-Front leg: mounted at +X, -Y
rf_ee_offset = np.array([0.06, -0.09, 0])
rf_ee_triangle_positions = np.array([
    touch_down_position,
    stand_position_1,
    stand_position_2,
    stand_position_3,
    liftoff_position,
    mid_swing_position,
]) + rf_ee_offset

# Left-Front leg: mounted at +X, +Y
lf_ee_offset = np.array([0.06, 0.09, 0])
lf_ee_triangle_positions = np.array([
    touch_down_position,
    stand_position_1,
    stand_position_2,
    stand_position_3,
    liftoff_position,
    mid_swing_position,
]) + lf_ee_offset

# Right-Back leg: mounted at -X, -Y
rb_ee_offset = np.array([-0.11, -0.09, 0])
rb_ee_triangle_positions = np.array([
    touch_down_position,
    stand_position_1,
    stand_position_2,
    stand_position_3,
    liftoff_position,
    mid_swing_position,
]) + rb_ee_offset

# Left-Back leg: mounted at -X, +Y
lb_ee_offset = np.array([-0.11, 0.09, 0])
lb_ee_triangle_positions = np.array([
    touch_down_position,
    stand_position_1,
    stand_position_2,
    stand_position_3,
    liftoff_position,
    mid_swing_position,
]) + lb_ee_offset
```

### Why these offsets?
| Leg | Offset X | Offset Y | Reason |
|-----|----------|----------|--------|
| RF  | +0.06    | -0.09    | Front-right hip position |
| LF  | +0.06    | +0.09    | Front-left hip position |
| RB  | -0.11    | -0.09    | Back-right hip position |
| LB  | -0.11    | +0.09    | Back-left hip position |

All positions are stored together for easy access:

```python
self.ee_triangle_positions = [
    rf_ee_triangle_positions,   # index 0
    lf_ee_triangle_positions,   # index 1
    rb_ee_triangle_positions,   # index 2
    lb_ee_triangle_positions    # index 3
]
```

---

## Step 3: Interpolate Along the Triangle Path

The `interpolate_triangle` function computes the XYZ foot position at any point along the cycle.

```python
def interpolate_triangle(self, t, leg_index):
    # Phase offset for trotting: diagonal legs in sync
    # RF(0) and LB(3) are in phase, LF(1) and RB(2) are offset by 0.5
    if leg_index == 1 or leg_index == 2:
        t = (t + 0.5) % 1.0

    positions = self.ee_triangle_positions[leg_index]
    n = len(positions)  # n = 6 waypoints

    # Map t ∈ [0, 1) to a segment index and local interpolation factor
    segment = t * n               # e.g., t=0.3 → segment=1.8
    segment_index = int(segment)  # e.g., 1 (we're on segment 1→2)
    if segment_index >= n:
        segment_index = n - 1
    local_t = segment - segment_index  # e.g., 0.8 (80% through this segment)

    start = positions[segment_index]
    end = positions[(segment_index + 1) % n]  # wraps around: 5→0
    return start + local_t * (end - start)     # linear interpolation
```

### How it works:

1. **Phase offset**: LF and RB legs get `t = (t + 0.5) % 1.0`, meaning they are half a cycle ahead of RF and LB. This creates the diagonal trotting pattern.

2. **Segment mapping**: With 6 waypoints, the path has 6 segments (0→1, 1→2, 2→3, 3→4, 4→5, 5→0). The parameter `t ∈ [0, 1)` is scaled to `[0, 6)` to pick the right segment.

3. **Linear interpolation**: Within each segment, `local_t` interpolates linearly between the start and end waypoints.

### Example walkthrough:
| `t` | `segment` | `segment_index` | `local_t` | Interpolating between |
|-----|-----------|-----------------|-----------|----------------------|
| 0.0 | 0.0 | 0 | 0.0 | touch_down → stand_1 (at touch_down) |
| 0.1 | 0.6 | 0 | 0.6 | touch_down → stand_1 (60% through) |
| 0.5 | 3.0 | 3 | 0.0 | stand_3 → liftoff (at stand_3) |
| 0.83 | 5.0 | 5 | 0.0 | mid_swing → touch_down (at mid_swing) |
| 0.99 | 5.94 | 5 | 0.94 | mid_swing → touch_down (94% through) |

---

## Step 4: Inverse Kinematics Solver

Converts a target XYZ foot position into the 3 joint angles needed to reach it.

```python
def get_error_leg(self, theta, desired_position):
    current_position = self.leg_forward_kinematics(theta)
    error = np.linalg.norm(current_position - desired_position)
    return error

def inverse_kinematics_single_leg(self, target_ee, leg_index, initial_guess=[0, 0, 0]):
    self.leg_forward_kinematics = self.fk_functions[leg_index]
    result = scipy.optimize.minimize(self.get_error_leg, initial_guess, args=(target_ee,))
    return result.x
```

### How it works:
- **Objective function** (`get_error_leg`): Computes the Euclidean distance between where the foot currently is (via forward kinematics) and where we want it.
- **Optimizer** (`scipy.optimize.minimize`): Adjusts the 3 joint angles (theta) to minimize this distance.
- **Initial guess**: Uses the previous solution as a starting point for faster convergence — the foot doesn't jump far between timesteps.

---

## Step 5: Cache All Joint Positions

Pre-computes the full gait cycle so the robot can play it back in real-time without solving IK on-the-fly.

```python
def cache_target_joint_positions(self):
    target_joint_positions_cache = []
    target_ee_cache = []
    for leg_index in range(4):
        target_joint_positions_cache.append([])
        target_ee_cache.append([])
        target_joint_positions = [0] * 3   # initial guess: all zeros
        for t in np.arange(0, 1, 0.02):   # 50 steps (0.00, 0.02, ..., 0.98)
            # Get XYZ target for this leg at time t
            target_ee = self.interpolate_triangle(t, leg_index)
            # Solve IK using previous solution as warm start
            target_joint_positions = self.inverse_kinematics_single_leg(
                target_ee, leg_index, initial_guess=target_joint_positions
            )
            target_joint_positions_cache[leg_index].append(target_joint_positions)
            target_ee_cache[leg_index].append(target_ee)

    # Reshape: (4 legs, 50 steps, 3 angles) → (50 steps, 12 angles)
    target_joint_positions_cache = np.concatenate(target_joint_positions_cache, axis=1)
    target_ee_cache = np.concatenate(target_ee_cache, axis=1)

    return target_joint_positions_cache, target_ee_cache
```

### Key details:
- **50 timesteps** per cycle (`np.arange(0, 1, 0.02)`)
- **Warm-starting**: Each IK solve uses the previous step's solution as `initial_guess`, which dramatically improves convergence
- **Final shape**: `(50, 12)` — at each of the 50 timesteps, we have 12 joint angles (3 per leg x 4 legs)
- The cache is stored as `self.target_joint_positions_cache` and `self.target_ee_cache`

---

## Step 6: Playback the Cached Gait

During execution, the robot cycles through the cached positions:

```python
def get_target_joint_positions(self):
    target_joint_positions = self.target_joint_positions_cache[self.counter]
    target_ee = self.target_ee_cache[self.counter]
    self.counter += 1
    if self.counter >= self.target_joint_positions_cache.shape[0]:
        self.counter = 0  # loop back to the start
    return target_ee, target_joint_positions
```

This is called from the IK timer callback at 100 Hz. The PD controller timer at 200 Hz publishes the latest target joint positions to the robot's motors.

---

## Step 7: Validate with Plotting (Playground)

In `lab_3_playground.py`, the gait is validated by plotting the front-right foot's trajectory:

```python
if len(inverse_kinematics.target_ee_cache):
    x_list = []
    z_list = []
    for position in inverse_kinematics.target_ee_cache:
        x_list.append(position[0])   # X coordinate (RF leg)
        z_list.append(position[2])   # Z coordinate (RF leg)
    plt.xlabel('X(m)')
    plt.ylabel('Z(m)')
    plt.title('EE front right foot trot gait')
    plt.plot(x_list, z_list)
    plt.show()
```

**Expected output**: A closed triangle shape in the X-Z plane, confirming the foot follows the correct trajectory.

---

## Summary: Complete Trotting Gait Pipeline

```
1. Define 6 waypoints (triangle in X-Z plane)
         ↓
2. Add leg offsets (position each triangle at the correct hip)
         ↓
3. interpolate_triangle(t, leg_index)
   - Apply phase offset for diagonal pairing
   - Linear interpolation between waypoints
         ↓
4. inverse_kinematics_single_leg(target_ee, leg_index)
   - scipy.optimize.minimize to find joint angles
         ↓
5. cache_target_joint_positions()
   - Pre-compute 50 steps × 4 legs → (50, 12) array
         ↓
6. Playback: cycle through cache at 100 Hz
   - PD controller publishes commands at 200 Hz
```

### Trotting Pattern Visualization

```
Time t=0.0:     RF ↓ ground    LF ↑ swing     RB ↑ swing     LB ↓ ground
Time t=0.25:    RF → stance    LF ↓ landing   RB ↓ landing   LB → stance
Time t=0.5:     RF ↑ swing     LF ↓ ground    RB ↓ ground    LB ↑ swing
Time t=0.75:    RF ↓ landing   LF → stance    RB → stance    LB ↓ landing
```

Diagonal legs (RF+LB and LF+RB) always mirror each other, creating a stable two-point support pattern.
