# Step 1: Add modules to provide access to specific libraries and functions
import os  # Module provides functions to handle file paths, directories, environment variables
import sys  # Module provides access to Python-specific system parameters and functions
import random
import numpy as np
import matplotlib.pyplot as plt  # Visualization

# Step 1.1: (Additional) Imports for Deep Q-Learning
import tensorflow as tf
from tensorflow import keras
from keras import layers

from collections import deque
import random

REPLAY_MEMORY_SIZE = 2000
MIN_REPLAY_SIZE = 100
BATCH_SIZE = 32

replay_buffer = deque(maxlen=REPLAY_MEMORY_SIZE)


# Step 2: Establish path to SUMO (SUMO_HOME)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Step 3: Add Traci module to provide access to specific libraries and functions
import traci  # Static network information (such as reading and analyzing network files)

# Step 4: Define Sumo configuration
Sumo_config = [
    'sumo-gui',  # Use SUMO GUI for visualization
    '-c', 'test.sumocfg',
    '--step-length', '0.10',
    '--delay', '0',
    '--lateral-resolution', '0'
]

# Step 5: Open connection between SUMO and Traci
traci.start(Sumo_config)
#traci.gui.setSchema("View #0", "real world")

# -------------------------
# Step 6: Define Variables
# -------------------------

# Variables for RL State (queue lengths from detectors and current phase)

current_phase = 0

previous_total_delay = 0 

# ---- Reinforcement Learning Hyperparameters ----
TOTAL_STEPS = 1000   # The total number of simulation steps for continuous (online) training.

ALPHA = 0.00001            # Learning rate (α) between[0, 1]    #If α = 1, you fully replace the old Q-value with the newly computed estimate.
                                                            #If α = 0, you ignore the new estimate and never update the Q-value.
GAMMA = 0.99            # Discount factor (γ) between[0, 1]  #If γ = 0, the agent only cares about the reward at the current step (no future rewards).
                                                            #If γ = 1, the agent cares equally about current and future rewards, looking at long-term gains.
EPSILON = 0.1          # Exploration rate (ε) between[0, 1] #If ε = 0 means very greedy, if=1 means very random

ACTIONS = [0, 1]       # The discrete action space (0 = keep phase, 1 = switch phase)



# ---- Additional Stability Parameters ----
MIN_GREEN_STEPS = 100
last_switch_step = -MIN_GREEN_STEPS

# -------------------------
# Step 7: Define Functions
# -------------------------

def Deep_Q_network(state_size, action_size):
    """
    Build a simple feedforward neural network that approximates Q-values.
    """
    '''
    self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
    self.bn1 = nn.BatchNorm2d(16)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
    self.bn2 = nn.BatchNorm2d(32)
    self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
    self.bn3 = nn.BatchNorm2d(32)
    self.head = nn.Linear(448, 2)
    '''
    model = keras.Sequential([
        layers.Input(shape=(128, 128, 4)),  # (128, 128, 4)
        layers.Conv2D(16, kernel_size=8, strides=4, activation='relu'),
        layers.Conv2D(32, kernel_size=4, strides=2, activation='relu'),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(action_size, activation='linear')  # Q-values for each action
    ])
    
    model.compile(
        loss='mse',
        optimizer=keras.optimizers.Adam(learning_rate=0.00001)
    )
    return model

    # Create the DQN model
state_size = 7   # (q_EB_0, q_EB_1, q_EB_2, q_SB_0, q_SB_1, q_SB_2, current_phase)
action_size = len(ACTIONS)
dqn_model = Deep_Q_network(state_size, action_size)

def get_max_Q_value_of_state(state): #1. Objective Function
    Q_values = dqn_model.predict(state, verbose=0)[0]  # shape: (action_size,)
    return np.max(Q_values)

def get_total_cumulative_delay():
    total_delay = 0
    vehicle_ids = traci.vehicle.getIDList()
    for veh_id in vehicle_ids:
        delay = traci.vehicle.getAccumulatedWaitingTime(veh_id)
        total_delay += delay
    return total_delay


def get_reward():
    global previous_total_delay
    current_total_delay = get_total_cumulative_delay()
    reward = previous_total_delay - current_total_delay
    previous_total_delay = current_total_delay
    return reward

def get_state():
    import base64
    import os
    import uuid
    import time

    # Ensure snapshots directory exists
    if not os.path.exists("snapshots"):
        os.makedirs("snapshots")
    
    unique_id = str(uuid.uuid4())
    filename = rf"C:\NHP\UIT_CS106\snapshots\state_snapshot_{unique_id}.png"

    
    print(f"Waiting for file {filename} to be created...")
    traci.gui.screenshot(viewID="View #0", filename=filename)
    timeout = 100  # seconds
    traci.simulationStep()  # Advance simulation by one step

    # start_time = time.time()
    # while not os.path.exists(filename):
    #     if time.time() - start_time > timeout:
    #         raise TimeoutError(f"Timed out waiting for file {filename} to be created.")
    #     time.sleep(0.1)  # wait 100ms

    print("File exists, continuing...")
    
    # print(f"Capturing screenshot to {filename}...") 
    # # Capture screenshot from SUMO GUI
    
    # time.sleep(3)  # Allow time for screenshot to be saved

    print(f"Waiting for file {filename} to be created...")
    # Wait until file actually exists to be safe
    # while not os.path.exists(filename):
    #     time.sleep(3)
    
    # Read and encode image as base64
    with open(filename, "rb") as f:
        image_bytes = f.read()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    print("done")
    return encoded_image




from PIL import Image
import numpy as np
import base64
import io
from collections import deque

frame_height = 128
frame_width = 128
num_frames = 4

# Tạo 4 ảnh 128x128 toàn số 0
frame_history = deque([np.zeros((frame_height, frame_width), dtype=np.uint8) for _ in range(num_frames)],
                      maxlen=num_frames)

def decode_and_preprocess(base64_str):
    # 1. Giải mã từ base64 → ảnh
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data)).convert('L')  # 'L' = Greyscale

    # 2. Resize ảnh về (128, 128)
    image = image.resize((128, 128))

    # 3. Convert ảnh thành numpy array và chuẩn hóa [0, 1]
    img_array = np.array(image, dtype=np.float32) / 255.0

    # 4. Thêm vào bộ nhớ khung hình
    frame_history.append(img_array)

    # 5. Nếu chưa đủ 4 frame, lặp lại frame đầu tiên cho đủ
    while len(frame_history) < 4:
        frame_history.appendleft(frame_history[0])

    # 6. Stack thành tensor đầu vào (128, 128, 4)
    stacked_frames = np.stack(frame_history, axis=-1)  # shape: (128, 128, 4)
    
    return stacked_frames  # ready for CNN input


def apply_action(action, tls_id="clusterJ4_J5_J6"): #5. Constraint 5
    """
    Executes the chosen action on the traffic light, combining:
      - Min Green Time check
      - Switching to the next phase if allowed
    Constraint #5: Ensure at least MIN_GREEN_STEPS pass before switching again.
    """
    global last_switch_step
    
    if action == 0:
        # Do nothing (keep current phase)
        return
    elif action == 1:
        # Check if minimum green time has passed before switching
        if current_simulation_step - last_switch_step >= MIN_GREEN_STEPS:
            program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            num_phases = len(program.phases)
            next_phase = (get_current_phase(tls_id) + 1) % num_phases
            traci.trafficlight.setPhase(tls_id, next_phase)
            # Record when the switch happened
            last_switch_step = current_simulation_step






def train_from_replay():
    if len(replay_buffer) < MIN_REPLAY_SIZE:
        return  # chưa đủ dữ liệu để train

    minibatch = random.sample(replay_buffer, BATCH_SIZE)

    states = np.array([sample[0][0] for sample in minibatch])
    actions = np.array([sample[1] for sample in minibatch])
    rewards = np.array([sample[2] for sample in minibatch])
    next_states = np.array([sample[3][0] for sample in minibatch])
    dones = np.array([sample[4] for sample in minibatch])
    
    # Dự đoán Q hiện tại và Q tương lai
    q_values = dqn_model.predict(states, verbose=0)
    next_q_values = dqn_model.predict(next_states, verbose=0)

    targets = q_values.copy()  # tránh sửa trực tiếp q_values gốc
    for i in range(BATCH_SIZE):
        if dones[i]:
            target_q = rewards[i]
        else:
            target_q = rewards[i] + GAMMA * np.max(next_q_values[i])
        targets[i][actions[i]] = target_q

    dqn_model.fit(states, targets, verbose=0)


def get_action_from_policy(state): #7. Constraint 7
    """
    Epsilon-greedy strategy using the DQN's predicted Q-values.
    """
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    else:
        Q_values = dqn_model.predict(state, verbose=0)[0]
        return int(np.argmax(Q_values))

def get_queue_length(detector_id): #8.Constraint 8
    return traci.lanearea.getLastStepVehicleNumber(detector_id)

def get_current_phase(tls_id): #8.Constraint 8
    return traci.trafficlight.getPhase(tls_id)

# -------------------------
# Step 8: Fully Online Continuous Learning Loop
# -------------------------

# Lists to record data for plotting
step_history = []
reward_history = []
queue_history = []
cur_reward_history = []
cumulative_reward = 0.0

print("\n=== Starting Fully Online Continuous Learning (DQN) ===")
for step in range(TOTAL_STEPS):
    current_simulation_step = step  # keep this variable for apply_action usage
    
    snapshot = get_state()
    # get_state()
    
    
    state = decode_and_preprocess(snapshot)   
    
    state = np.expand_dims(state, axis=0)  # shape: (1, 128, 128, 4)
    # time step thu t 
    print("State shape:", state.shape)
    print("State:", state)
    
    dqn_model.predict(state, verbose=0)
    
    action = get_action_from_policy(state)
    apply_action(action)
    
    # traci.simulationStep()  # Advance simulation by one step
    
    done = traci.simulation.getMinExpectedNumber() == 0
    
    new_snapshot = get_state()
    new_state = decode_and_preprocess(new_snapshot)  
    new_state = np.expand_dims(new_state, axis=0)  # shape: (1, 128, 128, 4)
    
    reward = get_reward()
    cumulative_reward += reward
    
    # 1. Lưu trải nghiệm vào replay buffer
    replay_buffer.append((state, action, reward, new_state, done))

    # 2. Train từ replay buffer nếu đủ dữ liệu
    train_from_replay()

    
    # Print Q-values for the old_state right after update
    updated_q_vals = dqn_model.predict(state, verbose=0)[0]

    # Record data every 100 steps
    if step % 1 == 0:
        updated_q_vals = dqn_model.predict(state, verbose=0)[0]
        print(f"Step {step}, Current_State: {state}, Action: {action}, New_State: {new_state}, Reward: {reward:.2f}, Cumulative Reward: {cumulative_reward:.2f}, Q-values(current_state): {updated_q_vals}")
        step_history.append(step)
        reward_history.append(reward)
        cur_reward_history.append(cumulative_reward)

# -------------------------
# Step 9: Close connection between SUMO and Traci
# -------------------------
traci.close()

# ~~~ Print final model summary (replacing Q-table info) ~~~
print("\nOnline Training completed.")
print("DQN Model Summary:")
dqn_model.summary()

# -------------------------
# Visualization of Results
# -------------------------

# Plot Cumulative Reward over Simulation Steps
plt.figure(figsize=(10, 6))
plt.plot(step_history, cur_reward_history, marker='o', linestyle='-', label="Cumulative Reward")
plt.xlabel("Simulation Step")
plt.ylabel("Cumulative Reward")
plt.title("RL Training (DQN): Cumulative Reward over Steps")
plt.legend()
plt.grid(True)
plt.show()

# Plot Total Queue Length over Simulation Steps
plt.figure(figsize=(10, 6))
plt.plot(step_history, reward_history, marker='o', linestyle='-', label="Reward")
plt.xlabel("Simulation Step")
plt.ylabel("Total Reward")
plt.title("RL Training (DQN): Reward over Steps")
plt.legend()
plt.grid(True)
plt.show()