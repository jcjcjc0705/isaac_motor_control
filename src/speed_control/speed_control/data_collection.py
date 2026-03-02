import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import numpy as np
import pandas as pd
import random
import math
import time
import os
from scipy.signal import chirp
from scipy.interpolate import interp1d
from .config import DEFAULT_CONFIG as cfg

REAL_HARD_LIMIT = 1.57     # 事後統計用的真實極限 (90度)
ALGORITHM_SAFE_LIMIT = 1.2 # 生成器使用的「假牆壁」，預留了很大的物理滑行空間 (約70度)
VEL_TO_POS_GAIN = 0.7    
MAX_SLEW_RATE = 4.0 
ABORT_LIMIT = 1.5 # 超過這個角度就直接進入 RESET 階段 (約86度)     

def simulate_integral(u, dt):
    return np.cumsum(u) * dt * VEL_TO_POS_GAIN

def apply_linear_scaling_for_safety(u, dt):
    pos = simulate_integral(u, dt)
    max_drift = np.max(np.abs(pos))
    SAFE_LIMIT = ALGORITHM_SAFE_LIMIT - 0.05 
    
    if max_drift > SAFE_LIMIT:
        scale_factor = SAFE_LIMIT / max_drift
        u = u * scale_factor
    return u

def make_prbs(length, dt, amp, step_range, sign):
    u = np.zeros(length)
    idx = 0
    sim_pos = 0.0
    min_s, max_s = int(step_range[0]), int(step_range[1])
    SAFE_LIMIT = ALGORITHM_SAFE_LIMIT - 0.05
    current_sign = sign
    
    while idx < length:
        current_amp = np.random.uniform(0.3 * amp, amp)
        current_val = current_sign * current_amp
        
        if current_val > 0:
            space_left = SAFE_LIMIT - sim_pos
        else:
            space_left = sim_pos - (-SAFE_LIMIT)
            
        max_safe_steps = max(1, int(space_left / (abs(current_val) * VEL_TO_POS_GAIN * dt)))
        actual_max_s = min(max_s, max_safe_steps)
        actual_min_s = min(min_s, actual_max_s) 
        
        hold_step = np.random.randint(actual_min_s, actual_max_s + 1)
        end = min(idx + hold_step, length)
        u[idx:end] = current_val
        
        sim_pos += current_val * (end - idx) * dt * VEL_TO_POS_GAIN
        current_sign *= -1  
        idx = end
        
    return u

def make_ramps(length, dt, amp, freq_range, sign):
    u = np.zeros(length)
    t_step = 0
    current_sign = sign
    last_val = 0.0  
    
    while t_step < length:
        current_freq = np.random.uniform(freq_range[0], freq_range[1])
        period_steps = max(1, int((1.0 / (2.0 * current_freq)) / dt))
        end = min(t_step + period_steps, length)
        
        current_amp = np.random.uniform(0.3 * amp, amp)
        target_val = current_sign * current_amp
        
        u[t_step:end] = np.linspace(last_val, target_val, end - t_step)
        
        last_val = target_val
        t_step = end
        current_sign *= -1 
        
    return apply_linear_scaling_for_safety(u, dt)

def make_chirp(length, dt, amp, freq_range, sign):
    t = np.arange(0, length * dt, dt)[:length]
    u = sign * amp * chirp(t, f0=freq_range[0], t1=t[-1], f1=freq_range[1], method='linear')
    return apply_linear_scaling_for_safety(u, dt)

def make_multisine(length, dt, amp, *args):
    t = np.arange(length) * dt; u = np.zeros(length); current_max_amp = 0.0
    for min_f, max_f in [(0.5, 1.5), (1.5, 3.0), (3.0, 6.0)]:
        u += random.uniform(0.5, 1.5) * (np.sin if random.choice([True, False]) else np.cos)(random.uniform(min_f, max_f) * t + random.uniform(0, 2 * np.pi))
        current_max_amp += 1.5
    u = (u / current_max_amp) * amp if current_max_amp > 0 else u
    return apply_linear_scaling_for_safety(u, dt)

def make_smooth_noise(length, dt, amp, *args):
    num_points = length // int(0.5 / dt) + 2
    x_key = np.linspace(0, length, num_points)
    magnitudes = np.random.uniform(0.1 * amp, amp, num_points)
    signs = np.ones(num_points)
    signs[1::2] = -1
    key_points = magnitudes * signs
    u = np.clip(interp1d(x_key, key_points, kind='cubic', fill_value="extrapolate")(np.arange(length)), -amp, amp)
    
    return apply_linear_scaling_for_safety(u, dt)

def apply_slew_rate_limiter(signal, limit):
    smoothed = np.zeros_like(signal)
    last_val = signal[0]
    smoothed[0] = last_val
    for i in range(1, len(signal)):
        delta = np.clip(signal[i] - last_val, -limit, limit)
        smoothed[i] = last_val + delta
        last_val = smoothed[i]
    return smoothed

# ==========================================
# 排程邏輯
# ==========================================

def get_dynamic_levels(n_episodes):
    if n_episodes < 2: return [1]
    levels = [1]; current = 2
    while current <= int(math.sqrt(n_episodes)):
        levels.append(current); current *= 2
    return sorted(levels, reverse=True)

def generate_type_schedule(sig_type, target_episodes, global_params):
    schedule = []; levels = get_dynamic_levels(target_episodes)
    amp_levels = [2.0, 4.0, 6.0, 8.0]; signs = [1, -1]
    full_range = global_params.get(sig_type, [0.1, 1.0])

    while len(schedule) < target_episodes:
        for amp in amp_levels:
            for n_splits in levels:
                edges = np.linspace(full_range[0], full_range[1], n_splits + 1)
                if sig_type == "PRBS" and len(np.unique(edges.astype(int))) < 2: 
                    edges = np.array([full_range[0], full_range[1]])
                for i in range(len(edges)-1):
                    for s in signs:
                        schedule.append((sig_type, amp, (edges[i], edges[i+1]), s))
    return schedule[:target_episodes]

def create_train_schedule(total_episodes, mix_config, global_ranges):
    counts = {name: int(total_episodes * ratio) if item != mix_config[-1] else total_episodes - sum(int(total_episodes * r) for n, r in mix_config[:-1]) for item, (name, ratio) in zip(mix_config, mix_config)}
    schedules = {name: generate_type_schedule(name, c, global_ranges) for name, c in counts.items()}
    return [schedules[name][i] for i in range(max(len(s) for s in schedules.values()) if schedules else 0) for name, _ in mix_config if i < len(schedules.get(name, []))]

def build_signal_from_info(config, info):
    length = config.episode_len; dt = config.dt
    sig_type, amp, param_range, sign = info
    
    if sig_type == "PRBS":
        raw = make_prbs(length, dt, amp, param_range, sign)
    elif sig_type == "RAMPS":
        raw = make_ramps(length, dt, amp, param_range, sign)
    elif sig_type == "CHIRP":
        raw = make_chirp(length, dt, amp, param_range, sign)
    elif sig_type == "MULTISINE":
        raw = make_multisine(length, dt, amp, param_range, sign)
    elif sig_type == "SMOOTH_NOISE":
        raw = make_smooth_noise(length, dt, amp, param_range, sign)
    else:
        raw = np.zeros(length)
        
    return apply_slew_rate_limiter(raw, MAX_SLEW_RATE)

class IsaacDataCollector(Node):
    def __init__(self, mode='train'):
        super().__init__('isaac_data_collector')
        
        # 統一使用外部載入的 config
        self.cfg = cfg
        self.mode = mode

        if self.mode == 'test':
            print(f"\n[Mode] 測試模式")
            # 覆蓋為測試需要的短回合長度
            self.cfg.total_episodes = 10
            self.cfg.episode_len = 1000
            self.cfg.reset_len = 100
            self.episode_schedule = [("SMOOTH_NOISE", 10.0, None, 1)] *10
            self.output_filename = os.path.join(self.cfg.save_dir, "test.csv")
        else:
            print(f"\n[Mode] 訓練資料蒐集模式")
            self.episode_schedule = create_train_schedule(
                self.cfg.total_episodes, 
                self.cfg.signal_to_mix, 
                self.cfg.signal_ranges
            )
            self.output_filename = self.cfg.save_path

        self.state = "RUN"
        self.is_aborted = False
        self.current_episode = 0
        self.step_in_phase = 0
        self.global_step_counter = 0

        self.hard_limit_count = 0
        
        self.current_signal = build_signal_from_info(self.cfg, self.episode_schedule[0]) if self.episode_schedule else np.zeros(self.cfg.episode_len)
        self.history_cmd = []
        self.history_state = []
        self.current_pos = 0.0
        self.current_joint_pos = 0.0
        self.current_joint_vel = 0.0

        self.sub = self.create_subscription(JointState, '/joint_states', self.listener_callback, 10)
        self.pub = self.create_publisher(Float64, '/motor_control', 10)
        self.timer = self.create_timer(self.cfg.dt, self.timer_callback)
        self.start_time = self.get_clock().now().nanoseconds
        print(f"--- 開始蒐集 ({len(self.episode_schedule)} 回合) ---")

    def get_time_sec(self): 
        return (self.get_clock().now().nanoseconds - self.start_time) / 1e9

    def listener_callback(self, msg):
        current_t = self.get_time_sec()
        found = False
        p1 = v1 = p2 = v2 = 0.0
        
        if "Motor" in msg.name: 
            idx = msg.name.index("Motor")
            p1, v1 = msg.position[idx], msg.velocity[idx]
            self.current_pos = p1
            found = True
        if "Joint" in msg.name: 
            idx = msg.name.index("Joint")
            p2, v2 = msg.position[idx], msg.velocity[idx]
            self.current_joint_pos = p2
            self.current_joint_vel = v2

            if abs(p2) > REAL_HARD_LIMIT:
                self.hard_limit_count += 1
            
        if found: 
            self.history_state.append([current_t, p1, v1, p2, v2])

    def timer_callback(self):
        if self.current_episode >= self.cfg.total_episodes: 
            self.finish_and_save()
            return
            
        actual_t = self.get_time_sec()
        t_ideal = self.global_step_counter * self.cfg.dt
        u_val = 0.0

        recorded_episode = self.current_episode    
        if self.state == "RUN":
            if not self.is_aborted:
                if self.step_in_phase < len(self.current_signal):
                    signal_u = float(self.current_signal[self.step_in_phase])
                else:
                    signal_u = 0.0

                lookahead = 0.1
                predicted_pos = self.current_joint_pos + (self.current_joint_vel * lookahead)
                
                if abs(self.current_joint_pos) > ABORT_LIMIT or abs(predicted_pos) > REAL_HARD_LIMIT:
                    self.is_aborted = True
            
            if self.is_aborted:
                kp = 5.0
                target_vel = -kp * self.current_pos
                u_val = np.clip(target_vel, -1.5, 1.5)
                if abs(self.current_pos) < 0.02:
                    u_val = 0.0
            else:
                u_val = signal_u
            
            self.send_command(u_val)
            self.step_in_phase += 1
            
            if self.step_in_phase >= self.cfg.episode_len: 
                self.state = "RESET"
                self.step_in_phase = 0
                self.is_aborted = False

        elif self.state == "RESET":
            kp = 5.0
            target_vel = -kp * self.current_pos
            u_val = np.clip(target_vel, -1.5, 1.5)
            if abs(self.current_pos) < 0.02:
                u_val = 0.0
            
            self.send_command(u_val)
            self.step_in_phase += 1
            
            if self.step_in_phase >= self.cfg.reset_len:
                self.state = "RUN"
                self.step_in_phase = 0
                self.current_episode += 1
                
                if self.current_episode < len(self.episode_schedule):
                    self.current_signal = build_signal_from_info(self.cfg, self.episode_schedule[self.current_episode])
                if self.current_episode % 10 == 0:
                    total_data_points = len(self.history_cmd)
                    print(f"進度: {self.current_episode}/{self.cfg.total_episodes} 回合完成 | "
                          f"目前總資料數: {total_data_points} | "
                          f"累積超限數: {self.hard_limit_count}")

        self.history_cmd.append([actual_t, t_ideal, u_val, recorded_episode])
        self.global_step_counter += 1

    def send_command(self, u): 
        msg = Float64()
        msg.data = float(u)
        self.pub.publish(msg)
        
    def finish_and_save(self): 
        print("\n--- 蒐集結束，正在處理數據 ---")
        self.send_command(0.0)
        self.process_data()
        raise SystemExit

    def process_data(self):
        cmd_data = np.array(self.history_cmd)
        state_data = np.array(self.history_state)
        
        if len(state_data) == 0: 
            return print("錯誤：沒有收到任何狀態數據！")
        
        t_state = state_data[:, 0]
        y_state = state_data[:, 1:]
        
        _, unique_idx = np.unique(t_state, return_index=True)
        t_state = t_state[unique_idx]
        y_state = y_state[unique_idx]
        
        y_aligned = interp1d(t_state, y_state, axis=0, kind='linear', fill_value="extrapolate")(cmd_data[:, 0])
        
        df = pd.DataFrame({
            "time_actual": cmd_data[:, 0], 
            "time_ideal": cmd_data[:, 1], 
            "episode_id": cmd_data[:, 3], 
            "input_u": cmd_data[:, 2], 
            "pos_motor": y_aligned[:, 0], 
            "vel_motor": y_aligned[:, 1], 
            "pos_joint": y_aligned[:, 2], 
            "vel_joint": y_aligned[:, 3]
        })
        
        exceed_mask = np.abs(df["pos_joint"]) > REAL_HARD_LIMIT
        hard_exceed_count = np.sum(exceed_mask)
        failed_episodes = df[exceed_mask]["episode_id"].unique().astype(int).tolist()
        
        os.makedirs(self.cfg.save_dir, exist_ok=True)
        df.to_csv(self.output_filename, index=False)
        
        print("\n==========================================")
        print(f"✅ 成功儲存數據至: {self.output_filename}")
        print("------------------------------------------")
        print("🛡️  邊界統計報告:")
        print(f"   ➤ 實際超出 {REAL_HARD_LIMIT} rad 的資料筆數 : {hard_exceed_count} 筆 ({hard_exceed_count/len(df)*100:.2f}%)")
        if hard_exceed_count > 0: 
            print(f"   ➤ 發生超限的回合 (Episode ID) : {failed_episodes}")
        print("==========================================\n")

def main(args=None):
    print("請選擇執行模式:\n1: 蒐集訓練資料 (Train Mode)\n2: 蒐集測試資料 (Test Mode)")
    mode = 'test' if input("輸入選項 (1/2): ").strip() == '2' else 'train'
    
    rclpy.init(args=args)
    node = IsaacDataCollector(mode=mode)
    
    try: 
        rclpy.spin(node)
    except SystemExit: 
        pass
    except KeyboardInterrupt: 
        pass
    finally: 
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__': 
    main()