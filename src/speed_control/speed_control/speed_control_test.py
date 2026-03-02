import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState  
from std_msgs.msg import Float64        
import math

TEST_SPEED = 10.0
RESET_DURATION = 5
RESET_KP = 2.0
RESET_MAX_SPEED = 2.0

JOINT_LIMIT = 3.0
MOTOR_LIMIT = 3.0

class IsaacMotorTest(Node):
    def __init__(self):
        super().__init__('isaac_motor_test')

        self.target_names = ["Motor", "Joint"]
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.listener_callback,
            10
        )
        self.get_logger().info("已啟動：關節狀態監聽中...")

        self.topic_name = '/motor_control'
        self.publisher_ = self.create_publisher(Float64, self.topic_name, 10)

        self.dt = 0.05
        self.timer = self.create_timer(self.dt, self.timer_callback)

        self.state = "RUN_POS"
        self.test_duration = 0.05 
        self.phase_timer = 0.0     

        self.current_joint_pos = 0.0
        self.current_motor_pos = 0.0

        self.get_logger().info(f"已啟動：速度發送中 -> {self.topic_name}")

    def listener_callback(self, msg):
        output_str = ""
        found_any = False 

        for target in self.target_names:
            if target in msg.name:
                idx = msg.name.index(target)
                pos = msg.position[idx]
                vel = msg.velocity[idx]

                if target == "Joint":
                    self.current_joint_pos = pos
                elif target == "Motor":
                    self.current_motor_pos = pos
                
                output_str += f"[{target}] 角:{pos:.2f} 速:{vel:.2f} | "
                found_any = True
            else:
                pass 

        if found_any:
            print(f"📥 [狀態] {output_str}")

    def timer_callback(self):     
        msg = Float64()
        fail_joint = abs(self.current_joint_pos) > JOINT_LIMIT
        fail_motor = abs(self.current_motor_pos) > MOTOR_LIMIT
        
        if fail_joint or fail_motor:
            print(f"\n\n🚨 極限已找到！")
            
            if fail_joint:
                print(f"❌ 原因: Joint 角度過大 ({abs(self.current_joint_pos):.2f} > {JOINT_LIMIT})")
            if fail_motor:
                print(f"❌ 原因: Motor 轉動過多 ({abs(self.current_motor_pos):.2f} > {MOTOR_LIMIT})")
                
            print(f"🛑 安全持續時間上限: {self.test_duration:.2f} 秒")
            
            # 強制停止
            msg.data = 0.0
            self.publisher_.publish(msg)
            self.state = "STOP"
            raise SystemExit
            
        # ===========================
        # 🔄 狀態機邏輯
        # ===========================
        
        # 1. 正轉階段 (+10)
        if self.state == "RUN_POS":
            if self.phase_timer < self.test_duration:
                msg.data = TEST_SPEED 
                self.phase_timer += self.dt
            else:
                self.state = "RUN_NEG"
                self.phase_timer = 0.0

        # 2. 反轉階段 (-10)
        elif self.state == "RUN_NEG":
            if self.phase_timer < self.test_duration:
                msg.data = -TEST_SPEED 
                self.phase_timer += self.dt
            else:
                # 通過測試
                self.state = "RESET"
                self.phase_timer = 0.0
                msg.data = 0.0
                print(f"✅ 通過 {self.test_duration:.2f}s (Joint:{abs(self.current_joint_pos):.2f}/Motor:{abs(self.current_motor_pos):.2f}) -> 歸零中...")

        # 3. 歸零階段 (讓 Motor 回到 0)
        elif self.state == "RESET":
            error = 0.0 - self.current_motor_pos
            control_effort = error * RESET_KP

            # 歸零速限
            if control_effort > RESET_MAX_SPEED:
                control_effort = RESET_MAX_SPEED
            elif control_effort < -RESET_MAX_SPEED:
                control_effort = -RESET_MAX_SPEED
            
            msg.data = control_effort

            if self.phase_timer < RESET_DURATION:
                self.phase_timer += self.dt
            else:
                # 休息結束，增加時間
                self.state = "RUN_POS"
                self.phase_timer = 0.0
                self.test_duration += 0.05 
                print(f"🔄 增加時間至: {self.test_duration:.2f} 秒")
                
        elif self.state == "STOP":
            msg.data = 0.0

        self.publisher_.publish(msg)
        print(f"📤 [命令] 發送速度: {msg.data:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = IsaacMotorTest()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()