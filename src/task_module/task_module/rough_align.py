import time
import networkx as nx
import matplotlib.pyplot as plt
from transitions import Machine
from functools import wraps
from enum import Enum, auto

import rclpy
from rclpy.node import Node
from std_msgs.msg import String,Float32MultiArray
from common_msgs.msg import StateCmd,MotionState, MotionCmd

#parameters
timer_period = 0.5  # seconds


# --- ROS2 Node ---
class DataNode(Node):
    def __init__(self):

        self.state_cmds ={
            'pause_button': False,
        }

        self.func_cmds = {
            'pick_button': False,
            'push_button': False,
        }
       
        self.depth_data = [500.0,500.0]

        self.state_info = "idle"  # 狀態信息
       
        # 初始化 ROS2 Node
        #subscriber
        super().__init__('data_node')
        self.state_info_subscriber = self.create_subscription(
            String,
            '/state_info',
            self.state_info_callback,
            10
        )
        self.motion_state_subscriber = self.create_subscription(
            MotionState,
            "/motion_state",
            self.motion_state_callback,
            10
        )
        self.depth_data_subscriber = self.create_subscription(
            Float32MultiArray,
            "/depth_data",
            self.depth_data_callback,
            10
        )

        #publisher
        self.motion_state_publisher = self.create_publisher(MotionState, '/motion_state', 10)
        self.motion_cmd_publisher = self.create_publisher(MotionCmd, '/motion_cmd', 10)

    # def state_cmd_callback(self, msg: StateCmd):
    #     print(f"接收到狀態命令: {msg}")
    #     # 在這裡可以處理狀態命令
    #     self.state_cmds = {
    #         'pause_button': msg.pause_button,
    #     }

    def state_info_callback(self, msg: String):
        print(f"接收到狀態信息: {msg.data}")
        self.state_info = msg.data      
    
    def motion_state_callback(self, msg=MotionState):
        print(f"接收到運動狀態: {msg}")
        # 在這裡可以處理運動狀態
        self.motion_states = {
            'motion_finish': msg.motion_finish,
            'init_finish': msg.init_finish,
            'pull_finish': msg.pull_finish,                 
            'push_finish': msg.push_finish,
            'rough_pos_finish': msg.rough_pos_finish,
            'auto_pos_finish': msg.auto_pos_finish,
            'system_error': msg.system_error
        }

    def depth_data_callback(self, msg: Float32MultiArray):
        print(f"接收到深度數據: {msg.data}")
        # 在這裡可以處理深度數據
        self.depth_data = msg.data      
        # 更新深度數據
        if len(self.depth_data) >= 2:
            self.depth_data[0] = msg.data[0]
            self.depth_data[1] = msg.data[1]        
        else:
            self.get_logger().warn("接收到的深度數據長度不足，無法更新。")


class ManualAlignState(Enum):
    IDLE = "idle"
    INIT = "init"
    CHECK_DEPTH = "check_depth"
    ROUGH_ALIGN = "rough_align"
    DONE = "done"
    FAIL = "fail"

class ManualAlignFSM(Machine):
    def __init__(self, data_node: DataNode):
        self.phase = ManualAlignState.IDLE  # 初始狀態
        self.data_node = data_node
        self.run_mode = "pick"

        states = [
            ManualAlignState.IDLE.value,
            ManualAlignState.INIT.value,
            ManualAlignState.CHECK_DEPTH.value,
            ManualAlignState.ROUGH_ALIGN.value,
            ManualAlignState.DONE.value,
            ManualAlignState.FAIL.value
        ]
        
        transitions = [
            {'trigger': 'idle_to_init', 'source': ManualAlignState.IDLE.value, 'dest': ManualAlignState.INIT.value},
            {'trigger': 'init_to_check_depth', 'source': ManualAlignState.INIT.value, 'dest': ManualAlignState.CHECK_DEPTH.value},
            {'trigger': 'check_depth_to_rough_align', 'source': ManualAlignState.CHECK_DEPTH.value, 'dest': ManualAlignState.ROUGH_ALIGN.value},
            {'trigger': 'rough_align_to_done', 'source': ManualAlignState.ROUGH_ALIGN.value, 'dest': ManualAlignState.DONE.value},
            {'trigger': 'done_to_fail', 'source': ManualAlignState.DONE.value, 'dest': ManualAlignState.FAIL.value},    
            {'trigger': 'return_to_idle', 'source': '*', 'dest': ManualAlignState.IDLE.value},
        ]

        self.machine = Machine(model=self, states=states,transitions=transitions,initial=self.phase.value,
                               auto_transitions=False,after_state_change=self._update_phase)
        
    def _update_phase(self):
        self.phase = ManualAlignState(self.state)

    def depth_ref(self,run_mode):
        """根據運行模式返回參考深度"""
        if run_mode == "pick":
            return 90.0
        elif run_mode == "push":
            return 0.3

    def reset_parameters(self):
        """重置參數"""
        self.run_mode = "pick"
        self.data_node.state_info = "idle"
        self.data_node.depth_data = [600.0, 600.0]
        self.data_node.state_cmds = {
            'pause_button': False,
        }
        self.data_node.func_cmds = {
            'pick_button': False,
            'push_button': False,
        }

    def step(self):
        if self.data_node.state_cmds.get("pause_button", False):
            print("[ManualAlignmentFSM] 被暫停中")
            return  # 暫停中，不執行
        
        if self.data_node.state_info == "rough_align":
            print("[ManualAlignmentFSM] 開始手動對齊任務")
            self.run()
        else:
            print("[ManualAlignmentFSM] 手動對齊任務未啟動，等待中")
            self.reset_parameters()  # 重置參數
            self.return_to_idle()  # 返回到空閒狀態
            self.run()
            return

        # 任務完成或失敗時自動清除任務旗標

    def run(self):
        if self.state == ManualAlignState.IDLE.value:
            print("[ManualAlignmentFSM] 等待開始手動對齊")
            if self.data_node.state_info == "rough_align":
                print("[ManualAlignmentFSM] 開始手動對齊")
                self.idle_to_init()
            else:
                print("[ManualAlignmentFSM] 手動對齊任務未啟動，等待中")
        
        elif self.state == ManualAlignState.INIT.value:
            print("[ManualAlignmentFSM] 初始化階段")
            if self.data_node.func_cmds.get("pick_button", False):
                self.run_mode = "pick"
            elif self.data_node.func_cmds.get("push_button", False):
                self.run_mode = "push"
            else:
                print("[ManualAlignmentFSM] 未選擇運行模式，等待人為選擇")
                return
            #open_vision_guide_line(red=True, green=False, blue=False)
            #open_guide_raser
            self.init_to_check_depth()

        elif self.state == ManualAlignState.CHECK_DEPTH.value:
            print("[ManualAlignmentFSM] 深度檢查階段")
            # 檢查深度數據
            depth_ref = self.depth_ref(self.run_mode)

            if self.data_node.depth_data[0] < depth_ref:
                print("depth_data 小於參考深度，進入粗對齊階段")
                self.check_depth_to_rough_align()
            else:
                print("depth_data 大於或等於參考深度，waiting for human push")
        
        elif self.state == ManualAlignState.ROUGH_ALIGN.value:
            print("[ManualAlignmentFSM] 粗對齊階段")
            #open rough aligh check
            #point_dist = vision.get_point_distance(self.run_mode)
            # if point_dist < 0.05:  # 假設 0.05 是粗對齊的閾值
            #     print("粗對齊完成，進入完成階段")
            #     self.rough_align_to_done()
            # else:
            #     print("粗對齊未完成，等待人為調整")
            #     # 這裡可以加入等待人為調整的邏輯
        
        elif self.state == ManualAlignState.DONE.value:
            print("[ManualAlignmentFSM] 對齊完成階段")
            #change_vision_guide_line(red=False, green=True, blue=False)
            #open_guide_raser
            
        elif self.state == ManualAlignState.FAIL.value:
            print("[ManualAlignmentFSM] 對齊失敗階段")
            #change_vision_guide_line(red=False, green=False, blue=True)
            #open_guide_raser
            # 這裡可以加入對齊失敗的處理邏輯，例如重試或報錯

        



        



def main():
    rclpy.init()
    data = DataNode()                 # ROS2 subscriber node
    system = ManualAlignFSM(data)    # FSM 實體

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(data)

    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
            system.step()
            print(f"[現在狀態] {system.state}")
            time.sleep(timer_period)

    except KeyboardInterrupt:
        pass
    finally:
        data.destroy_node()
        rclpy.shutdown()
        plt.ioff()
        plt.show()

# 🏁 若此檔案直接執行，就進入 main()
if __name__ == "__main__":
    main()
