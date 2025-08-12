# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 23:14:59 2025

@author: USER
"""

import time
import matplotlib.pyplot as plt
from transitions import Machine
from enum import Enum, auto

import rclpy
from rclpy.node import Node
from std_msgs.msg import String,Int32,Float32MultiArray,Int32MultiArray
from common_msgs.msg import ClipperCmd
from pymodbus.client import ModbusTcpClient
import time
import csv

#parameters
timer_period = 0.1  # seconds


# --- ROS2 Node ---
class DataNode(Node):
    def __init__(self):
        super().__init__('clipper_control')

        self.mode = "open"  
       
        self.clipper_cmd_subscriber = self.create_subscription(
            ClipperCmd,
            'clipper_cmd',
            self.clipper_cmd_callback,
            10
        )
        
        self.clipper_io_cmd_publisher = self.create_publisher(Int32MultiArray, 'clipper_io_cmd', 10)


    def clipper_cmd_callback(self, msg: ClipperCmd):
        self.mode = msg.mode
                
class ClipperControlState(Enum):
    IDLE = "idle"
    OPENING = "opening"
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    STOP = "stop"
    FAIL = "fail"

class ClipperControl(Machine):
    def __init__(self, data_node: DataNode):

        self.phase = ClipperControlState.IDLE  # 初始狀態
        self.data_node = data_node

        self.D17 = 0  # 控制right_receipt
        self.D19 = 0  # 控制right_move
        self.D20 = 0  # 控制right_stop
        
        self.D22 = 0  # 控制left_receipt
        self.D24 = 0  # 控制left_move
        self.D25 = 0  # 控制left_stop


        states = [
            ClipperControlState.IDLE.value,
            ClipperControlState.OPENING.value,
            ClipperControlState.OPEN.value, 
            ClipperControlState.CLOSING.value,
            ClipperControlState.CLOSED.value,
            ClipperControlState.STOP.value,
            ClipperControlState.FAIL.value
        ]

        transitions = [
            # 狀態轉換
            {'trigger': 'opening', 'source': [ClipperControlState.IDLE.value,ClipperControlState.CLOSED.value,], 'dest': ClipperControlState.OPENING.value},
            {'trigger': 'closing', 'source': [ClipperControlState.IDLE.value,ClipperControlState.OPEN.value], 'dest': ClipperControlState.CLOSING.value},
            {'trigger': 'open_finish', 'source': ClipperControlState.OPENING.value, 'dest': ClipperControlState.OPEN.value},
            {'trigger': 'close_finish', 'source': ClipperControlState.CLOSING.value, 'dest': ClipperControlState.CLOSED.value},
            {'trigger': 'stop', 'source': '*', 'dest': ClipperControlState.STOP.value},
            {'trigger': 'fail', 'source': '*', 'dest': ClipperControlState.FAIL.value},
            {'trigger': 'reset', 'source': '*', 'dest': ClipperControlState.IDLE.value}
        ]

        self.machine = Machine(model=self, states=states,transitions=transitions,initial=self.phase.value,
                               auto_transitions=False,after_state_change=self._update_phase)
        
    def _update_phase(self):
        self.phase = ClipperControlState(self.state)

    def step(self):
        
        self.run()
        # if self.data_node.state_cmd.get("pause_button", False):
        #     print("[ManualAlignmentFSM] 被暫停中")
        #     return  # 暫停中，不執行
        
        # if self.data_node.mode == "run":
        #     print("[ForkliftControl] 開始執行叉車控制任務")
        #     self.run()
        #     return
        
        # else:
        #     if self.state != ForkliftControlState.IDLE.value:
        #         print("[ForkliftControl] 非執行模式，強制回到 IDLE 狀態")
        #         self.return_to_idle()
        #         self.data_node.can_forklift_cmd = True  # 允許發送新的命令
        #         print(self.data_node.current_speed, self.data_node.current_direction)
        #         self.forklift_controller("slow","stop", self.data_node.current_height)  # 停止叉車
                
        #     # else:
        #     #     # print("[ForkliftControl] 叉車控制系統已經處於空閒狀態")

            # return

    def encode(self, speed, direction):
        """將速度和方向編碼為 Modbus 寫入值"""
        if speed == "fast":
            y0 = 1
            y1 = 0
        elif speed == "slow":
            y0 = 0
            y1 = 1
        elif speed == "stop":
            y0 = 0
            y1 = 0
        if direction == "up":
            y2 = 1
            y3 = 0
        elif direction == "down":
            y2 = 0
            y3 = 1
        elif direction == "stop":
            y2 = 0
            y3 = 0
        
        # value = (2**3)*y3 + (2**2)*y2 + (2**1)*y1 + (2**0)*y0
        value = [y0, y1, y2, y3]
        value = Int32MultiArray(data=value)  # 封裝為 Int32MultiArray

        return value

    def clipper_controller(self, mode):
        """控制夾爪的開啟和關閉"""
        result = 'waiting'

        if mode == "open":
            value = Int32MultiArray(data=[1, 0, 0, 1, 0, 0])  # 封裝為 Int32MultiArray
            self.data_node.clipper_io_cmd_publisher.publish(value)#change receipt
            value = Int32MultiArray(data=[1, 1, 0, 1, 1, 0])  # 封裝為 Int32MultiArray
            self.data_node.clipper_io_cmd_publisher.publish(value)#open move
            time.sleep(1)  # 等待夾爪開啟完成
            result = 'done'

        elif mode == "close":
            value = Int32MultiArray(data=[0, 0, 0, 0, 0, 0])  # 封裝為 Int32MultiArray
            self.data_node.clipper_io_cmd_publisher.publish(value)  #change receipt
            value = Int32MultiArray(data=[0, 1, 0, 0, 1, 0])  # 封裝為 Int32MultiArray
            self.data_node.clipper_io_cmd_publisher.publish(value)  #close move
            time.sleep(1)  # 等待夾爪關閉完成
            result = 'done'

        elif mode == "stop":
            value = Int32MultiArray(data=[0, 0, 1, 0, 0, 1])    # 封裝為 Int32MultiArray
            self.data_node.clipper_io_cmd_publisher.publish(value)  #stop move
            time.sleep(1)  # 等待夾爪停止完成
            result = 'done'
        
        return result
            



    def run(self):
        """執行狀態機的邏輯"""
        if self.state == ClipperControlState.IDLE.value:
            print("[ClipperControl] 狀態機處於空閒狀態")
            if self.data_node.mode == "open_clipper":
                print("[ClipperControl] 開始開啟夾爪")
                self.opening()
            elif self.data_node.mode == "close_clipper":
                print("[ClipperControl] 開始關閉夾爪")
                self.closing()
            return
        
        elif self.state == ClipperControlState.OPENING.value:
            print("[ClipperControl] 狀態機正在開啟夾爪")
            result = self.clipper_controller("open")
            
            if result == 'done':
                print("[ClipperControl] 夾爪開啟完成")
                self.open_finish()
            else:
                print("[ClipperControl] 夾爪開啟失敗")
                self.fail()
        
        elif self.state == ClipperControlState.OPEN.value:
            print("[ClipperControl] 狀態機處於夾爪開啟狀態")
            if self.data_node.mode == "close_clipper":
                print("[ClipperControl] 開始關閉夾爪")
                self.closing()
            elif self.data_node.mode == "stop_clipper":
                print("[ClipperControl] 停止夾爪操作")
                self.stop()
            return
        
        elif self.state == ClipperControlState.CLOSING.value:
            print("[ClipperControl] 狀態機正在關閉夾爪")
            result = self.clipper_controller("close")
            
            if result == 'done':
                print("[ClipperControl] 夾爪關閉完成")
                self.close_finish()
            else:
                print("[ClipperControl] 夾爪關閉失敗")
                self.fail()
        
        elif self.state == ClipperControlState.CLOSED.value:
            print("[ClipperControl] 狀態機處於夾爪關閉狀態")
            if self.data_node.mode == "open_clipper":
                print("[ClipperControl] 開始開啟夾爪")
                self.opening()
            elif self.data_node.mode == "stop_clipper":
                print("[ClipperControl] 停止夾爪操作")
                self.stop()
            return

        elif self.state == ClipperControlState.STOP.value:
            print("[ClipperControl] 狀態機處於停止狀態")
            result = self.clipper_controller("stop")
            if result == 'done':
                print("[ClipperControl] 夾爪已停止")
                self.reset()
            else:
                print("[ClipperControl] 停止夾爪失敗")
                self.fail()
        
        elif self.state == ClipperControlState.FAIL.value:
            print("[ClipperControl] 狀態機處於失敗狀態")
            # 在失敗狀態下，可以選擇重置或其他操作
            self.reset()
            return




def main():
    rclpy.init()
    data = DataNode()                 # ROS2 subscriber node
    system = ClipperControl(data)    # FSM 實體

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