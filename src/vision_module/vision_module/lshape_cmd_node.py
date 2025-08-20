import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class LShapeCmdNode(Node):
    """
    按鍵：
      c = capture（要求主節點進入 ROI 選框並建立模板）
      x = reset
      s = save
      q/ESC = quit（讓主節點結束，也會關這個工具）
    """
    def __init__(self):
        super().__init__('lshape_cmd')
        self.pub = self.create_publisher(String, '/lshape/cmd', 10)
        self.win = "L-Shape CMD (c=capture, x=reset, s=save, q=quit)"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, 520, 80)
        self.timer = self.create_timer(0.03, self.loop_once)

    def send(self, text):
        self.pub.publish(String(data=text))
        self.get_logger().info(f"👉 發送指令: {text}")

    def loop_once(self):
        img = 255 * (cv2.getWindowImageRect(self.win)[2] != 0)  
        canvas = (255 * (cv2.UMat(60, 500, cv2.CV_8UC3).get())).copy()
        cv2.putText(canvas, "c=capture, x=reset, s=save, q=quit",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,128,0), 2, cv2.LINE_AA)
        cv2.imshow(self.win, canvas)
        k = cv2.waitKey(10) & 0xFF
        if k == ord('c'):
            self.send('capture')
        elif k == ord('x'):
            self.send('reset')
        elif k == ord('s'):
            self.send('save')
        elif k in (ord('q'), 27):
            self.send('quit')
            rclpy.shutdown()

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main():
    rclpy.init()
    node = LShapeCmdNode()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
