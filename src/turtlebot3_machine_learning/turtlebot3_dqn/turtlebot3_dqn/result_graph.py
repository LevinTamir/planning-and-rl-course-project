#!/usr/bin/env python
import signal
import sys
import threading
from PyQt5.QtCore import QTimer, pyqtSignal, QObject
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
import pyqtgraph
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

class DataBridge(QObject):
    data_received = pyqtSignal(object)

class GraphSubscriber(Node):
    def __init__(self, data_bridge):
        super().__init__("graph")
        self.data_bridge = data_bridge
        self.subscription = self.create_subscription(
            Float32MultiArray, "/result", self.data_callback, 10
        )

    def data_callback(self, msg):
        # Emit signal instead of direct GUI update
        self.data_bridge.data_received.emit(msg)

class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.setWindowTitle("Result")
        self.setGeometry(50, 50, 600, 650)
        self.ep = []
        self.data_list = []
        self.rewards = []
        self.count = 1
        
        # Create data bridge for thread-safe communication
        self.data_bridge = DataBridge()
        self.data_bridge.data_received.connect(self.receive_data)
        
        self.plot()
        
        # Start ROS2 in separate thread
        self.ros_subscriber = GraphSubscriber(self.data_bridge)
        self.ros_thread = threading.Thread(
            target=rclpy.spin, args=(self.ros_subscriber,), daemon=True
        )
        self.ros_thread.start()

    def receive_data(self, msg):
        # This runs on main thread via Qt signal
        self.data_list.append(msg.data[0])
        self.ep.append(self.count)
        self.count += 1
        self.rewards.append(msg.data[1])

    def plot(self):
        self.qValuePlt = pyqtgraph.PlotWidget(self, title="Average max Q-value")
        self.qValuePlt.setGeometry(0, 320, 600, 300)
        self.rewardsPlt = pyqtgraph.PlotWidget(self, title="Total reward")
        self.rewardsPlt.setGeometry(0, 10, 600, 300)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(200)
        self.show()

    def update(self):
        self.rewardsPlt.showGrid(x=True, y=True)
        self.qValuePlt.showGrid(x=True, y=True)
        self.rewardsPlt.plot(self.ep, self.data_list, pen=(255, 0, 0), clear=True)
        self.qValuePlt.plot(self.ep, self.rewards, pen=(0, 255, 0), clear=True)

    def closeEvent(self, event):
        if hasattr(self, 'ros_subscriber') and self.ros_subscriber is not None:
            self.ros_subscriber.destroy_node()
        rclpy.shutdown()
        event.accept()

def main():
    import os
    # macOS Qt threading fix
    os.environ['QT_MAC_WANTS_LAYER'] = '1'
    
    rclpy.init()
    
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    
    win = Window()

    def shutdown_handler(sig, frame):
        print("shutdown")
        if hasattr(win, 'ros_subscriber') and win.ros_subscriber is not None:
            win.ros_subscriber.destroy_node()
        rclpy.shutdown()
        app.quit()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    
    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        shutdown_handler(signal.SIGINT, None)

if __name__ == "__main__":
    main()