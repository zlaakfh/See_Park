#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import struct
import math
import threading
import json

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from ros_robot_controller_msgs.msg import BuzzerState

BTN_START = 9
BTN_END   = 8
BTN_X     = 3

AXIS_LY = 1
AXIS_RX = 2

# 'mode' = 12

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

class JoystickNode(Node):
    """
    - thread: /dev/input/js0 read -> raw_axes 갱신
    - timer: publish_hz로 /cmd_vel_joy 계속 publish (스틱 고정해도 계속 움직임)
    - 버튼 press에서만 /joy/event + buzzer
    """

    def __init__(self):
        super().__init__('joystick_node')

        self.max_linear = float(self.declare_parameter('max_linear', 0.25).value)
        self.wheelbase  = float(self.declare_parameter('wheelbase', 0.145).value)
        self.max_steer  = math.radians(float(self.declare_parameter('max_steer_deg', 45.0).value))
        self.deadzone   = float(self.declare_parameter('deadzone', 0.05).value)
        self.inv_ly     = bool(self.declare_parameter('invert_ly', True).value)
        self.inv_rx     = bool(self.declare_parameter('invert_rx', True).value)
        self.pub_hz     = float(self.declare_parameter('publish_hz', 20.0).value)

        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel_joy', 10)
        self.pub_evt = self.create_publisher(String, '/joy/event', 10)
        self.pub_buz = self.create_publisher(BuzzerState, '/ros_robot_controller/set_buzzer', 10)

        self.raw_axes = [0.0] * 16
        self.raw_buttons = [0] * 32
        self._btn_state = [0] * 64  # 버튼 상태 기억 (rising-edge용)

        self._lock = threading.Lock()
        self.running = True

        self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.read_thread.start()

        period = 1.0 / self.pub_hz if self.pub_hz > 0 else 0.05
        self.create_timer(period, self._publish_tick)

        self.get_logger().info(f'[JOY] ready publish_hz={self.pub_hz} -> /cmd_vel_joy (v,yaw_rate)')

    def _beep(self, kind: str):
        freq_map = {'START': 2600, 'X': 1800, 'END': 1200}
        msg = BuzzerState()
        msg.freq = int(freq_map.get(kind, 2000))
        msg.on_time = 0.05
        msg.off_time = 0.01
        msg.repeat = 1
        self.pub_buz.publish(msg)

    def _read_loop(self):
        dev = "/dev/input/js0"
        if not os.path.exists(dev):
            self.get_logger().error(f'{dev} not found')
            return

        try:
            with open(dev, "rb") as f:
                while self.running and rclpy.ok():
                    data = f.read(8)
                    if not data:
                        continue

                    _, value, type_, number = struct.unpack("IhBB", data)
                    real_type = type_ & ~0x80

                    if real_type == 1:  # button
                        if number < len(self.raw_buttons):
                            pressed = 1 if value else 0

                            with self._lock:
                                self.raw_buttons[number] = pressed

                            # ✅ rising edge (0->1)일 때만 이벤트 발행
                            prev = self._btn_state[number] if number < len(self._btn_state) else 0
                            if pressed == 1 and prev == 0:
                                self.get_logger().info(f'[JOY] button pressed number={number}')

                                if number == BTN_START:
                                    self.pub_evt.publish(String(data=json.dumps({"type": "START"})))
                                elif number == BTN_X:
                                    self.pub_evt.publish(String(data=json.dumps({"type": "X"})))
                                elif number == BTN_END:
                                    self.pub_evt.publish(String(data=json.dumps({"type": "END"})))

                            # 상태 업데이트
                            if number < len(self._btn_state):
                                self._btn_state[number] = pressed


                    elif real_type == 2:  # axis
                        if number < len(self.raw_axes):
                            norm = float(value) / 32767.0
                            if abs(norm) < self.deadzone:
                                norm = 0.0
                            with self._lock:
                                self.raw_axes[number] = norm

        except Exception as e:
            self.get_logger().error(f'Joystick read error: {e}')

    def _publish_tick(self):
        with self._lock:
            ly = self.raw_axes[AXIS_LY] if AXIS_LY < len(self.raw_axes) else 0.0
            rx = self.raw_axes[AXIS_RX] if AXIS_RX < len(self.raw_axes) else 0.0

        if self.inv_ly:
            ly = -ly
        if self.inv_rx:
            rx = -rx

        v = clamp(ly, -1.0, 1.0) * self.max_linear
        delta = clamp(rx, -1.0, 1.0) * self.max_steer

        if abs(v) < 1e-4 or abs(delta) < 1e-6:
            w = 0.0
        else:
            w = float(v * math.tan(delta) / self.wheelbase)

        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)   # yaw_rate
        self.pub_cmd.publish(msg)

    def destroy_node(self):
        self.running = False
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = JoystickNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()
