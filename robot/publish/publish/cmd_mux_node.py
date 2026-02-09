#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from ros_robot_controller_msgs.msg import SetPWMServoState, PWMServoState

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

class CmdMuxNode(Node):
    """
    - default_mode를 'joy'로 둬서 bringup 즉시 조이스틱 먹게 함
    - /mux/select : joy|policy|stop
    - /cmd_vel_joy, /cmd_vel_drive : (v, yaw_rate=w)
    - /controller/cmd_vel : (v, yaw_rate=w)
    - servo는 (v,w) -> delta -> pwm
    """

    def __init__(self):
        super().__init__('cmd_mux_node')

        self.wheelbase = float(self.declare_parameter('wheelbase', 0.145).value)
        self.max_steer_rad = math.radians(float(self.declare_parameter('max_steer_deg', 45.0).value))

        self.servo_id = int(self.declare_parameter('servo_id', 3).value)
        self.servo_center = int(self.declare_parameter('servo_center', 1500).value)
        self.servo_scale = float(self.declare_parameter('servo_scale', 2000.0).value)
        self.servo_duration = float(self.declare_parameter('servo_duration', 0.02).value)
        self.enable_servo = bool(self.declare_parameter('enable_servo', True).value)

        self.max_linear = float(self.declare_parameter('max_linear', 0.25).value)

        # ✅ 핵심: 기본 모드 joy
        self.mode = str(self.declare_parameter('default_mode', 'joy').value).strip()
        if self.mode not in ('joy', 'policy', 'stop'):
            self.mode = 'joy'

        self.last_joy = Twist()
        self.last_policy = Twist()

        self.pub_cmd = self.create_publisher(Twist, '/controller/cmd_vel', 10)
        self.pub_servo = self.create_publisher(SetPWMServoState, '/ros_robot_controller/pwm_servo/set_state', 10)

        self.create_subscription(String, '/mux/select', self.cb_mode, 10)
        self.create_subscription(Twist, '/cmd_vel_joy', self.cb_joy, 10)
        self.create_subscription(Twist, '/cmd_vel_policy', self.cb_policy, 10)

        self.get_logger().info(f'[MUX] ready default_mode={self.mode} (no timer/timeout)')

    def cb_mode(self, msg: String):
        m = (msg.data or '').strip()
        if m not in ('joy', 'policy', 'stop'):
            return
        if m == self.mode:
            return
        self.mode = m

        if self.mode == 'stop':
            self._publish_stop()
        else:
            # 모드 바뀌자마자 마지막 값 1번 반영
            self._apply_and_publish(self.last_joy if self.mode == 'joy' else self.last_policy)

    def cb_joy(self, msg: Twist):
        self.last_joy = msg
        if self.mode == 'joy':
            self._apply_and_publish(msg)

    def cb_policy(self, msg: Twist):
        self.last_policy = msg
        if self.mode == 'policy':
            self._apply_and_publish(msg)

    def _delta_to_servo_pos(self, delta_rad: float) -> int:
        return int(self.servo_center + (math.degrees(delta_rad) / 180.0) * self.servo_scale)

    def _publish_servo(self, delta_rad: float):
        if not self.enable_servo:
            return

        pos = self._delta_to_servo_pos(delta_rad)

        servo_state = PWMServoState()
        servo_state.id = [self.servo_id]
        servo_state.position = [pos]

        msg = SetPWMServoState()
        msg.state = [servo_state]
        msg.duration = float(self.servo_duration)
        self.pub_servo.publish(msg)

    def _yaw_rate_to_delta(self, v: float, w: float) -> float:
        if abs(v) < 1e-3:
            return 0.0
        return float(math.atan((w * self.wheelbase) / v))

    def _apply_and_publish(self, cmd: Twist):
        if self.mode == 'stop':
            self._publish_stop()
            return

        v = clamp(float(cmd.linear.x), -self.max_linear, self.max_linear)
        w = float(cmd.angular.z)

        delta = self._yaw_rate_to_delta(v, w)
        delta = clamp(delta, -self.max_steer_rad, self.max_steer_rad)

        self._publish_servo(delta)

        out = Twist()
        out.linear.x = v
        out.angular.z = w
        self.pub_cmd.publish(out)

    def _publish_stop(self):
        self.pub_cmd.publish(Twist())
        self._publish_servo(0.0)

def main(args=None):
    rclpy.init(args=args)
    node = CmdMuxNode()
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
