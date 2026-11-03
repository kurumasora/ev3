#!/usr/bin/env pybricks-micropython
"""
EV3 server program (based on the senpai's version).

Sends  "distance_mm,gyro_angle\n"  to the Raspberry Pi.
Receives "left_speed,right_speed\n"  from the Raspberry Pi and runs the motors.

Changes from the original:
  * SO_REUSEADDR so restarting does not give "port already in use"
  * Motor directions are set by two constants (see below)
  * Takes only the LAST line if several commands arrive together
  * If no command arrives for 1 second, the motors stop (safety)
  * Shorter loop delay so the robot reacts faster
  * No f-strings (older MicroPython on the EV3 may not support them)
"""
import socket
import time

from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor, UltrasonicSensor, GyroSensor
from pybricks.parameters import Port, Stop, Direction

ev3 = EV3Brick()

# Motors on opposite sides of the robot are mounted mirrored, so one side
# usually needs the opposite direction. TEST with the robot on blocks:
# if "forward" makes the robot spin, flip one of these two.
LEFT_DIR = Direction.CLOCKWISE
RIGHT_DIR = Direction.COUNTERCLOCKWISE

motor_left1 = Motor(Port.A, LEFT_DIR)
motor_left2 = Motor(Port.B, LEFT_DIR)
motor_right1 = Motor(Port.C, RIGHT_DIR)
motor_right2 = Motor(Port.D, RIGHT_DIR)
motors = (motor_left1, motor_left2, motor_right1, motor_right2)

us = UltrasonicSensor(Port.S1)
gyro = GyroSensor(Port.S2)
gyro.reset_angle(0)          # keep the robot still while this runs

HOST = ""                    # listen on all network interfaces
PORT = 9999

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print("Waiting for connection...")
conn, addr = server_socket.accept()
print("Connected by " + str(addr))
conn.settimeout(1.0)         # do not wait forever for a command


def stop_all():
    for m in motors:
        m.stop(Stop.BRAKE)


try:
    while True:
        distance = us.distance()
        angle = gyro.angle()

        # Send the sensor values to the Pi
        conn.send((str(distance) + "," + str(angle) + "\n").encode())

        # Receive the motor command from the Pi
        try:
            data = conn.recv(1024)
        except OSError:
            # No command within 1 second: stop for safety and keep waiting
            stop_all()
            continue

        if not data:
            print("Pi disconnected")
            break

        # If several commands arrived together, use the newest complete one
        lines = data.decode().strip().split("\n")
        try:
            left_speed, right_speed = map(int, lines[-1].split(","))
        except ValueError:
            continue         # ignore a broken message

        motor_left1.run(left_speed)
        motor_left2.run(left_speed)
        motor_right1.run(right_speed)
        motor_right2.run(right_speed)

        time.sleep(0.03)

finally:
    stop_all()
    conn.close()
    server_socket.close()