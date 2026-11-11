#!/usr/bin/env pybricks-micropython
import socket
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor, UltrasonicSensor
from pybricks.parameters import Port

ev3 = EV3Brick()
motor_left = Motor(Port.A)
motor_right = Motor(Port.B)
us = UltrasonicSensor(Port.S1)

HOST = '10.42.0.176'   # EV3's own IP
PORT = 9999
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(1)
print("Waiting for connection...")
conn, addr = server.accept()
print("Connected by " + str(addr))
conn.settimeout(1.0)   # no command for 1 s -> stop motors

try:
    while True:
        conn.send((str(us.distance()) + "\n").encode())
        try:
            cmd = conn.recv(1024).decode().strip()
            if cmd:
                left, right = map(int, cmd.split(',')[:2])
                motor_left.run(left)
                motor_right.run(right)
        except OSError:
            motor_left.stop()
            motor_right.stop()
finally:
    motor_left.stop()
    motor_right.stop()
    conn.close()
    server.close()
