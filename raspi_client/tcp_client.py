import socket

EV3_IP = '10.42.0.176'
PORT = 9999

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((EV3_IP, PORT))

try:
    while True:
        data = sock.recv(1024).decode().strip()
        if not data:
            continue
        distance = float(data.split('\n')[-1])
        print("Distance:", distance, "mm")
        speed = 0 if distance < 200 else 200   # stop if something is closer than 20 cm
        sock.send((str(speed) + "," + str(speed) + "\n").encode())
except KeyboardInterrupt:
    print("Stopping")
finally:
    sock.send(b"0,0\n")
    sock.close()
