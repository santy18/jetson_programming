import serial
import time

PORT = "/dev/ttyUSB0"
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=1)

# Arduino resets when serial opens
time.sleep(2)

print("Connected. Sending commands...")

while True:
    cmd = input("Enter 1 (ON) or 0 (OFF): ")

    if cmd not in ["0", "1"]:
        continue

    ser.write(cmd.encode())
