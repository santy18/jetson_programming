import serial
import time

PORT = "/dev/ttyUSB0"
BAUD = 115200

def main():
    ser = serial.Serial(PORT, BAUD, timeout=0.2)
    time.sleep(2.0)  # allow Arduino reset on open

    print(f"Opened {PORT} @ {BAUD}")

    # Read initial lines (like ARDUINO_READY)
    t0 = time.time()
    while time.time() - t0 < 3:
        line = ser.readline().decode(errors="ignore").strip()
        if line:
            print(f"<< {line}")

    # Send some messages
    for msg in ["PING", "hello arduino", "set:led=1"]:
        ser.write((msg + "\n").encode())
        print(f">> {msg}")

        # read responses for a short window
        t1 = time.time()
        while time.time() - t1 < 0.6:
            line = ser.readline().decode(errors="ignore").strip()
            if line:
                print(f"<< {line}")

    print("Entering interactive mode. Type and press Enter. Ctrl+C to exit.")
    while True:
        msg = input("> ")
        ser.write((msg + "\n").encode())
        time.sleep(0.05)

        # read whatever came back quickly
        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                break
            print(f"<< {line}")

if __name__ == "__main__":
    main()
