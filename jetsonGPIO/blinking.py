import Jetson.GPIO as GPIO
import time

# PIN = 7  # G4 on Pi Wedge
PIN = 31  # G4 on Pi Wedge
GPIO.setmode(GPIO.BOARD)
GPIO.setup(PIN, GPIO.OUT, initial=GPIO.LOW)

try:
    while True:
        GPIO.output(PIN, GPIO.HIGH)
        print("PIN 11 HIGH")
        time.sleep(5)
        GPIO.output(PIN, GPIO.LOW)
        print("PIN 11 LOW")
        time.sleep(5)
finally:
    GPIO.cleanup()
