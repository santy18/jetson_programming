# jetson orin nano commands

gpioinfo



# Setting up and turning on other pins
```
sudo /opt/nvidia/jetson-io//jetson-io.py
```

# following kernel messages
sudo dmesg --follow


# connecting arduino and flashing through the board

### Install arduino cli
```bash 
curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh
sudo mv bin/arduino-cli /usr/local/bin/

# check version
arduino-cli version
```

### initialize uno core
```bash
arduino-cli config init
arduino-cli core update-index
arduino-cli core install arduino:avr
```

### make test folder
```bash
mkdir arduino_serial
cd arduino_serial
code arduino_serial.ino
```

add code and then:

```bash
arduino-cli compile --fqbn arduino:avr:uno
```

upload over usb
```bash
arduino-cli upload -p /dev/ttyUSB0 --fqbn arduino:avr:uno

# should see avrdude done. thank you
```

you can also do together:
```bash
arduino-cli compile --fqbn arduino:avr:uno .
arduino-cli upload -p /dev/ttyUSB0 --fqbn arduino:avr:uno .
```