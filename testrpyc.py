# To use this file
# Start SSH connection to ev3dev lego robot using vis studio code ev3d browser extension
# In the SSH terminal execute: ./rcpy_server.sh, containing two lines
# #!/bin/bash
# python3 'which rpyc_classic.py' --host 0.0.0.0
# 
# see python ev3 dev docs for sensor info: https://media.readthedocs.org/pdf/python-ev3dev/latest/python-ev3dev.pdf
# if trouble connecting to tpyc reboot EV3 robot

import rpyc 
conn = rpyc.classic.connect('ev3dev.local') # host name or IP address of the EV3
sensors = conn.modules['ev3dev2.sensor.lego']      # import ev3dev2 remotely
# m = ev3.LargeMotor('outA')
# m.run_timed(time_sp=1000, speed_sp=600)
ir = sensors.InfraredSensor()
ir.mode = 'IR-SEEK' # Put sensor in seek (beacon) mode
#angle = ir.heading # measure of angle of beacon from center of sensor (-25 to 25)
ts = sensors.TouchSensor()


# if beacon is not sensed, ir.proxixity() returns None
#while ir.distance() is None or ir.distance() > 0:
while not ts.is_pressed:
    print("Prox:", ir.distance(), "Angle:", ir.heading())