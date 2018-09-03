#!/usr/bin/env python3
# To use this file through rpyc
# Start SSH connection to ev3dev lego robot using vis studio code ev3d browser extension
# In the SSH terminal execute: ./rcpy_server.sh, containing two lines
# #!/bin/bash
# python3 'which rpyc_classic.py' --host 0.0.0.0
# 
# see python ev3 dev docs for sensor info: https://media.readthedocs.org/pdf/python-ev3dev/latest/python-ev3dev.pdf
# if trouble connecting to tpyc reboot EV3 robot

import socket
hostname = socket.gethostname()

if hostname == 'ev3dev':
    # We are running on the EV3, import modules directly
    import ev3dev2.sensor.lego as sensors
    import ev3dev2.motor as motors
    import os
    os.system('setfont Lat15-TerminusBold14')
    import ev3dev2.display as display # Only need this when running on ev3
    disp = display.Display()
else:
    # We are running remotely, so assume we are using RPYC
    import rpyc 
    conn = rpyc.classic.connect('ev3dev.local') # host name or IP address of the EV3
    motors = conn.modules['ev3dev2.motor']

tank = motors.MoveTank('outB', 'outC')
tank.off()