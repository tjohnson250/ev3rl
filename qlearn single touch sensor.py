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
motors = conn.modules['ev3dev2.motor'] 
tank = motors.MoveTank('outB', 'outC')
#ir = sensors.InfraredSensor()
#ir.mode = 'IR-SEEK' # Put sensor in seek (beacon) mode
ts = sensors.TouchSensor()

import numpy as np
import random

numstates = 2
numactions = 4

# states
# 0: touchsensor not pressed
# 1: touchsensor pressed

# actions
# 0: move forward
# 1: move backwards
# 2: turn right backwards
# 3: turn left backwards

## EV3 Tank movement actions
def forward():
    tank.on_for_seconds(10, 10, 2)
def backward():
    tank.on_for_seconds(-10, -10, 2)
def rightback():
    tank.on_for_seconds(-10, 10, 2)
def leftback():
    tank.on_for_seconds(10, -10, 2)

forward()
backward()
rightback()
leftback()
def ev3action(a):
    if a == 0:
        forward()
    elif a == 1:
        backward()
    elif a == 2:
        rightback()
    elif a == 3:
        leftback()
    else:
        print("Invalid EV3 action")

q_table = np.zeros([numstates, numactions]) # indexed by 0

# Hyperparametes
alpha = 0.1
gamma = 0.9
# epsilon = 0.1 Here lets vary epsilon based on N0: epsilon = N0/(N0 + t) where t is the number of trials
N0 = 10

# Run a fixed number of trials
trials = 1000

# initialize state variable
s = ts.value() # touchsensor

for x in range(1, trials):
    # Use epsilon greedy policy based on Q table
    epsilon = N0 / (N0 + x)
    if random.random() > epsilon:
        a = np.argmax(q_table[s]) # find action (index) with max q value for state
    else:
        a = random.randint(0, 3)

    # Send selected command to EV3 robot
    ev3action(a)

    # read touch and ir sensors to find current state after taking action
    sp = ts.value()

    # calculate the reward
    if sp == 1: # robot hit wall
        r = -5
    elif a == 0: # robot moved forward without hitting wall
        r = 1
    else:       # all other possibilities
        r = -1
    
    # Update Q table
    q_table[s, a] = q_table[s, a] + alpha * (r + gamma * np.amax(q_table[sp]) - q_table[s, a])
    s = sp
    print("*****Step #", x, " *****")
    print('Epsilon: ', epsilon)
    print(q_table)

