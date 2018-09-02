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
else:
    # We are running remotely, so assume we are using RPYC
    import rpyc 
    conn = rpyc.classic.connect('ev3dev.local') # host name or IP address of the EV3
    sensors = conn.modules['ev3dev2.sensor.lego']      # import ev3dev2 remotely
    motors = conn.modules['ev3dev2.motor']

tank = motors.MoveTank('outB', 'outC')
ir = sensors.InfraredSensor()
ir.mode = 'IR-PROX' # Put sensor in Proximity mode to measure distance from an obstacle
ts = sensors.TouchSensor()

import numpy as np
import random

np.set_printoptions(precision = 3)

numtouchsensorstates = 2
numirsensorstates = 5       # IR sensor returns 0 to 100 (v). This is rescaled using v%25

# states
# touch sensor
# 0: touchsensor not pressed
# 1: touchsensor pressed

# ir sensor
# 0 to 5 (scaled from 0 to 100)

## EV3 Tank movement actions
tank.stop_action = STOP_ACTION_COAST
def forward():
    tank.on_for_seconds(10, 10, 1, block=False)
def turnleft():
    tank.on_for_seconds(5, 10, 1, block=False)
def turnright():
    tank.on_for_seconds(10, 5, 1, block=False)
def rotateright():
    tank.on_for_seconds(10, -10, 1, block=False)
def rotateleft():
    tank.on_for_seconds(-10, 10, 1, block=False)
def backward():
    tank.on_for_seconds(-10, -10, 1, block=False)

actions = [forward, turnleft, turnright, rotateright, rotateleft, backward]
numactions = len(actions)

def ev3action(a):
    actions[a]()

# For expected SARSA 
# Return expected value of state (st, sir) given epsilon and q table
# def expSARSA(st, sir, epsilon, q, numactions):
#    result = 0
#    best_action = np.argmax(q[st, sir])
#    for a in range(numactions):
#        result += (1/numactions * epsilon + (1-epsilon)*int(a=best_action)) * q[st, sir, a]

q_table = np.zeros([numtouchsensorstates, numirsensorstates, numactions]) # indexed by 0

# Hyperparametes
alpha = 0.1
gamma = 0.9
# epsilon = 0.1 Here lets vary epsilon based on N0: epsilon = N0/(N0 + t) where t is the number of trials
N0 = 50

# Run a fixed number of trials
trials = 1000

# initialize state variables
st = ts.value() # touchsensor
sir = round(ir.proximity/25.0) # ir sensor proximity

total_reward = 0

for x in range(1, trials):
    # Use epsilon greedy policy based on Q table
    epsilon = N0 / (N0 + x)
    if random.random() > epsilon:
        a = np.argmax(q_table[st, sir]) # find action (index) with max q value for state
    else:
        a = random.randint(0, numactions)

    # Send selected command to EV3 robot
    ev3action(a)

    # read touch and ir sensors to find current state after taking action
    stp = ts.value()
    sirp = round(ir.proximity/25.) # simplify to 5 bins

    # calculate the reward
    if stp == 1: # robot hit wall
        r = -5
    elif a == 0: # robot moved forward without hitting wall
        r = 1
    elif a == 1 or a == 2: # robot turned left or right
        r = 0.5
    else:       # all other possibilities
        r = -1
    
    total_reward += r

    ## Use only one of the update rules below
    # Update Q table using Q Learning update function
    q_table[st, sir, a] = q_table[st, sir, a] + alpha * (r + gamma * np.amax(q_table[stp, sirp]) - q_table[st, sir, a])
    
    # Update Q table using Expected SARSA update function
    #q_table[st, sir, a] = q_table[st, sir, a] + alpha * (r + gamma * expSARSA(stp, sirp, epsilon, q_table, numactions) - q_table[st, sir, a])

    st = stp
    sir = sirp
    if hostname != 'ev3dev':
        if x%1 == 0:
            print("*****Step #", x, " *****")
            print('Epsilon: ', epsilon, 'Touch: ', st, 'Proximity: ', sir, "Total Reward: ", total_reward)
            print(q_table)

