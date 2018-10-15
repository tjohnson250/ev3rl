#!/usr/bin/env python3
# This script can be downloaded to the EV3 (running EV3DEV) and run as a script 
# from the EV3's control panel.
#
# To run this script remotely on a computer, using RPYC:
# Start SSH connection to ev3dev lego robot using vis studio code ev3d browser extension
# In the SSH terminal execute: ./rcpy_server.sh, containing two lines
# #!/bin/bash
# python3 'which rpyc_classic.py' --host 0.0.0.0
# 
# see python ev3 dev docs for sensor info: https://media.readthedocs.org/pdf/python-ev3dev/latest/python-ev3dev.pdf
# if trouble connecting to rpyc reboot EV3 robot and try again
# It is far better to use a wifi dongle on the EV3 for the connection to the computer from which you SSH

import datetime
import numpy as np
import random
import socket
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from heatmap import heatmap, annotate_heatmap, DivergingNorm

hostname = socket.gethostname()

# Load libraries based on where the script is running
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
    sensors = conn.modules['ev3dev2.sensor.lego']      # import ev3dev2 remotely
    motors = conn.modules['ev3dev2.motor']

tank = motors.MoveTank('outB', 'outC')

# Uncomment these lines if your EV3 has an IR sensor
distSensorType = 'Ultrasonic' # Change to 'IR' for the IR sensor
if distSensorType == 'Ultrasonic':
    distSensor = sensors.UltrasonicSensor()
    distSensor.mode = 'US-DIST-CM'  # Measure distance in centimeters
else:
    distSensor = sensors.InfraredSensor()
    distSensor.mode = 'IR-PROX' # Put sensor in Proximity mode to measure distance from an obstacle


ts = sensors.TouchSensor()

np.set_printoptions(precision = 3)

# touch sensor
# 0: touchsensor not pressed
# 1: touchsensor pressed
numTouchSensorStates = 2

# Ultrasonic returns 0 to 255 cm. IR sensor returns 0 to 100 (v).
# Discretize these values to the following number of states
numDistSensorStates = 5

# Create bins that discretize distance so that it is finer grained up close to an obstacle
# This is a series, such as [0, 4, 8, 16, 32...]
bins = [0]
for i in range(numDistSensorStates - 1):
    bins.append((i+1)*4)

def getCoarseDistance(sensor, distSensor, bins):
    if distSensor == 'Ultrasonic':
        rawDist = sensor.distance_centimeters
    else:
        rawDist = sensor.proximity
    dist = np.digitize([rawDist], bins)[0] # Discretize distance returns bins from 1 to number of bins
    return(dist-1)   # Subtract 1 to index arrays from 0

## EV3 Tank movement actions

cycle_time = 990
speed = motors.SpeedPercent(10)
block = True          # should the program wait until the move finishes?
def forward():
    tank.on(speed, speed)
def turnleft():
    tank.on(speed/2, speed)
def turnright():
    tank.on(speed, speed/2)
def rotateright():
    tank.on(speed, -speed)
def rotateleft():
    tank.on(-speed, speed)
def backward():
    tank.on(-speed, -speed)

actions = [forward, turnleft, turnright, rotateright, rotateleft, backward]
numactions = len(actions)
action_names = (a.__name__ for a in actions) # convert functions to their names

# Issue the selected action to the EV3
def ev3action(a):
    actions[a]()


# For expected SARSA 
# Return expected value of state (st, sir) given epsilon and q table
# def expSARSA(st, sir, epsilon, q, numactions):
#    result = 0
#    best_action = np.argmax(q[st, sir])
#    for a in range(numactions):
#        result += (1/numactions * epsilon + (1-epsilon)*int(a=best_action)) * q[st, sir, a]

q_table = np.zeros([numTouchSensorStates, numDistSensorStates, numactions]) # indexed by 0

# Hyperparametes
alpha = 0.1
gamma = 0.9
# epsilon = 0.1 Here lets vary epsilon based on N0: epsilon = N0/(N0 + t) where t is the number of trials
N0 = 50

# Softmax with temperature T as an alternative to epsilon-Greedy exploration
def softmax(l, T):
    return(np.exp(l/T)/np.sum(np.exp(l/T)))

def softmaxAction(l, T):
    return(np.random.choice(len(l), 1, p = softmax(l, T))[0])

# Run a fixed number of steps
steps = 1000

# initialize state variables
st = ts.value() # touchsensor
dist = getCoarseDistance(distSensor, distSensorType, bins)

# Function to always have 0 be white in a diverging heatmap
class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
    
# create a matplotlib figure for the Q table
fig = plt.figure('Q-table')
ax = fig.add_subplot(111)
im, cbar = heatmap(q_table[0], list(range(numDistSensorStates)), action_names, ax=ax,
                    cmap="RdBu", cbarlabel="Q table",
                    norm=DivergingNorm(vcenter=0.0))
                    #norm=MidpointNormalize(midpoint=0))
texts = annotate_heatmap(im, valfmt="{x:.1f}")
plt.draw()
#fig, ax = plt.subplots()

total_reward = 0
start_time = datetime.datetime.now() # log start of episode
rewards = np.zeros(100)
for step in range(0, steps-1): 
    # Use epsilon greedy policy based on Q table
    epsilon = N0 / (N0 + step)
    if random.random() > epsilon:
        a = np.argmax(q_table[st, dist]) # find action (index) with max q value for state
    else:
        a = random.randint(0, numactions-1)

    # Try softmax action selection
    a = softmaxAction(q_table[st, dist], 1)
    # Send selected command to EV3 robot
    ev3action(a)
    tank.wait_while('running', cycle_time/2) 
    # read touch and ir sensors to find current state after taking action
    stp = ts.value()
    distp = getCoarseDistance(distSensor, distSensorType, bins)

    # calculate the reward
    if stp == 1: # robot hit wall
        r = -5
    elif a == 0: # robot moved forward without hitting wall
        r = 1
    elif a == 1 or a == 2: # robot turned left or right
        r = 0.5
    else:       # all other possibilities
        r = -1
    
    rewards[step%100] = r
    total_reward += r

    ## Use only one of the update rules below
    # Update Q table using Q Learning update function
    q_table[st, dist, a] = q_table[st, dist, a] + alpha * (r + gamma * np.amax(q_table[stp, distp]) - q_table[st, dist, a])
    
    # Update Q table using Expected SARSA update function
    #q_table[st, dist, a] = q_table[st, dist, a] + alpha * (r + gamma * expSARSA(stp, distp, epsilon, q_table, numactions) - q_table[st, dist, a])

    st = stp
    dist = distp
    if hostname != 'ev3dev':
        if step%1 == 0:
            print("*****Step #", step, " *****")
            print('Epsilon: ', epsilon, 'Touch: ', st, 'Proximity: ', dist, "Total Reward: ", total_reward)
            print(q_table)
            if step%100 > 0:
                print("PLOTTING HEATMAP")
                #im, cbar = heatmap(q_table[1], list(range(numirsensorstates)), actions, ax=ax,
                #    cmap="RdBu", cbarlabel="Q table")
                [t.remove() for t in texts]
                texts = annotate_heatmap(im, valfmt="{x:.1f}", threshold=0)
                #fig.tight_layout()
                im.set_data(q_table[0])
                cbar.set_clim([q_table[0].min(), q_table[0].max()])
                cbar.update_normal(im)
                plt.draw()
                plt.pause(0.1)
                #plt.cla()

    else:
        # disp.text_grid("Test", x=0, y=0) # Too slow
        # disp.update()
        print("*****Step #", step, " *****\n")
        print('Epsilon: ', epsilon, '\nTouch: ', st, '\nProximity: ', dist, "\nTotal Reward: ", total_reward, "\n")

    
tank.off()
