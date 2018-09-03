#!/usr/bin/env python3
import ev3dev2.sensor.lego as sensors
import ev3dev2.motor as motors
import os
#from rl.agents.cem import CEMAgent

os.system('setfont Lat15-TerminusBold14')
mL = motors.LargeMotor('outB'); ml.stop_action = 'hold'
mR = motors.LargeMotor('outC'); mR.stop_action = 'hold'
print('Hello, my name is EV3!')
Sound.speak('Hello, my name is EV3!').wait()
mL.on_to_position(10, 10)
mR.on_to_position(10, 10)
mL.wait_while('running')
mR.wait_while('running')
