# EV3RL
This is work in progress to implement reinforcement learning on a Lego Mindstorms EV3 robot for use in middle school science projects. The code uses ev3dev and ev3dev-python. It is based on the vscode-hello-python. (See below for pointers to all of these.)

The main learning task is for the robot to learn to make as much forward progress as possible while avoiding obstacles. Right now, the code uses Q learning over discretized distance input and a few discrete actions. This is appropriate for younger students, because the table-based approach used in Q learning is easy to understand and explain to others (e.g., science fair judges). It is also easy for them to develop their own experiments, such as the effect of exploration vs. exploitation, or the effect of the learning and discount rate, etc. Older children might want to compare different learning algorithms, such as SARSA, Expected SARSA, SARSA(\Lambda), etc.

More advanced students might apply continuous state and action space methods, such as DDPG or TRPO. This is possible by using RPYC to run the heavy computation on a fast remote machine. 

Currently, this code is in development and will change rapidly. It is not cleanly written.

The code assumes a standard Lego EV3 driving base with two large motors (connected to B and C) and an IR sensor plus one touch sensor, both mounted to the front. The IR Sensor detects distance to an object. The Touch sensor is connected to a bumper to detect when the robot has hit an obstacle. 

The code can run directly on the EV3, or from a remote machine using RPYC. If RPYC is used, you will see a heatmap displayed on the remote machine. I recommend using Visual Studio Code for all development.

I will be updating this code regularly since we are the midst of fall science fair season.

[ev3dev](http://www.ev3dev.org)

[visual studio code](https://code.visualstudio.com/)

[python](https://www.python.org/)

[rpyc](https://rpyc.readthedocs.io/en/latest/)
