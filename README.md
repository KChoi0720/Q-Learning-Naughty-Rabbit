# Q-Learning-Naughty-Rabbit

In this project, we set red rects as the robots, while the black ones as the obstructions which bring a -10 reward when encountered. Yellow circles are also obstructions with a lighter punishment. 

Our goal is to make the two robots get to the 16th line with higher reward.

When robots encountered and get a negative reward of running result, we can just set the epsilon to be 1 to avoid decrease but may cause local best.
