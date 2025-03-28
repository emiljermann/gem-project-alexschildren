The ROS environment for e2 vs e4 is very similar. There are just a few small differences, e.g., 4 arena (corner) cameras on e4 versus a front-left front-right only corner cameras on e2. The actual python code (not various CMake configs), is nearly identical, aside from potential some parameters about vehicle length/width, etc.

$ source devel/setup.bash 
$ roslaunch basic_launch sensor_init.launch 

$ source devel/setup.bash 
$ bash src/utility/radar_start.sh 

# --------------------------------------------

$ source devel/setup.bash 
$ roslaunch basic_launch visualization.launch 

$ source devel/setup.bash
$ roslaunch basic_launch dbw_joystick.launch


