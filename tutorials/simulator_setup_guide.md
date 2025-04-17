__1) Initial Setup__

1.1) Install simulator github
```shell
$ mkdir -p ~/gem_ws/src
$ cd ~/gem_ws/src
$ git clone https://github.com/hangcui1201/POLARIS_GEM_e2_Simulator
```

1.2) Add required packages
```shell
$ git clone https://github.com/ros-geographic-info/unique_identifier
$ git clone https://github.com/ros-geographic-info/geographic_info
```

1.3) Compile with catkin_make
```shell
$ source /opt/ros/noetic/setup.bash
$ cd ~/gem_ws
$ catkin_make
```

__2) Run ROS Environment__

2.1) Startup environment (for track1 world). In a new terminal window:
```shell
$ cd ~/gem_ws/src
$ source devel/setup.bash
$ cd ~/gem_ws
$ roslaunch gem_launch gem_init.launch world_name:="track1.world"  
```

2.2) Startup sensor RViz overlay. In a second terminal window:
```shell
$ cd ~/gem_ws/src
$ source devel/setup.bash
$ cd ~/gem_ws
$ roslaunch gem_launch gem_sensor_info.launch
```

2.3) Startup pure pursuit controller: In a third terminal window:
```shell
$ cd ~/gem_ws/src
$ source devel/setup.bash
$ cd ~/gem_ws
$ rosrun gem_pure_pursuit_sim pure_pursuit_sim.py
```
