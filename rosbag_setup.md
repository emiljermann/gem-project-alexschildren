__Rosbag Setup__

1) Start master node
```shell
$ source /opt/ros/noetic/setup.bash
$ roscore
```

2) Run rosbag (in a new terminal window):
```shell
$ source /opt/ros/noetic/setup.bash
$ cd ~/directory_with_rosbag_inside
$ rosbag play rosbag_filename.bag --clock --loop
$ # --clock publishes the rostime to a topic for use with rviz (im 75% sure rviz will not work without this)
$ # --loop makes the bag keep looping
```

3) Run scripts (in a new terminal window):
```shell
$ source /opt/ros/noetic/setup.bash # not sure if this is actually needed here but it doesn't hurt
$ cd ~/something/e2/src/vehicle_drivers/gem_gnss_control/scripts
$ python3 your_script.py
```

__Other helpful things__
```shell
$ rostopic list # lists all current rostopics, helpful to check bag is publishing properly
$ rosbag info rosbag_filename.bag # Lists info on rosbag including the rostopics it will publish (and previously recorded)
$ rviz # Brings up rviz and if the bag and scripts are running you can view the published topics like you're on the gem car. Better for debugging imo
```
