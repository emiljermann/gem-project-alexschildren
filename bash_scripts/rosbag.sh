#!/bin/bash


#USAGE bash rosbag.sh ros_bag_path [optional name_of_script]
# CTRL-C to exit
#   ~/Downloads/2025-04-13-09-22-00.bag
#   default script is e2/src/vehicle_drivers/gem_gnss_control/scripts/pedestrian_detector.py by default


ROSBAG="$1"

if [ $# -lt 2 ]; then
    SCRIPTS=("e2/src/vehicle_drivers/gem_gnss_control/scripts/pedestrian_detector.py" "e2/src/vehicle_drivers/gem_gnss_control/scripts/image_printer.py")
else
    SCRIPTS="${@:2}"
fi

declare -A SCRIPT_TIMES
declare -A SCRIPT_PIDS

trap "itsover" ERR EXIT INT TERM

itsover() {
    trap '' INT TERM
    echo 'Cleaning up the kitchen'
    kill -TERM 0 
    echo 'The kitchen cleanup is achieved'
    exit 0
}


source /opt/ros/noetic/setup.bash

roscore &
sleep 2
rosbag play "$ROSBAG" --clock --loop &
rviz &

for SCRIPT in "${SCRIPTS[@]}"; do
    echo "Reloading $SCRIPT..."
    python3 "$SCRIPT" &
    SCRIPT_PID=$!
    SCRIPT_PIDS["$SCRIPT"]=$SCRIPT_PID
    LAST_MOD_TIME=$(stat -c %Y "$SCRIPT")
    SCRIPT_TIMES["$SCRIPT"]=$LAST_MOD_TIME
done

reload() {
    for SCRIPT in "${SCRIPTS[@]}"; do
        SCRIPT_PID="${SCRIPT_PIDS["$SCRIPT"]}"
        echo "Reloading $SCRIPT..."
        kill "$SCRIPT_PID" 2>/dev/null
        python3 "$SCRIPT" &
        SCRIPT_PID=$!
        SCRIPT_PIDS["$SCRIPT"]=$SCRIPT_PID
    done
}



while true; do 
    sleep 1;
    for SCRIPT in "${SCRIPTS[@]}"; do
        NEW_MOD_TIME=$(stat -c %Y "$SCRIPT")
        LAST_MOD_TIME="${SCRIPT_TIMES["$SCRIPT"]}"
        if [ "$NEW_MOD_TIME" != "$LAST_MOD_TIME" ]; then 
            LAST_MOD_TIME="$NEW_MOD_TIME"
            SCRIPT_TIMES["$SCRIPT"]=$LAST_MOD_TIME
            reload
        fi
    done
done