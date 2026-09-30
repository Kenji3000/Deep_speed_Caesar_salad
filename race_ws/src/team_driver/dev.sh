#!/usr/bin/env bash
# Development loop: restarts your driver every time you save driver.py,
# and puts the car back on the start grid first. The simulator keeps running.
#
# Run inside the container (root@roboracer), with the simulator already running:
#   bash /hackathon/race_ws/src/team_driver/dev.sh
#   bash /hackathon/race_ws/src/team_driver/dev.sh -p v_max:=7.0 -p a_lat:=5.0
# Stop with Ctrl+C. Practice only - never teleport the car in a scored run.

PKG=/hackathon/race_ws/src/team_driver
FILE=$PKG/team_driver/driver.py
source /hackathon/race_ws/install/local_setup.bash

ARGS=()
[ $# -gt 0 ] && ARGS=(--ros-args "$@")
PID=""

stop_driver() {
  if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
    kill -TERM -- -"$PID" 2>/dev/null
    for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$PID" 2>/dev/null || break; sleep 0.3; done
    kill -KILL -- -"$PID" 2>/dev/null; wait "$PID" 2>/dev/null
  fi
  PID=""
}

reset_car() {
  echo ">>> stopping the car and moving it to the start line"
  # The simulator keeps the last command, so ask for zero speed first.
  ros2 topic pub -t 5 -r 10 /drive ackermann_msgs/msg/AckermannDriveStamped \
    "{drive: {speed: 0.0, steering_angle: 0.0}}" > /dev/null
  ros2 topic pub -t 10 -r 5 /initialpose geometry_msgs/msg/PoseWithCovarianceStamped \
    "{header: {frame_id: map}, pose: {pose: {position: {x: -1.68, y: -0.01}, orientation: {w: 1.0}}}}" > /dev/null
}

start_driver() {
  if ! python3 -m py_compile "$FILE"; then
    echo ">>> driver.py has an error (see above). Fix it and save again."
    return
  fi
  echo ">>> starting driver ${ARGS[*]}"
  setsid ros2 run team_driver driver "${ARGS[@]}" &
  PID=$!
}

trap 'echo; stop_driver; exit 0' INT TERM

last=$(stat -c %Y "$FILE")
reset_car
start_driver
echo ">>> watching driver.py - save it in VS Code to restart. Ctrl+C to quit."
while true; do
  sleep 1
  now=$(stat -c %Y "$FILE")
  if [ "$now" != "$last" ]; then
    last=$now
    echo ">>> driver.py changed"
    stop_driver
    reset_car
    start_driver
    echo ">>> in RViz: click Reset (bottom-left) to see the car again"
  fi
done