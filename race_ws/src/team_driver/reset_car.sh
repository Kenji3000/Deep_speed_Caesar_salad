#!/usr/bin/env bash
# Put the car back on the start grid. Practice only.
# Sent 10 times over 2 s because a single /initialpose message is often missed.
ros2 topic pub -t 10 -r 5 /initialpose geometry_msgs/msg/PoseWithCovarianceStamped \
  "{header: {frame_id: map}, pose: {pose: {position: {x: -1.68, y: -0.01}, orientation: {w: 1.0}}}}" > /dev/null
echo "Car reset to the start line."
