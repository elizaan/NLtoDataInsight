#!/bin/bash
echo " compose $1"
convert  -loop 0 $1*.png $1_animation.gif
