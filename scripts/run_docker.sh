DEFAULT_IMAGE=197237853439.dkr.ecr.us-west-2.amazonaws.com/bytetrack/training
DOCKER_IMAGE="${1:-$DEFAULT_IMAGE}"

echo $DISPLAY
xauth list

# cd ~/ByteTrack

docker stop $(docker ps -aq) && docker rm $(docker ps -aq) || \
docker run --gpus all -it \
-v $PWD:/workspace/ByteTrack \
-v /tmp/.X11-unix/:/tmp/.X11-unix:rw \
--device /dev/video0:/dev/video0:mwr \
--net=host \
--ipc=host \
-e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
-e DISPLAY=$DISPLAY \
--privileged \
--name byte \
$DOCKER_IMAGE