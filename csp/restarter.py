from subprocess import run
from time import sleep

# Path and name to the script you are trying to start
# command = ["run-docker", "--image_name='simsek/dl:cuda11_detectron2_jupy_py36'", "2", "''", \
#             "python3 train.py --logdir mini_set_newanchors --ratio 0.101 --batch 16 --anchors"] 

command = ["run-docker", "--image_name='simsek/dl:cuda11_detectron2_jupy_py36'", "3", "''", \
        "python3 train.py --logdir R50_512x2048_defanchors --model R50 --img 512 2048 --resume --batch 8"]

command = ["run-docker", "--image_name='simsek/dl:cuda10_pytorch_mish2006_py36'", "6", "''", \
        "python3 train.py --weights ./runs/exp44_yolov4-csp-bulb-dir-all-multGPU-cspweights/weights/best_yolov4-csp-bulb-dir-all-multGPU-cspweights.pt \
            --img 2048 --data ./data/dtld_bulb.yaml --batch 2 --cfg ./models/yolov4-csp.yaml --epochs 50 --name csp_640to2048_resume_bulb --rect"]

restart_timer = 2
def start_script():
    # if cmd:
    run(command, check=True)
    try:
        # Make sure 'python' command is available
        run('python3 train.py', check=True) 
    except:
        # Script crashed, lets restart it!
        handle_crash()

def handle_crash():
    sleep(restart_timer)  # Restarts the script after 2 seconds
    start_script()

start_script()