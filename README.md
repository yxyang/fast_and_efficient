# Fast and Efficient Locomotion via Learned Gait Transitions

This repository contains the code for the paper ["Fast and Efficient Locomotion via Learned Gait Transitions"](https://arxiv.org/abs/2104.04644). Apart from implementation of the paper's results, this repository also contains the entire suite of software interface for the A1 quadruped robot, including:

* A reasonably accurate simulation of the A1 in PyBullet.
* The real-robot interface in Python, which allows sim-to-real switch to be done simply with a command line flag (`--use_real_robot`).
* An implementation of the [Convex MPC Controller](https://ieeexplore.ieee.org/document/8594448), which achieves robust locomotion on the robot.

## Reproducing Paper Results
### Setup the environment

First, make sure the environment is setup by following the steps in the [Setup](#Setup) section.

### Evaluating Policy

We provide a learned gait policy in `example_checkpoints`. You can check it out by running:
```bash
python -m src.agents.cmaes.eval_cmaes --logdir=example_checkpoint/lin_policy_plus_150.npz --show_gui=True --save_data=False --save_video=False
```

Please check the Python file for all available command line flags. Note that when running on the real robot (`--use_real_robot=True`), the code requires a Xbox-like gamepad as an E-stop. See the [Code Structure](#Code-Structure) section for further details.

### Training

To speed up training, we use [ray](https://www.ray.io/) to parallelize rollouts. First, start `ray` by running:
```bash
ray start --head --port=6379 --num-cpus=[NUM_CPUS] --redis-password=1234
```

and start training by running:
```bash
python -m src.agents.cmaes.train_cmaes --config src/agents/cmaes/configs/gait_change_deluxe.py --experiment_name="exp"
```

You can then see the checkpoints and tensorboard logs in the `logs` folder and evaluate the trained policy using `eval_cmaes` as described above.


## Setup

### Software
It is recommended to create a separate virtualenv or conda environment to avoid conflicting with existing system packages. The required packages have been tested under Python 3.8.5, though they should be compatiable with other Python versions.

The following three steps are required to set up the Python environment. We tried to consolidate all into a simple `setup.py` but it does not seem easy.

1. First, install all dependent packages by running:

   ```bash
   pip install -r requirements.txt
   ```

2. Second, install the C++ binding for the convex MPC controller:

   ```bash
   python setup.py install
   ```

3. Lastly, build and install the interface to Unitree's SDK. The Unitree [repo](https://github.com/unitreerobotics/unitree_legged_sdk) has been releasing new SDK versions. For convenience, we have included the version that we used in `third_party/unitree_legged_sdk`.

   First, make sure the required packages are installed, following Unitree's [guide](https://github.com/unitreerobotics/unitree_legged_sdk). Most nostably, please make sure to install `Boost` and `LCM`:

   ```bash
   sudo apt install libboost-all-dev liblcm-dev
   ```

   Then, go to `third_party/unitree_legged_sdk` and create a build folder:
   ```bash
   cd third_party/unitree_legged_sdk
   mkdir build && cd build
   ```

   Now, build the libraries and move them to the main directory by running:
   ```bash
   cmake ..
   make
   mv robot_interface* ../../..
   ```

### Additional Setup for Real Robot
Follow these steps if you want to run policies on the real robot.

1. **Setup correct permissions for non-sudo user**

   Since the Unitree SDK requires memory locking and high process priority, root priority with `sudo` is usually required to execute commands. To run the SDK without `sudo`, write the following to `/etc/security/limits.d/90-unitree.conf`:

   ```bash
   <username> soft memlock unlimited
   <username> hard memlock unlimited
   <username> soft nice eip
   <username> hard nice eip
   ```

   Log out and log back in for the above changes to take effect.

2. **Connect to the real robot**

   Connect from computer to the real robot using an Ethernet cable, and set the computer's IP address to be `192.168.123.24` (or anything in the `192.168.123.X` range that does not collide with the robot's existing IPs). Make sure you can ping/SSH into the robot's TX2 computer (by default it is `unitree@192.168.123.12`).

3. **Test connection**

   Start up the robot. After the robot stands up, enter joint-damping mode by pressing L2+B on the remote controller. Then, run the following:
   ```bash
   python -m src.robots.a1_robot_exercise_example --use_real_robot=True
   ```

   The robot should be moving its body up and down following a pre-set trajectory. Terminate the script at any time to bring the robot back to joint-damping position.


## Code Structure

### Simulation

The simulation infrastructure is mostly a lightweight wrapper around `pybullet` that provides convenient APIs for locomotion purposes. The main files are:
* `src/robots/robot.py` contains general robot API.
* `src/robots/a1.py` contains A1-specific configurations.
* `src/robots/motors.py` contains motor configurations.

### Real Robot Interface

The real robot infrastructure is mostly implemented in `robots/a1_robot.py`, which invokes the C++ interface via pybind to communicate with Unitree SDKs. In addition:

* `src/robots/a1_robot_state_estimator.py` provides a simple KF-based implementation to estimate the robot's speed.

* `src/robots/gamepad_reader.py` contains a simple wrapper to read x-box like gamepads, which is useful for remote-controlling the robot. The gamepads that have been tested to work includes [Logitech F710](https://www.amazon.com/Logitech-Wireless-Nano-Receiver-Controller-Vibration/dp/B0041RR0TW/ref=sr_1_1?keywords=logitech+f710&qid=1637563719&qsid=134-5696489-9387867&sr=8-1&sres=B0041RR0TW%2CB00FZP2O18%2CB079QMW4N8%2CB01M1LWHWL%2CB0057GAF3E%2CB00CJAEX5M%2CB003VAHYQY%2CB008GOUQ3I%2CB07PQ62D7V%2CB07T8JKVNT%2CB07HG51ZYK%2CB087LXCTFJ%2CB07NSSPV9S%2CB07L4BM851%2CB07DX5TYQN%2CB07DS73GVX) and [GameSir T1s](https://www.amazon.com/GameSir-T1s-Wireless-Bluetooth-Controller/dp/B08GCFW4DW/ref=sr_1_3?keywords=gamesir+t1s&qid=1637563742&qsid=134-5696489-9387867&sr=8-3&sres=B08GCFW4DW%2CB06XBXHG41%2CB082WYXRLB%2CB07HG51ZYK%2CB07HQT3GVM%2CB07CPFL5SK%2CB07PQ62D7V%2CB07DHFTPV3%2CB088GQY8FH%2CB07SR1P14R%2CB08RJ2NWQ7%2CB08H7MBRYQ%2CB082ZZ5X8S%2CB07ST8DL8R%2CB07PXJC64S%2CB08P54DQTN), though any gamepad with similar functionality should likely work.

### Convex MPC Controller

The `src/convex_mpc_controller` folder contains a Python implementation of MIT's [Convex MPC Controller](https://ieeexplore.ieee.org/document/8594448). Some notable files include:

* `torque_stance_leg_controller_mpc.py` sets up and solves the MPC problem for stance legs.
* `mpc_osqp.cc` actually sets up the QP and calls a QP library to solve it.
* `raibert_swing_leg_controller.py` controlls swing legs.

### Gait Change Environment
The hierarchical gait change environment is defined in `src/intermediate_envs`.

### CMA-ES Agent

The code to train the policy using CMA-ES is in `src/agents`.

## Credits

We thank authors of the following repos for their contributions to our codebase:

* After many iterations, the simulation infrastructure is now based on [Limbsim](https://github.com/UWRobotLearning/limbsim). We thank Rosario Scalise for his contributions.

* The original convex MPC implementation is derived from [motion_imitation](https://github.com/google-research/motion_imitation) with modifications.

* The underlying simulator is [PyBullet](https://pybullet.org/wordpress/).
