# arm6_mujoco - command sheet

Project folder (inside Ubuntu/WSL): `~/robotics_projects/arm6_mujoco`
Python environment: `venv_arm6`      Windows Downloads as seen from Ubuntu: `/mnt/c/Users/YOUR_WINDOWS_USER/Downloads/`

---------------------------------------------------------------------------
## 1. Open the project (every time)

1. Start menu -> **Ubuntu**  (a terminal opens, prompt looks like `user@PC:~$`)
2. Go to the project:
   ```
   cd ~/robotics_projects/arm6_mujoco
   ```
3. Activate the Python environment (the prompt then starts with `(venv_arm6)`):
   ```
   source ../venv_arm6/bin/activate
   ```
   If that says "No such file", find where you created it, then activate that path:
   ```
   find ~ -maxdepth 3 -name activate -path "*venv_arm6*"
   source <the path printed above>
   ```
4. Open VS Code on this folder (still in the Ubuntu terminal):
   ```
   code .
   ```
   (first time after a restart it may take a few seconds: "VS Code Server" starts)
5. In VS Code, if it uses the wrong Python (error `No module named 'mujoco'`):
   `Ctrl+Shift+P` -> **Python: Select Interpreter** -> choose the one containing `venv_arm6`.
   Use the terminal inside VS Code with ``Ctrl+` `` - it activates the environment by itself.

Quick check that everything works:
```
python3 -c "import mujoco, scipy; print(mujoco.__version__)"      # prints 3.17.0 (or similar)
ls                                                                 # you should see arm6_pick.xml, pick_place.py, controllers/ ...
```

---------------------------------------------------------------------------
## 2. Watch the robot (live MuJoCo window)

```
python3 watch_pick_place.py                   # asks which controller to start with
python3 watch_pick_place.py all               # every controller does the same task in turn
python3 watch_pick_place.py mpc               # one controller: pd | pid | pdg | ct | cti | mpc
python3 watch_pick_place.py ct --cond mismatch    # model-error test (nominal | mismatch | weak)
python3 watch_pick_place.py pid --seed 5      # different random tasks
```
While the window is open, click the Ubuntu terminal, type and press Enter:

| command | meaning |
|---|---|
| `pd` `pid` `pdg` `ct` `cti` `mpc` | switch controller, SAME task replayed |
| `all` | every controller on the current task |
| `n` | new random task |
| `r` | replay current task |
| `task 0.40 0.10 0.35 -0.15` | your own task: cube (x y) -> goal (x y) in metres, optional yaw as 5th number |
| `help` / `q` | list commands / quit |

Reachable area: full 360 deg ring, 0.30-0.50 m from the base axis.
Close the MuJoCo window (or type `q`) to stop.

Record a video instead of a window (file appears in the project folder):
```
python3 watch_pick_place.py all --video all.mp4 --n 2
```

---------------------------------------------------------------------------
## 3. Run the experiments (numbers, no window)

```
python3 run_pick_place.py 50                  # PD, PID, PD+gravity, CT, CT+I on 50 random tasks   (~2 min)
python3 run_pick_place.py 40 pid ct           # only some controllers
python3 run_pick_place.py 40 --speed 3        # 3x faster motion
python3 run_pick_place.py 40 --payload 1      # unknown 1 kg payload on the tool

python3 run_model_based.py 20                 # PID / CT / CT+I / MPC under 3 conditions (~10 min, MPC is slow)
python3 run_model_based.py 20 nominal mismatch
python3 run_model_based.py 20 weak            # weak motors, 3x speed
python3 run_model_based.py 10 weak pid mpc    # subset

python3 benchmark_classical.py --cond nominal varied stress --seeds 0 1 2 --tasks 10 --out results/classical_raw.csv   # full classical benchmark (~40 min, MPC dominates)
python3 benchmark_classical.py --cond varied --seeds 0 --tasks 3 --ctrl pid cti --out results/quick.csv                 # quick subset
python3 summarize_benchmark.py "results/classical_raw*.csv"          # builds results/classical_benchmark.md (mean +- std across seeds)
```

---------------------------------------------------------------------------
## 4. Rebuild the model (only if you changed CAD / gen files)

```
python3 gen_pick_scene.py          # regenerates arm6_pick.xml (do this after replacing gen_pick_scene.py)
python3 view.py                    # plain viewer of the arm model (if present)
```

---------------------------------------------------------------------------
## 5. Copy downloaded files into the project

Files downloaded from Claude land in Windows Downloads. From the project folder:
```
cp /mnt/c/Users/YOUR_WINDOWS_USER/Downloads/pick_place.py .
cp /mnt/c/Users/YOUR_WINDOWS_USER/Downloads/gen_pick_scene.py .
cp /mnt/c/Users/YOUR_WINDOWS_USER/Downloads/watch_pick_place.py .
cp /mnt/c/Users/YOUR_WINDOWS_USER/Downloads/run_model_based.py .
cp /mnt/c/Users/YOUR_WINDOWS_USER/Downloads/plant.py .
cp /mnt/c/Users/YOUR_WINDOWS_USER/Downloads/mpc.py /mnt/c/Users/YOUR_WINDOWS_USER/Downloads/model.py \
   /mnt/c/Users/YOUR_WINDOWS_USER/Downloads/computed_torque.py /mnt/c/Users/YOUR_WINDOWS_USER/Downloads/computed_torque_utils.py \
   /mnt/c/Users/YOUR_WINDOWS_USER/Downloads/__init__.py controllers/
```
If a file was downloaded twice Windows names it `pick_place (1).py`: use quotes, e.g.
`cp "/mnt/c/Users/YOUR_WINDOWS_USER/Downloads/pick_place (1).py" pick_place.py`

---------------------------------------------------------------------------
## 6. Project layout

```
arm6_mujoco/
  arm6.xml, arm6_pick.xml, meshes/, cad_props.json     robot model (arm6_pick.xml = arm + gripper + cube + goal)
  gen_pick_scene.py                                    builds arm6_pick.xml
  ik.py                                                inverse kinematics
  pick_place.py                                        planner + pick-and-place sequence
  plant.py                                             "true" robot with optional model error
  controllers/  pd.py pid.py pd_gravity.py computed_torque.py mpc.py model.py ...
  watch_pick_place.py                                  live viewer with controller choice
  run_pick_place.py  run_model_based.py                experiments
```

---------------------------------------------------------------------------
## 7. Common problems

| symptom | fix |
|---|---|
| `ModuleNotFoundError: No module named 'mujoco'` | environment not active: `source ../venv_arm6/bin/activate` (or pick the interpreter in VS Code) |
| `No module named 'scipy'` | `pip install scipy` (with the environment active) |
| `can't open display` / no MuJoCo window | needs WSLg (Windows 11 or recent Windows 10 + `wsl --update` in PowerShell, then restart Ubuntu); video mode works without it |
| `FileNotFoundError: arm6_pick.xml` | you are in the wrong folder: `cd ~/robotics_projects/arm6_mujoco`, or run `python3 gen_pick_scene.py` |
| `AttributeError ... payload` or `_set_preview` | an old `pick_place.py` is in use: copy the newest one (section 5) |
| EGL traceback at the end of a video run | harmless |
| window very slow with MPC | normal, MPC is heavy; use `pid`/`ct` for smooth viewing |

---------------------------------------------------------------------------
## 8. Stage 2-4 commands (hybrid network, adaptive, learned policies)

```
# Stage 2-3: network training and evaluation
python3 train_live.py --headless --tasks 200 --history --out nn_hist.npz        # add --noise-train for the noise-aware network
# trained networks (models/): stage2_nohist_s{1,2,3}.npz  stage3_hist_s{1,2,3}.npz  stage3_noiseaware_s{1,2,3}.npz  stage2_nn_150.npz
python3 eval_shift.py --shifts both heavy friction mixed --seeds 10 11 12 --tasks 3 --noise 1e-4 1e-2 \
    --cti w60=60 --adaptive asl60=60,64 --nets hist=models/stage3_hist_s1.npz,models/stage3_hist_s2.npz,models/stage3_hist_s3.npz --nets-wn 60 --out results/fair/example.csv
python3 eval_shift.py --delay 5 ...                                              # control delay in simulation steps (5 = 10 ms)

# Stage 4: learned policies
python3 check_env.py --episodes 40                                               # scripted expert inside the RL environment (~2 min)
python3 imitate.py --iters 8 --out runs/il                                       # DAgger imitation (~20 min); then: cp runs/il/policy.npz runs/il_policy.npz
python3 imitate.py --eval runs/il_policy.npz --episodes 40                       # evaluate a policy in four conditions
python3 eval_policies.py --episodes 40 --workers 4 --expert --policy il=runs/il_policy.npz \
    --policy ppo1=runs/cluster_ppo/ppo_s1/last_policy.npz                         # fair comparison on unseen tasks
python3 record_stage4.py --policy runs/cluster_ppo/ppo_s2/last_policy.npz         # expert vs learned policy video
```

Cluster (RBCCPS: partition `debug`, account `YOUR_ACCOUNT`, conda env `rl`):
```
rsync -av --exclude 'venv*' --exclude '__pycache__' --exclude '*.mp4' ./ YOUR_USER@CLUSTER_HOST:~/arm6_mujoco/
sbatch run_ppo.sbatch          # PPO, seeds 1-5          squeue -u $USER      tail -f logs/ppo_*_1.out
sbatch eval_ppo.sbatch         # fair evaluation, prints two tables in logs/eval_*.out
rsync -av YOUR_USER@CLUSTER_HOST:~/arm6_mujoco/runs/ppo_s*/ ~/robotics_projects/arm6_mujoco/runs/cluster_ppo/   # bring results back
```
