# Reproducibility


<br>

## 1. Usage


<br>

### In terminal

```
# Run the main file (at the root of the project)
python main_TSP_edge_classification.py --dataset TSP  --log_path './logs/result_log`'   --config 'configs/TSP_LiteGT.json' # for CPU
python main_TSP_edge_classification.py --dataset TSP  --log_path './logs/result_log`'   --config 'configs/TSP_LiteGT.json' --gpu_id 0 # for GPU
```
The training and network parameters for each experiment is stored in a json file in the [`configs/`](../configs) directory.



<br>

## 2. Output, checkpoints and visualizations

Output results are located in the folder defined by the variable `out_dir` in the corresponding config file (eg. [`configs/TSP_LiteGT.json`](../configs/TSP_GraphTransformer_LapPE_TSP_500k_sparse_graph_BN.json) file).  

If `out_dir = 'out/TSP/'`, then 

#### 2.1 To see checkpoints and results
1. Go to`out/TSP/results` to view all result text files.
2. Directory `out/TSP/checkpoints` contains model checkpoints.

#### 2.2 To see the screen print information

1. Go to the logs directory, i.e. `logs/result_log`.

#### 2.3 To see the training logs in Tensorboard on local machine

1. Go to the out directory, i.e. `out/TSP/logs/`.
2. Run the commands
```
source activate graph_transformer
tensorboard --logdir='./' --port 6006
```
3. Open `http://localhost:6006` in your browser. Note that the port information (here 6006 but it may change) appears on the terminal immediately after starting tensorboard.


#### 2.4 To see the training logs in Tensorboard on remote machine
1. Go to the logs directory, i.e. `out/TSP/logs/`.
2. Run the [script](../scripts/TensorBoard/script_tensorboard.sh) with `bash script_tensorboard.sh`.
3. On your local machine, run the command `ssh -N -f -L localhost:6006:localhost:6006 user@xx.xx.xx.xx`.
4. Open `http://localhost:6006` in your browser. Note that `user@xx.xx.xx.xx` corresponds to your user login and the IP of the remote machine.



<br>

## 3. Reproduce results 


```
# At the root of the project 

# reproduce main results of TSP (Table 1, 2 in paper) 
bash scripts/TSP/script_main_TSP_edge_classification.sh 
bash scripts/SBMs/script_main_SBMs_node_classification_CLUSTER.sh 
bash scripts/SBMs/script_main_SBMs_node_classification_PATTERN.sh
```

Scripts are [located](../scripts/) at the `scripts/` directory of the repository.

 

 <br>


















<br><br><br>