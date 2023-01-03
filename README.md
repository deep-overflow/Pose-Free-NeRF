# Pose-Free-NeRF
All implementations are based on the codes of SRT and RUST.<br/>
The following changes are needed :<br/>
1. DataLoader for Synthetic Dataset. (Done)
2. Trainer to train the PoseNetwork.(Done)
3. MLP to estimate the final pose. (Done)

You can run the script by :<br>
python main.py configs/posenet.yaml (path to configuration file)

Adjust the wandb info at configs/posenet.yaml for wandb.