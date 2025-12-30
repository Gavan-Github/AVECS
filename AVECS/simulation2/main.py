import torch
import os
import numpy as np
import random
import pandas as pd
from utils import generate_vehicle_rsu_data

from simulation2.cyber_space.TaskModel import Task
from simulation2.network.DDQN import DDQNSystem
from simulation2.network.A2C import A2C
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

from simulation2.network.MAPPO import MAPPO
from simulation2.cyber_space.DigitalTwinMutiAgentEnv import DummyMultiAgentEnv
import wandb

# 登录 wandb（第一次运行时会提示输入 API key）
wandb.login(key="11f84ef73d9cca839dbed895eb4d75638692fbd6")

# -------------------------------
# 设置设备（GPU / CPU）
# --------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# 主训练循环
# -------------------------------
def main():
    wandb.init()
    # 创建保存图像和模型的目录
    figs_dir = "../output/mappo/figs"
    models_dir = "../output/mappo/models"
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    # 获取超参数
    config = wandb.config

    num_vehicles=config.num_vehicles
    num_rsus=config.num_rsus
    vehicle_speed=config.vehicle_speed
    generate_vehicle_rsu_data(
        num_vehicles=num_vehicles,
        num_rsus=num_rsus,
        vehicle_speed=vehicle_speed,
        time_frames=config.max_time_frame,
        output_path="../preprocess/virtual_data")


    data_set =f"{num_rsus}r{num_vehicles}v"
    num_episodes = config.num_episodes
    #总共走的实践步
    update_timestep=config.update_timestep
    federated_timestep = config.federated_timestep  # 每 federated_interval 个 episode 进行一次联邦聚合
    lr = config.lr
    offload_model = config.offload_model
    byzantine_attack=config.byzantine_attack
    byzantine_defense=config.byzantine_defense
    byzantine_ratio=config.byzantine_ratio
    task_arrival_rate=config.task_arrival_rate

    discount_factor=config.discount_factor

    group_name=config.group_name

    if offload_model == "MAPPO":
        wandb.run.name = f"group_name_lr{lr:.0e}_df{discount_factor:.2e}_{data_set}_sp{vehicle_speed}_lam{task_arrival_rate}_{offload_model}"
        if byzantine_attack:
            wandb.run.name += f"_BA"
        if byzantine_defense:
            wandb.run.name += f"_DT"
    else:
        wandb.run.name = f"{data_set}_{offload_model}"

    last_car_avg_reward = 0
    last_rsu_avg_reward = 0

    # 创建一个 Table
    table = wandb.Table(columns=["episode","car_avg_reward", "car_min_reward","car_max_reward","rsu_avg_reward","rsu_min_reward","rsu_max_reward"])
    # 创建一个 Table
    table_infos = wandb.Table(
        columns=['episode','completed_ratio', 'dropped_ratio',  'throughput', 'avg_completed_delay', 'avg_dropedd_delay'])

    env = DummyMultiAgentEnv(
        date=data_set,
        offload_model=offload_model
    )
    car_obs, rsu_obs, global_state,car_masks, rsu_masks = env.reset()


    table_avg_aoi=wandb.Table(
        columns=['episode']+[f'rsu_{rsu_id} aoi' for rsu_id in env.rsu_list.keys()]
    )

    # 收集rsu的AOI信息

    if offload_model == "MAPPO":
        car_offload_model = MAPPO(
            num_agents=env.num_car_agents,
            obs_dim=env.obs_car_dim,
            n_actions=env.n_car_actions,
            global_state_dim=env.global_state_dim,
            lr=lr,
            agent_type="car",
            byzantine_attack=byzantine_attack,
            byzantine_defense=byzantine_defense,
            byzantine_ratio=byzantine_ratio
            # global_actor_path=os.path.join(models_dir, "car_global_actor.pth"),
            # global_critic_path=os.path.join(models_dir, "car_global_critic.pth")
        )

        rsu_offload_model = MAPPO(
            num_agents=env.num_rsu_agents,
            obs_dim=env.obs_rsu_dim,
            n_actions=env.n_rsu_actions,
            global_state_dim=env.global_state_dim,
            lr=lr,
            agent_type="rsu",
            byzantine_attack=byzantine_attack,
            byzantine_defense=byzantine_defense
            # global_actor_path=os.path.join(models_dir, "rsu_global_actor.pth"),
            # global_critic_path=os.path.join(models_dir, "rsu_global_critic.pth")
        )
    elif offload_model=="A2C":
        car_offload_model = A2C(
            num_agents=env.num_car_agents,
            obs_dim=env.obs_car_dim,
            n_actions=env.n_car_actions,
            lr=lr,
        )
        rsu_offload_model = A2C(
            num_agents=env.num_rsu_agents,
            obs_dim=env.obs_rsu_dim,
            n_actions=env.n_rsu_actions,
            lr=lr,
        )
    elif offload_model=="DDQN":
        car_offload_model = DDQNSystem(
            num_agents=env.num_car_agents,
            obs_dim=env.obs_car_dim,
            n_actions=env.n_car_actions,
            lr=lr,
        )
        rsu_offload_model = DDQNSystem(
            num_agents=env.num_rsu_agents,
            obs_dim=env.obs_rsu_dim,
            n_actions=env.n_rsu_actions,
            lr=lr,
        )

    for episode in range(num_episodes):
        timestep = 0
        # 存储一个 episode 内的轨迹数据
        car_obs_list = []  # 每步每个car的局部观测
        car_actions_list = []  # 每步每个car的动作
        car_log_probs_list = []  # 每步每个car的 log 概率
        car_rewards_list = []  # car全局奖励（各智能体奖励的均值）
        car_rewards_list_episode = []  # car全局奖励（各智能体奖励的均值）

        rsu_obs_list = []  # 每步每个智能体的局部观测
        rsu_actions_list = []  # 每步每个智能体的动作
        rsu_log_probs_list = []  # 每步每个智能体的 log 概率
        rsu_rewards_list = []  # car全局奖励（各智能体奖励的均值）
        rsu_rewards_list_episode = []  # car全局奖励（各智能体奖励的均值）

        dones_list = []
        global_states_list = []  # 每步全局状态

        car_obs, rsu_obs, global_state,car_masks, rsu_masks = env.reset()
        done = False

        rsu_aoi_history=[]
        while not done:
            timestep+=1

            # 将 car_obs 和 rsu_obs 转换为 tensor 并送到 GPU
            # car_obs_tensor = torch.tensor(car_obs, dtype=torch.float32).to(device)
            car_obs_tensor = torch.tensor(np.array(car_obs), dtype=torch.float32).to(device)

            # rsu_obs_tensor = torch.tensor(rsu_obs, dtype=torch.float32).to(device)
            rsu_obs_tensor = torch.tensor(np.array(rsu_obs), dtype=torch.float32).to(device)

            # 计算每个车辆的action
            car_actions, car_log_probs = car_offload_model.select_action(car_obs_tensor,car_masks)
            car_obs_list.append(car_obs)
            car_actions_list.append(car_actions)
            car_log_probs_list.append(car_log_probs)

            # 计算每个rsu的action
            rsu_actions, rsu_log_probs = rsu_offload_model.select_action(rsu_obs_tensor,rsu_masks)
            rsu_obs_list.append(rsu_obs)
            rsu_actions_list.append(rsu_actions)
            rsu_log_probs_list.append(rsu_log_probs)

            # 存储总体状态
            global_states_list.append(global_state)

            # 获取下一时刻的状态和奖励值
            next_car_obs, next_rsu_obs, next_global_state, car_rewards, rsu_rewards, done,car_masks, rsu_masks = env.step(car_actions, rsu_actions)

            # 存储 car 和 rsu 的奖励序列
            car_rewards_list.append(np.mean(car_rewards))
            rsu_rewards_list.append(np.mean(rsu_rewards))
            car_rewards_list_episode.append(np.mean(car_rewards))
            rsu_rewards_list_episode.append(np.mean(rsu_rewards))
            dones_list.append(1 if done else 0)

            # 记录下一时刻状态
            car_obs, rsu_obs = next_car_obs, next_rsu_obs
            global_state = next_global_state

            #记录一下rsu的AOI
            rsu_aoi_history.append(env.get_AOI_rsu())

            # 每 federated_interval 个 episode 对所有 Actor 进行联邦聚合
            if (timestep) % federated_timestep == 0 and byzantine_defense:
                car_offload_model.federated_aggregation(episode=episode)

            # update if its time
            if timestep % update_timestep == 0:

                # 构造轨迹数据字典
                car_trajectories = {
                    'obs': car_obs_list,
                    'global_states': global_states_list,
                    'actions': car_actions_list,
                    'log_probs': car_log_probs_list,
                    'rewards': car_rewards_list,
                    'dones': dones_list,
                }

                rsu_trajectories = {
                    'obs': rsu_obs_list,
                    'global_states': global_states_list,
                    'actions': rsu_actions_list,
                    'log_probs': rsu_log_probs_list,
                    'rewards': car_rewards_list,
                    'dones': dones_list,
                }

                # 更新模型
                car_offload_model.update(car_trajectories)
                rsu_offload_model.update(rsu_trajectories)

                car_obs_list = []  # 每步每个car的局部观测
                car_actions_list = []  # 每步每个car的动作
                car_log_probs_list = []  # 每步每个car的 log 概率
                car_rewards_list = []  # car全局奖励（各智能体奖励的均值）

                rsu_obs_list = []  # 每步每个智能体的局部观测
                rsu_actions_list = []  # 每步每个智能体的动作
                rsu_log_probs_list = []  # 每步每个智能体的 log 概率
                rsu_rewards_list = []  # car全局奖励（各智能体奖励的均值）

                dones_list = []
                global_states_list = []  # 每步全局状态

        # 保存结果
        car_avg_reward = np.mean(car_rewards_list_episode)
        car_min_reward = np.min(car_rewards_list_episode)
        car_max_reward = np.max(car_rewards_list_episode)
        wandb.log({
            "car_reward/episode": episode,
            "car_reward/Average": car_avg_reward,
            "car_reward/Min": car_min_reward,
            "car_reward/Max": car_max_reward
        })

        rsu_avg_reward = np.mean(rsu_rewards_list_episode)
        rsu_min_reward = np.min(rsu_rewards_list_episode)
        rsu_max_reward = np.max(rsu_rewards_list_episode)
        wandb.log({
            "rsu_reward/episode": episode,
            "rsu_reward/Average": rsu_avg_reward,
            "rsu_reward/Min": rsu_min_reward,
            "rsu_reward/Max": rsu_max_reward
        })

        table.add_data(episode,car_avg_reward, car_min_reward, car_max_reward,
                       rsu_avg_reward,rsu_min_reward,rsu_max_reward)

        last_car_avg_reward,last_rsu_avg_reward=car_avg_reward,rsu_avg_reward

        print(
            f"Episode {episode + 1} : car avg reward = {car_avg_reward:.6f}, rsu avg reward = {rsu_avg_reward:.6f}")



        completed_ratio,dropped_ratio,throughput,avg_completed_delay,avg_dropedd_delay=Task.get_Task_info(env.max_time_frame*env.time_slot)
        table_infos.add_data(episode,completed_ratio,dropped_ratio, throughput,
                             avg_completed_delay, avg_dropedd_delay)

        #计算得到所有rsu的平均aoi，在这个episode
        avg_aoi=np.mean(np.array(rsu_aoi_history),axis=0)
        avg_aoi=[episode]+avg_aoi.tolist()
        table_avg_aoi.add_data(*avg_aoi)
        Task.reset()

    wandb.log({"Training Table": table})
    wandb.log({"Info Table":table_infos})
    wandb.log({"Avg AoI Table":table_avg_aoi})
    # 保存全局模型参数
    car_global_actor_path = os.path.join(models_dir, "car_global_actor.pth")
    car_global_critic_path = os.path.join(models_dir, "car_global_critic.pth")
    car_offload_model.save_model(car_global_actor_path, car_global_critic_path)

    rsu_global_actor_path = os.path.join(models_dir, "rsu_global_actor.pth")
    rsu_global_critic_path = os.path.join(models_dir, "rsu_global_critic.pth")
    rsu_offload_model.save_model(rsu_global_actor_path, rsu_global_critic_path)

    if byzantine_defense:
        wandb.log({"Byzantine defense table": car_offload_model.byzantine_defense_table})
    wandb.finish()

"""
export PYTHONPATH=$PYTHONPATH:/home/power/WangChen/Digital-Twin-Byzantine2
cd /home/power/WangChen/Digital-twin-Byzantine2 
/home/power/anaconda3/envs/pytorch/bin/python -u simulation2/main.py

"""
if __name__ == "__main__":
    version=3

    sweep_config = {
        "method": "grid",  # 网格搜索
        "name": f"性能测试MAPPO",
        "parameters": {
            "group_name": {"values": ["compare"]},
            "offload_model": {"values": ["MAPPO"]},
            "num_vehicles": {"values": [25]},
            "num_rsus": {"values": [5]},
            "vehicle_speed": {"values": [20]},
            "num_episodes": {"values": [500]},
            "update_timestep": {"values": [50]},
            "federated_timestep": {"values": [100]},
            "byzantine_ratio": {"values": [0.2]},
            "completely_trust_ratio": {"values": [0.1]},
            "partly_trust_ratio": {"values": [0.2]},
            "byzantine_attack": {"values": [True,False]},
            "byzantine_defense": {"values": [True,False]},
            "task_arrival_rate": {"values": [0.4]},
            "max_time_frame": {"values": [1600]},

            # 搜索参数
            "lr": {"values": [1e-7]},
            "discount_factor": {"values": [0.9]}
        }
    }

    # 创建 Sweep
    #sweep_id = wandb.sweep(sweep_config, project="Digital Twin Enabled Byzantine attack system")
    #wandb.agent(sweep_id, function=main)  # 运行 4 次，每次不同参数组合

    sweep_config = {
        "method": "grid",  # 网格搜索
        "name": f"性能测试(DDQN,A2C)",
        "parameters": {
            "group_name": {"values": ["compare"]},
            "offload_model": {"values": ['DDQN',"A2C"]},
            "num_vehicles": {"values": [25]},
            "num_rsus": {"values": [5]},
            "vehicle_speed": {"values": [20]},
            "num_episodes": {"values": [500]},
            "update_timestep": {"values": [50]},
            "federated_timestep": {"values": [100]},
            "byzantine_ratio": {"values": [0.2]},
            "completely_trust_ratio": {"values": [0.1]},
            "partly_trust_ratio": {"values": [0.2]},
            "byzantine_attack": {"values": [False]},
            "byzantine_defense": {"values": [False]},
            "task_arrival_rate": {"values": [0.4]},
            "max_time_frame": {"values": [1600]},
            "lr": {"values": [1e-7]},
            "discount_factor":{"values":[0.9]}
        }
    }

    # 创建 Sweep
    sweep_id = wandb.sweep(sweep_config, project="Digital Twin Enabled Byzantine attack system")
    wandb.agent(sweep_id, function=main)  # 运行 4 次，每次不同参数组合


    sweep_config = {
        "method": "grid",  # 网格搜索
        "name": f"敏感度分析（学习率）",
        "parameters": {
            "group_name": {"values": ["compare"]},
            "offload_model": {"values": ["MAPPO"]},
            "num_vehicles": {"values": [25]},
            "num_rsus": {"values": [5]},
            "vehicle_speed": {"values": [20]},
            "num_episodes": {"values": [500]},
            "update_timestep": {"values": [50]},
            "federated_timestep": {"values": [100]},
            "byzantine_ratio": {"values": [0.2]},
            "completely_trust_ratio": {"values": [0.1]},
            "partly_trust_ratio": {"values": [0.2]},
            "byzantine_attack": {"values": [True]},
            "byzantine_defense": {"values": [True]},
            "task_arrival_rate": {"values": [0.4]},
            "max_time_frame": {"values": [1600]},

            # 搜索参数
            "lr": {"values": [1e-5,1e-6,1e-7,1e-8,1e-9]},
            "discount_factor": {"values": [0.9]}
        }
    }
    # 创建 Sweep
    #sweep_id = wandb.sweep(sweep_config, project="Digital Twin Enabled Byzantine attack system")
    #wandb.agent(sweep_id, function=main)  # 运行 4 次，每次不同参数组合


    sweep_config = {
        "method": "grid",  # 网格搜索
        "name": f"敏感度分析（折扣因子）",
        "parameters": {
            "group_name": {"values": ["compare"]},
            "offload_model": {"values": ["MAPPO"]},
            "num_vehicles": {"values": [25]},
            "num_rsus": {"values": [5]},
            "vehicle_speed": {"values": [20]},
            "num_episodes": {"values": [500]},
            "update_timestep": {"values": [50]},
            "federated_timestep": {"values": [100]},
            "byzantine_ratio": {"values": [0.2]},
            "completely_trust_ratio": {"values": [0.1]},
            "partly_trust_ratio": {"values": [0.2]},
            "byzantine_attack": {"values": [True]},
            "byzantine_defense": {"values": [True]},
            "task_arrival_rate": {"values": [0.4]},
            "max_time_frame": {"values": [1600]},

            # 搜索参数
            "lr": {"values": [ 1e-7]},
            "discount_factor": {"values":[0.85,0.875, 0.9,0.95,0.975]}
        }
    }
    # # 创建 Sweep
    # sweep_id = wandb.sweep(sweep_config, project="Digital Twin Enabled Byzantine attack system")
    # wandb.agent(sweep_id, function=main)  # 运行 4 次，每次不同参数组合






