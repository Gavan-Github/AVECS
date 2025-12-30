import random
import numpy as np
import torch
import wandb

import os
import csv
from simulation2.cyber_space.RSUModel import RSU
from simulation2.cyber_space.CarModel import Car
from collections import OrderedDict

from simulation2.cyber_space.TaskModel import Task
from simulation2.utils import generate_vehicle_rsu_data

base_folder='..//preprocess//virtual_data//process_data'

# -------------------------------
# 定义一个虚拟的多智能体环境
# -------------------------------
class DummyMultiAgentEnv:

    def reset(self):
        config=wandb.config
        num_vehicles = config.num_vehicles
        num_rsus = config.num_rsus
        vehicle_speed = config.vehicle_speed
        generate_vehicle_rsu_data(
            num_vehicles=num_vehicles,
            num_rsus=num_rsus,
            vehicle_speed=vehicle_speed,
            time_frames=config.max_time_frame,
            output_path="../preprocess/virtual_data")

        self.time_frame = 0
        for car_id in self.car_list.keys():
            self.is_running[car_id] = True

        for car in self.car_list.values():
            car.reset()
        for rsu in self.rsu_list.values():
            rsu.reset()

        car_masks = [1 if car.STATE == Car.IDLE else 0 for car in self.car_list.values()]
        rsu_masks = [1 if rsu.STATE == RSU.IDLE else 0 for rsu in self.rsu_list.values()]

        car_v2r_states,rsu_states,flattened_states=self.sense_state()
        # 全局状态由各个智能体的观测拼接而成
        return car_v2r_states,rsu_states,flattened_states,car_masks, rsu_masks


    def get_AOI_rsu(self):
        AOIs=[]
        for rsu in self.rsu_list.values():
            AOIs.append(rsu.AOI(self.time_frame)*self.time_slot)
        return AOIs


    def __init__(self,date,offload_model):
        self.time_slot = 0.5  # s
        Task.time_slot=self.time_slot
        self.time_frame = 0.0
        self.rsu_list = OrderedDict()
        self.car_list = OrderedDict()
        # 一系列参数
        self.phi = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        # 聚合间隔
        self.aggregation_interval = 60

        # 读取rsu的数据
        with open(os.path.join(base_folder, f'{date}/car_in_rsu/RSU_coordinates.csv'), 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                rsu_id = int(row[0])
                # 获得rsu的x，y坐标
                rsu_x, rsu_y = float(row[1]), float(row[2])
                self.rsu_list[rsu_id] = RSU(
                    rsu_id=rsu_id, rsu_x=rsu_x, rsu_y=rsu_y,
                    date=date,offload_model=offload_model)

        self.min_time_frame, self.max_time_frame = self.rsu_list[rsu_id].min_time_frame, self.rsu_list[
            rsu_id].max_time_frame
        #rsu智能体的数量
        self.num_rsu_agents=len(self.rsu_list)
        for rsu in self.rsu_list.values():
            rsu.set_rsu_count(count=self.num_rsu_agents)

        # 看哪些车在跑
        self.is_running = {}
        date_folder = os.path.join(base_folder, date)
        for car_file in os.listdir(date_folder):
            # 只读取车辆文件
            if car_file.endswith('.csv'):
                car_id = car_file[:-4]
                self.car_list[car_id] = Car(
                    car_id=car_id, date=date,
                    offload_model=offload_model)
                self.is_running[car_id] = True
                #最开始随便连接一个rsu
                self.car_list[car_id].connect_to(rsu)

        #Car智能体的数量
        self.num_car_agents=len(self.car_list)

        #车辆行为的维度
        self.n_car_actions = 2
        #rsu行为的维度
        self.n_rsu_actions = self.num_rsu_agents

        car_states, rsu_states, global_state=self.sense_state()
        self.obs_car_dim=len(car_states[0])
        self.obs_rsu_dim=len(rsu_states[0])
        self.global_state_dim=len(global_state)


    def sense_state(self):
        time_frame = self.time_frame
        time_slot = self.time_slot
        # 当前车辆的状态们
        car_states =[]
        for car in self.car_list.values():
            car_states.append(
                car.sense_state(time_frame=time_frame, time_slot=time_slot))
        # 存储rsu的状态们
        rsu_states = []
        # 获得当前rsu
        for rsu_id in self.rsu_list:
            rsu = self.rsu_list[rsu_id]
            rsu_state = rsu.sense_state()
            rsu_states.append(rsu_state)

        combined_rsu_states=np.concatenate(rsu_states)

        # 存储rsu的状态们
        rsu_states = []
        # 获得当前rsu
        for rsu_id in self.rsu_list:
            rsu = self.rsu_list[rsu_id]
            rsu_state =np.concatenate([rsu.get_rsu_id(mode='one_shot'),combined_rsu_states])
            rsu_states.append(rsu_state)

        # car_v2r_states=[np.concatenate([car_state,combined_rsu_states]) for car_state in car_states]
        car_v2r_states=car_states

        # 将 car_states 和 rsu_states 组合并拉平成一维数组
        combined_states = np.concatenate([np.concatenate(car_states),combined_rsu_states])
        flattened_states = combined_states.flatten()
        return car_v2r_states,rsu_states,flattened_states

    def step_all_task_queue(self):
        time_frame = self.time_frame
        time_slot = self.time_slot
        for rsu_id, rsu in self.rsu_list.items():
            # 更新等待队列中的任务时间，并移除超时任务
            expired_tasks = rsu.task_wait_queue.step(time_frame, time_slot)
            # 找到每个任务都是谁做的边缘卸载
            for task in expired_tasks:
                if task.offload_rsu_v2r:
                    # 谁做了这个卸载，就惩罚谁
                    task.offload_rsu_v2r.drop_task(task)
                else:
                    rsu.drop_task(task)

        for car_id, car in self.car_list.items():
            # 更新任务等待队列中的任务时间，并移除超时任务
            expired_tasks = car.task_wait_queue.step(time_frame, time_slot)
            if expired_tasks:
                #这个就不做惩罚了
                pass
                # car.drop_tasks(
                #     dropped_tasks=expired_tasks)

            # 更新消息队列中的任务时间，并移除超时任务
            expired_tasks = car.task_download_queue.step(time_frame, time_slot)
            if expired_tasks:
                car.drop_tasks(
                    dropped_tasks=expired_tasks)

    def car_step(self, car, action):
        time_frame = self.time_frame
        time_slot = self.time_slot
        # 目前在哪些rsu范围内
        connected_rsu_ids = car.get_connectedRSUs(time_frame)
        # 假设rsu没有发生切换
        handover = False
        # 先看此刻车辆是否离开了之前的rsu范围
        if car.connected_rsu is None or car.distance_to(car.connected_rsu, time_frame) > 200:
            # 就要连接新的rsu,计算吸引力
            F = []
            for rsu_id in connected_rsu_ids:
                attr_m = car.get_m() / self.rsu_list[rsu_id].get_m()
                attr_P = 1
                attr_R = car.get_R_up(time_frame=time_frame, rsu=self.rsu_list[rsu_id])
                attr_F = self.phi[1] * attr_m / (self.phi[2] * attr_P + self.phi[3] / attr_R)
                F.append(attr_F)

            index = F.index(max(F))
            rsu_id_max_F = connected_rsu_ids[index]
            # 选择有最大吸引力的rsu去链接
            car.connect_to(self.rsu_list[rsu_id_max_F])
            handover = True

        # 如果到达时间到了，生成新任务并加入任务队列
        while time_frame >= car.next_task_arrival_time:
            new_task = car.generate_task(time_frame)

            # 检查任务是否可以被添加到等待队列中（容量限制）
            removed_tasks = car.task_wait_queue.add_task(new_task)
            if removed_tasks:
                car.drop_tasks(dropped_tasks=removed_tasks)
            # 生成下一个任务的到达时刻
            car.next_task_arrival_time += random.expovariate(car.task_arrival_rate) * time_slot

        if car.STATE == Car.IDLE:
            # # 先查看等待队列中有没有任务要下载
            # task = car.task_download_queue.get_task_with_min_remaining_time()
            # # 如果有任务要下载，那么先下载任务
            # if task:
            #     car.start_download(task=task)
            #     return
            if not car.task_download_queue.empty():
                car.start_download()
                return


            # 从任务等待队列中找到一个任务这里不做删除
            task = car.task_wait_queue.get_task_with_min_remaining_time(
                time_frame, mode='look')
            if task:
                # 开启本地卸载
                if action == 0:
                    car.start_compute(time_frame=time_frame)
                # 开启边缘卸载
                else:
                    car.start_offload( time_frame=time_frame)


        # 如果是有任务要进行计算
        elif car.STATE == Car.COMPUTING:
            compute_ok, overtime = car.current_task.compute_one_time_slot(
                f_ex=car.f_ex,
                time_slot=time_slot)
            # 本地计算完成
            if compute_ok:
                car.end_compute()
            elif overtime:  # 如果任务超时了
                car.drop_task()
        # 在下载，卸载的任意一态发生切换，都会导致任务直接丢弃
        elif handover:
            if car.current_task:
                car.drop_task()

        # 处理当前正在执行的任务
        elif car.STATE == Car.OFFLOADING:
            # 获得此刻的卸载速率
            R_up = car.get_R_up(time_frame)
            # 卸载一个时间帧的数据
            offload_ok, overtime = car.current_task.offload_one_time_slot(
                R_up=R_up,
                time_slot=time_slot)
            if (offload_ok):
                car.end_offload()
            elif overtime:  # 如果任务超时了
                car.drop_task()

        # 把当前需要下载的任务一口气全部下载了
        elif car.STATE == Car.DOWNLOADING:
            for task in car.task_download_queue.get_cache():
                car.end_download(task)

    def rsu_step(self, rsu, action):
        time_frame = self.time_frame
        time_slot = self.time_slot

        # 先进行一个时间帧的计算
        completed_tasks, overtime_tasks = rsu.task_compute_queue.execute_tasks(
            time_frame=time_frame,
            time_slot=time_slot)
        if completed_tasks:
            rsu.complete_task(completed_tasks=completed_tasks)
            rsu.STATE = RSU.IDLE

        elif overtime_tasks:
            # 找到每个任务都是谁做的边缘卸载
            for task in overtime_tasks:
                if task.offload_rsu_v2r:
                    # 谁做了这个卸载，就惩罚谁
                    task.offload_rsu_v2r.drop_task(task)
                else:
                    rsu.drop_task(task)
            rsu.STATE = RSU.IDLE

        if rsu.STATE == RSU.IDLE:
            # 从任务等待队列中找到一个任务这里不做删除
            task = rsu.task_wait_queue.get_task_with_min_remaining_time(
                time_frame, mode='look')
            # 队列为空的时候就什么都不用做
            if task is None:
                return

            # 如果已经发生了二次卸载，或者当前网络告诉我们适合在本地计算1,说明就是在自己的这个rsu进行计算,
            if task.action_r2r is not None or action==rsu.rsu_id:
                # 把这个任务从任务队列中取出来
                task = rsu.task_wait_queue.get_task_with_min_remaining_time(
                    time_frame, mode='remove')

                rsu.start_edge_compute(
                    task=task,
                    time_frame=time_frame)

            # 如果没有发生二次卸载，说明该任务等待一个卸载策略
            else:
                # 从任务等待队列中找到一个任务
                task = rsu.task_wait_queue.get_task_with_min_remaining_time(
                    time_frame, mode='remove')
                # 静态卸载rsu
                offload_r2r_rsu = self.rsu_list[action]
                # 开启边缘计算模型
                task.start_r2r_offload(
                    action_r2r=action,
                    time_frame=time_frame,
                offload_r2r_rsu=offload_r2r_rsu)

    def send_task_from_task_offload_ok_to_task_wait_queue(self):
        for rsu_id, rsu in self.rsu_list.items():
            for task in rsu.task_offload_ok:
                rsu.task_wait_queue.add_task(task)
            rsu.task_offload_ok.clear()

    def step(self,car_actions,rsu_actions):
        time_frame = self.time_frame
        done=False

        # 更新所有等待队列，并获得所有rsu,状态
        self.step_all_task_queue()

        # 使用 lambda 表达式过滤出需要删除的键
        del_car_set = list(
            filter(lambda car_id: time_frame > self.car_list[car_id].max_time_frame, self.car_list.keys()))

        # 删除这些键
        for car_id in del_car_set:
            self.is_running[car_id] = False
            done=True

        for i, (car_id, car) in enumerate(self.car_list.items()):
            if self.is_running[car_id]:
                self.car_step(car=car,action=car_actions[i])

        for i, (rsu_id, rsu) in enumerate(self.rsu_list.items()):
            self.rsu_step(rsu=rsu,action=rsu_actions[i])

        # 最后进行真正的卸载操作
        self.send_task_from_task_offload_ok_to_task_wait_queue()

        #时间帧增加
        self.time_frame += self.time_slot
        # 下一时刻的观测值观测依然随机生成
        car_states, rsu_states, global_state=self.sense_state()

        car_rewards = [
            car.reward(self.time_frame) if self.is_running[car_id] else 0 for car_id, car in self.car_list.items()
        ]

        rsu_rewards = [
            rsu.reward(self.time_frame) for rsu in self.rsu_list.values()
        ]
        car_masks=[1 if car.STATE==Car.IDLE else 0 for car in self.car_list.values()]
        rsu_masks=[1 if rsu.is_action_masked() else 0 for rsu in self.rsu_list.values()]

        return car_states, rsu_states, global_state, car_rewards, rsu_rewards, done, \
               car_masks, rsu_masks