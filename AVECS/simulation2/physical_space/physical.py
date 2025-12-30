import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

import os
import csv
from simulation2.cyber_space.RSUModel import RSU
from simulation2.cyber_space.CarModel import Car
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simulation.cyber_space.TaskModel import Task

base_folder = '..//..//preprocess//NJUST_data//process_data'
rsu_path = '..//..//preprocess//NJUST_data//raw_data//RSU_coordinates.csv'


class physical:
    def __init__(self,
                 date,
                 car_offload_model="RandomModel",
                 rsu_offload_model="RandomModel",
                 trainning=True):
        self.time_slot = 0.5  # s
        self.time_frame = 0.0
        self.rsu_list = {}
        self.car_list = {}
        # 一系列参数
        self.phi = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        # 聚合间隔
        self.aggregation_interval = 60
        self.trainning = trainning

        # 读取rsu的数据
        with open(rsu_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                rsu_id = int(row[0])
                # 获得rsu的x，y坐标
                rsu_x, rsu_y = float(row[1]), float(row[2])
                self.rsu_list[rsu_id] = RSU(rsu_id, rsu_x, rsu_y, date)

        self.min_time_frame, self.max_time_frame = self.rsu_list[rsu_id].min_time_frame, self.rsu_list[
            rsu_id].max_time_frame

        # 看哪些车在跑
        self.is_running = {}
        date_folder = os.path.join(base_folder, date)
        for car_file in os.listdir(date_folder):
            # 只读取车辆文件
            if car_file.endswith('.csv'):
                car_id = car_file[:-4]
                self.car_list[car_id] = Car(car_id, date)
                self.is_running[car_id] = True

        # 感知一下rsu状态和env状态
        rsu_states = self.sense_rsu_states()
        env_state = self.sense_env_state(car=self.car_list[car_id], rsu_states=rsu_states)
        car_state_dim = len(env_state)
        car_action_dim = 2
        rsu_state_dim = len(rsu_states) + len(self.rsu_list)
        rsu_action_dim = len(self.rsu_list)

        self.car_state_dim = car_state_dim
        self.car_action_dim = car_action_dim
        self.car_offload_model = car_offload_model

        # 初始化一下r2r模型
        RSU.init_offload_r2r_model(
            rsu_state_dim=rsu_state_dim,
            rsu_action_dim=rsu_action_dim,
            model=rsu_offload_model
        )

        # 初始化rsu上的的v2r模型
        for rsu_id, rsu in self.rsu_list.items():
            rsu.init_offload_v2r_model(
                car_state_dim=car_state_dim,
                car_action_dim=car_action_dim,
                model=car_offload_model
            )

        # 初始化车辆的v2r模型
        for car_id, car in self.car_list.items():
            car.init_offload_v2r_model(
                car_state_dim=car_state_dim,
                car_action_dim=car_action_dim,
                model=car_offload_model
            )

    # 新的一天
    def new_day(self, date):
        # 看哪些车在跑
        self.is_running = {}
        date_folder = os.path.join(base_folder, date)
        for car_file in os.listdir(date_folder):
            # 只读取车辆文件
            if car_file.endswith('.csv'):
                car_id = car_file[:-4]
                self.car_list[car_id] = Car(car_id, date)
                self.is_running[car_id] = True
                # 初始化车辆的v2r模型

        for car_id, car in self.car_list.items():
            car.init_offload_v2r_model(
                car_state_dim=self.car_state_dim,
                car_action_dim=self.car_action_dim,
                model=self.car_offload_model
            )

    def sense_env_state(self, car, rsu_states=None):
        car_state = self.sense_car_state(car)
        if (rsu_states == None):
            rsu_states = self.sense_rsu_states()
        return car_state + rsu_states

    def sense_car_state(self, car):
        time_frame = self.time_frame
        time_slot = self.time_slot
        # 当前车辆的状态
        car_state = car.sense_state(time_frame=time_frame, time_slot=time_slot)
        # 生成链接的rsu_id的one_shot编码
        k_star = [0] * len(self.rsu_list)
        # 如果什么都没有的话，说明车辆没有在rsu范围内
        if car.get_connected_rsu():
            k_star[car.get_connected_rsu(mode='rsu_id')] = 1
        car_state = car_state + k_star

        return car_state

    def sense_rsu_states(self):
        time_frame = self.time_frame
        time_slot = self.time_slot
        # 存储rsu的状态们
        rsu_states = []
        # 获得当前rsu
        for rsu_id in self.rsu_list:
            rsu = self.rsu_list[rsu_id]
            rsu_state = rsu.sense_state()
            rsu_states += rsu_state
        return rsu_states

    def step_all_task_queue(self):
        time_frame = self.time_frame
        time_slot = self.time_slot

        rsu_states = self.sense_rsu_states()
        for rsu_id, rsu in self.rsu_list.items():
            # 更新等待队列中的任务时间，并移除超时任务
            expired_tasks = rsu.task_wait_queue.step(time_frame, time_slot)
            if expired_tasks:
                rsu.drop_task(
                    time_frame=time_frame,
                    time_slot=time_slot,
                    where_drop='rsu_wait_queue_overtime',
                    rsu_states=rsu_states,
                    dropped_tasks=expired_tasks)

        for car_id, car in self.car_list.items():
            # 更新任务等待队列中的任务时间，并移除超时任务
            expired_tasks = car.task_wait_queue.step(time_frame, time_slot)
            if expired_tasks:
                car.drop_task(
                    time_frame=time_frame,
                    where_drop='car_wait_queue_overtime',
                    next_state=None,
                    dropped_tasks=expired_tasks)

            # 更新消息队列中的任务时间，并移除超时任务
            expired_tasks = car.task_download_queue.step(time_frame, time_slot)
            if expired_tasks:
                env_state = self.sense_env_state(car, rsu_states=rsu_states)
                car.drop_task(
                    time_frame=time_frame,
                    where_drop='car_message_queue_overtime',
                    next_state=env_state,
                    dropped_tasks=expired_tasks)

        rsu_states = self.sense_rsu_states()
        return rsu_states

    def car_step(self, car, rsu_states=None):
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
                attr_P = car.get_offload_punish(
                    rsu=self.rsu_list[rsu_id],
                    time_frame_start=time_frame - 20 * time_slot,
                    time_frame_end=time_frame)
                attr_R = car.get_R_up(time_frame=time_frame, rsu=self.rsu_list[rsu_id])
                attr_F = self.phi[1] * attr_m / (self.phi[2] * attr_P + self.phi[3] / attr_R)
                F.append(attr_F)

            index = F.index(max(F))
            rsu_id_max_F = connected_rsu_ids[index]
            # 选择有最大吸引力的rsu去链接
            car.connect_to(self.rsu_list[rsu_id_max_F])
            handover = True

        # 感知当前状态
        cur_state = self.sense_env_state(car=car,
                                         rsu_states=rsu_states)
        x = 1
        # 如果到达时间到了，生成新任务并加入任务队列
        while time_frame >= car.next_task_arrival_time:
            new_task = car.generate_task(time_frame)
            # 检查任务是否可以被添加到等待队列中（容量限制）
            removed_tasks = car.task_wait_queue.add_task(new_task)
            if removed_tasks:
                car.drop_task(time_frame=time_frame,
                              next_state=cur_state,
                              where_drop='car_wait_queue_overtime',
                              dropped_tasks=removed_tasks)
            # 生成下一个任务的到达时刻
            car.next_task_arrival_time += random.expovariate(car.task_arrival_rate) * time_slot

        if car.STATE == Car.IDLE:
            # 先查看等待队列中有没有任务要下载
            task = car.task_download_queue.get_task_with_min_remaining_time()
            # 如果有任务要下载，那么先下载任务
            if task:
                car.start_download(task=task)
                return

            # 从任务等待队列中找到一个任务这里不做删除
            task = car.task_wait_queue.get_task_with_min_remaining_time(
                time_frame, mode='look')
            if task:
                # 根据当前状态选择策略
                action = car.choose_action(cur_state)
                # 开启本地卸载
                if action == 0:
                    car.start_compute(state=cur_state, time_frame=time_frame)
                # 开启边缘卸载
                else:
                    car.start_offload(state=cur_state, time_frame=time_frame)


        # 如果是有任务要进行计算
        elif car.STATE == Car.COMPUTING:
            compute_ok, overtime = car.current_task.compute_one_time_slot(
                f_ex=car.f_ex,
                time_slot=time_slot)
            # 本地计算完成
            if compute_ok:
                car.end_compute(
                    state=cur_state,
                    time_frame=time_frame)
            elif overtime:  # 如果任务超时了
                car.drop_task(time_frame=time_frame,
                              next_state=cur_state,
                              where_drop='car_compute_overtime')
        # 在下载，卸载的任意一态发生切换，都会导致任务直接丢弃
        elif handover:
            car.drop_task(time_frame=time_frame,
                          next_state=cur_state,
                          where_drop='handover')

        # 处理当前正在执行的任务
        elif car.STATE == Car.OFFLOADING:
            # 获得此刻的卸载速率
            R_up = car.get_R_up(time_frame)
            # car.current_task.D_ex==5356933110 and car.car_id=='T221212192300'
            # 卸载一个时间帧的数据
            offload_ok, overtime = car.current_task.offload_one_time_slot(
                R_up=R_up,
                time_slot=time_slot)
            if (offload_ok):
                car.end_offload()
                if self.time_frame not in car.connected_rsu.task_arrive_count:
                    car.connected_rsu.task_arrive_count[self.time_frame] = 0
                car.connected_rsu.task_arrive_count[self.time_frame] += 1

            elif overtime:  # 如果任务超时了
                car.drop_task(time_frame=time_frame,
                              next_state=cur_state,
                              where_drop='car_offload_overtime')

        # 如果当前没有任务，尝试从等待队列中取出新任务
        elif car.STATE == Car.DOWNLOADING:
            # 获得此刻的卸载速率
            R_down = car.get_R_down(time_frame)
            # 卸载一个时间帧的数据
            download_ok, overtime = car.current_task.download_one_time_slot(
                R_down=R_down,
                time_slot=time_slot)
            if (download_ok):
                car.end_download(
                    time_frame=time_frame,
                    state=cur_state
                )

            elif overtime:  # 如果任务超时了
                car.drop_task(time_frame=time_frame,
                              next_state=cur_state,
                              where_drop='car_download_overtime')

    def rsu_step(self, rsu, rsu_states=None):
        time_frame = self.time_frame
        time_slot = self.time_slot

        # 先进行一个时间帧的计算
        completed_tasks, overtime_tasks = rsu.task_compute_queue.execute_tasks(
            time_frame=time_frame,
            time_slot=time_slot)
        if completed_tasks:
            rsu.complete_task(completed_tasks=completed_tasks,
                              time_frame=time_frame,
                              rsu_states=rsu_states)
            rsu.STATE = RSU.IDLE

        elif overtime_tasks:
            # 丢弃任务
            rsu.drop_task(time_frame=time_frame,
                          time_slot=time_slot,
                          where_drop='rsu_compute_overtime',
                          rsu_states=rsu_states,
                          dropped_tasks=overtime_tasks)
            rsu.STATE = RSU.IDLE

        if rsu.STATE == RSU.IDLE:

            # 从任务等待队列中找到一个任务这里不做删除
            task = rsu.task_wait_queue.get_task_with_min_remaining_time(
                time_frame, mode='look')
            # 队列为空的时候就什么都不用做
            if task is None:
                return

            # 如果设定了卸载策略的话,说明就是在自己的这个rsu进行计算
            if task.action_r2r is not None:
                # 把这个任务从任务队列中取出来
                task = rsu.task_wait_queue.get_task_with_min_remaining_time(
                    time_frame, mode='remove')

                # 把该任务
                rsu.task_compute_queue.add_task(task)
                # 如果任务队列满了的话，说明rsu没有剩余资源去计算任务卸载了
                if (rsu.task_compute_queue.cache_full()):
                    rsu.STATE = RSU.ACTIVE

            # 如果没有进行卸载策略的判断，说明该任务等待一个卸载策略
            else:
                # 生成链接的rsu_id的one_shot编码
                k_one_shot = [0] * len(self.rsu_list)
                k_one_shot[rsu.rsu_id] = 1
                # 拼凑出r2r_state
                state_r2r = k_one_shot + rsu_states
                # 获得决策
                # 根据当前状态选择策略
                action = rsu.choose_action(state_r2r)

                # 从任务等待队列中找到一个任务
                task = rsu.task_wait_queue.get_task_with_min_remaining_time(
                    time_frame, mode='remove')

                # 当action等于自己的id时，就是本地卸载
                if action == rsu.rsu_id:
                    rsu.start_edge_compute(
                        state_r2r=state_r2r,
                        action_r2r=action,
                        task=task,
                        time_frame=time_frame)
                # 否则是进行边缘卸载
                else:
                    # 找到要写在的rsu
                    offload_r2r_rsu = self.rsu_list[action]
                    offload_r2r_rsu.offload(task)

    def send_task_from_task_offload_ok_to_task_wait_queue(self):
        for rsu_id, rsu in self.rsu_list.items():
            for task in rsu.task_offload_ok:
                rsu.task_wait_queue.add_task(task)
            rsu.task_offload_ok.clear()

    def step(self):

        time_frame = self.time_frame
        time_slot = self.time_slot

        # 如果到了聚集间隔
        if self.trainning and self.time_frame % self.aggregation_interval == 0 and self.time_frame != 0:
            for rsu_id, rsu in self.rsu_list.items():
                rsu.aggregate(
                    car_list=self.car_list,
                    time_frame=time_frame,
                    time_slot=time_slot,
                    aggregation_interval=self.aggregation_interval
                )

            for car_id, car in self.car_list.items():
                car.connect_to(car.connected_rsu)

        print(f"================{time_frame}/{self.max_time_frame}=================")

        # 更新所有等待队列，并获得所有rsu,状态
        rsu_states = self.step_all_task_queue()

        # 使用 lambda 表达式过滤出需要删除的键
        del_car_set = list(
            filter(lambda car_id: time_frame > self.car_list[car_id].max_time_frame, self.car_list.keys()))
        # 删除这些键
        for car_id in del_car_set:
            if self.is_running[car_id]:
                #
                print(f"{car_id}结束行驶")
            # del self.car_list[car_id]
            self.is_running[car_id] = False

        for car_id, car in self.car_list.items():
            if self.is_running[car_id]:
                self.car_step(car=car, rsu_states=rsu_states)
                if self.trainning:
                    # print(f"{car_id} start learning")
                    car.learn()

        for rsu_id, rsu in self.rsu_list.items():
            self.rsu_step(rsu=rsu, rsu_states=rsu_states)

        if self.trainning:
            print(f"RSU set start learning")
            RSU.learn()

        # 最后进行真正的卸载操作
        self.send_task_from_task_offload_ok_to_task_wait_queue()

    def plot_dropped_tasks_and_rewards(self, output_dir="../../output/figure/",
                                       model_mode=None,
                                       car_reward_filename="cars_average_rewards.png",
                                       car_task_stats_filename="cars_task_stats.png",
                                       rsu_output_filename="rsus_dropped_tasks_and_rewards.png",
                                       car_reward_data_filename="cars_average_rewards.csv",
                                       car_task_stats_data_filename="cars_task_stats.csv",
                                       rsu_data_filename="rsus_dropped_tasks_and_rewards.csv",
                                       window_size=25):
        model_mode = self.car_offload_model

        # 创建输出目录
        output_folder = os.path.join(output_dir, model_mode)
        os.makedirs(output_folder, exist_ok=True)

        # 时间帧列表
        time_frames = np.arange(self.min_time_frame, self.max_time_frame + self.time_slot, self.time_slot)

        # 统计车辆任务和奖励数据
        task_arrival_counts = []
        task_completion_counts = []
        task_dropped_counts = []
        reward_sums = []

        # 计算任务到达、任务完成、任务丢弃数量以及奖励总和
        for time_frame in time_frames:
            total_arrived_tasks = sum(car.task_arrive_count.get(time_frame, 0)
                                      for car in self.car_list.values()
                                      if time_frame <= car.max_time_frame)
            total_completed_tasks = sum(car.task_complete_count.get(time_frame, 0)
                                        for car in self.car_list.values()
                                        if time_frame <= car.max_time_frame)
            total_dropped_tasks = sum(len(car.dropped_tasks_local.get(time_frame, []))
                                      for car in self.car_list.values()
                                      if time_frame <= car.max_time_frame)
            total_reward = sum(sum(car.r_history.get(time_frame, []))
                               for car in self.car_list.values()
                               if time_frame <= car.max_time_frame)

            task_arrival_counts.append(total_arrived_tasks)
            task_completion_counts.append(total_completed_tasks)
            task_dropped_counts.append(total_dropped_tasks)
            reward_sums.append(total_reward)

        # 计算奖励时，去掉值为0的奖励进行平均
        non_zero_rewards = [r for r in reward_sums if r != 0]
        avg_reward = np.mean(non_zero_rewards) if non_zero_rewards else 0

        # 保存车辆任务和奖励数据到CSV
        with open(os.path.join(output_folder, car_task_stats_data_filename), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Time Frame", "Total Arrived Tasks", "Total Completed Tasks", "Total Dropped Tasks"])
            for i, time_frame in enumerate(time_frames):
                writer.writerow([time_frame, task_arrival_counts[i], task_completion_counts[i], task_dropped_counts[i]])

        with open(os.path.join(output_folder, car_reward_data_filename), mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Time Frame", "Total Rewards", "Average Reward"])
            for i, time_frame in enumerate(time_frames):
                writer.writerow([time_frame, reward_sums[i], avg_reward])

        # 计算平滑后的任务统计和奖励数据
        task_arrival_smooth = np.convolve(task_arrival_counts, np.ones(window_size) / window_size, mode='valid')
        task_completion_smooth = np.convolve(task_completion_counts, np.ones(window_size) / window_size, mode='valid')
        task_dropped_smooth = np.convolve(task_dropped_counts, np.ones(window_size) / window_size, mode='valid')
        reward_sums_smooth = np.convolve(reward_sums, np.ones(window_size) / window_size, mode='valid')

        # 计算平均奖励的滑动平均
        avg_reward_smooth = np.convolve([r for r in reward_sums if r != 0], np.ones(window_size) / window_size,
                                        mode='valid')

        # 绘制车辆的任务到达、完成和丢弃数量
        fig, axs = plt.subplots(2, 1, figsize=(12, 10))

        # 绘制任务统计
        axs[0].plot(time_frames[:len(task_arrival_smooth)], task_arrival_smooth, label='Total Arrived Tasks (Smoothed)',
                    color='tab:blue')
        axs[0].plot(time_frames[:len(task_completion_smooth)], task_completion_smooth,
                    label='Total Completed Tasks (Smoothed)', color='tab:green')
        axs[0].plot(time_frames[:len(task_dropped_smooth)], task_dropped_smooth, label='Total Dropped Tasks (Smoothed)',
                    color='tab:red')
        axs[0].set_title("Total Task Arrival, Completion, and Dropped Counts (Smoothed)")
        axs[0].set_xlabel("Time Frame")
        axs[0].set_ylabel("Task Count")
        axs[0].legend()
        axs[0].grid(True)

        # 绘制奖励总和（平滑后）
        axs[1].plot(time_frames[:len(reward_sums_smooth)], reward_sums_smooth, label='Smoothed Total Rewards',
                    color='tab:orange')
        axs[1].set_title("Total Rewards (Smoothed)")
        axs[1].set_xlabel("Time Frame")
        axs[1].set_ylabel("Total Rewards")
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()

        # 保存车辆图表
        fig.savefig(os.path.join(output_folder, car_reward_filename))

        # 清理图表
        plt.close(fig)

        # 统计RSU的任务和奖励数据
        for rsu in self.rsu_list.values():
            task_arrival_counts = []
            task_completion_counts = []
            task_dropped_counts = []
            reward_sums = []

            for time_frame in time_frames:
                task_arrival_counts.append(rsu.task_arrive_count.get(time_frame, 0))
                task_completion_counts.append(rsu.task_complete_count.get(time_frame, 0))
                task_dropped_counts.append(len(rsu.dropped_tasks_edge.get(time_frame, [])))
                reward_sums.append(sum(rsu.r_history.get(time_frame, [])))

            # 保存RSU数据到CSV（任务相关的部分）
            with open(os.path.join(output_folder, f"{rsu_data_filename}_{rsu.rsu_id}_task.csv"), mode='w',
                      newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Time Frame", "Total Arrived Tasks", "Total Completed Tasks", "Total Dropped Tasks"])
                for i, time_frame in enumerate(time_frames):
                    writer.writerow(
                        [time_frame, task_arrival_counts[i], task_completion_counts[i], task_dropped_counts[i]])

            # 保存RSU数据到CSV（奖励相关的部分）
            with open(os.path.join(output_folder, f"{rsu_data_filename}_{rsu.rsu_id}_reward.csv"), mode='w',
                      newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Time Frame", "Total Rewards"])
                for i, time_frame in enumerate(time_frames):
                    writer.writerow([time_frame, reward_sums[i]])

            # 计算平滑后的任务和奖励数据
            task_arrival_smooth = np.convolve(task_arrival_counts, np.ones(window_size) / window_size, mode='valid')
            task_completion_smooth = np.convolve(task_completion_counts, np.ones(window_size) / window_size,
                                                 mode='valid')
            task_dropped_smooth = np.convolve(task_dropped_counts, np.ones(window_size) / window_size, mode='valid')
            reward_sums_smooth = np.convolve(reward_sums, np.ones(window_size) / window_size, mode='valid')

            # 绘制RSU任务图表
            fig, axs = plt.subplots(2, 1, figsize=(12, 10))

            # 绘制任务统计
            axs[0].plot(time_frames[:len(task_arrival_smooth)], task_arrival_smooth, label='Arrived Tasks (Smoothed)',
                        color='tab:blue')
            axs[0].plot(time_frames[:len(task_completion_smooth)], task_completion_smooth,
                        label='Completed Tasks (Smoothed)', color='tab:green')
            axs[0].plot(time_frames[:len(task_dropped_smooth)], task_dropped_smooth, label='Dropped Tasks (Smoothed)',
                        color='tab:red')
            axs[0].set_title(f"RSU {rsu.rsu_id} - Task Arrival, Completion, and Dropped Counts (Smoothed)")
            axs[0].set_xlabel("Time Frame")
            axs[0].set_ylabel("Task Count")
            axs[0].legend()
            axs[0].grid(True)

            # 绘制奖励图表
            axs[1].plot(time_frames[:len(reward_sums_smooth)], reward_sums_smooth, label='Total Rewards (Smoothed)',
                        color='tab:orange')
            axs[1].set_title(f"RSU {rsu.rsu_id} - Total Rewards (Smoothed)")
            axs[1].set_xlabel("Time Frame")
            axs[1].set_ylabel("Total Rewards")
            axs[1].legend()
            axs[1].grid(True)

            plt.tight_layout()

            # 保存RSU图表
            fig.savefig(os.path.join(output_folder, f"{rsu_output_filename}_{rsu.rsu_id}.png"))

            # 清理图表
            plt.close(fig)


env = physical('2022-12-29',
               # car_offload_model="RandomModel",
               # rsu_offload_model="RandomModel",
               car_offload_model="DQN",
               rsu_offload_model="DQN",
               )

for env.time_frame in np.arange(env.min_time_frame, env.max_time_frame + env.time_slot, env.time_slot):
    env.step()

env.plot_dropped_tasks_and_rewards(
    window_size=50
)

"""
for t in range(max_time_frame)
    car step:
        检测车辆是否超出原来rsu的范围
        car task generate: car生成任务，先放入自己的等待队列中，如果任务等待队列已满，删除remain_time最小的任务

        此时车辆存在四个状态:
        IDLE: 
            如果先前车辆收到了任务下载message，则立刻进行任务下载，调整为DOWNLOAD态，如果收到了切换RSU的讯息，则在此刻切换RSU。
            否则，如果任务等待队列有任务，则感知状态，将任务送往模型计算action，本地计算则调整状态为COMPUTE，边缘计算则先将任务调整为OFFLOAD态。否则不进行任何操作。
        DOWNLOAD:
            T_tr_down+=1，下载一个时间帧，首先判断当前连接的RSU是否和上一帧一样，如果不一样，则发生了handover，产生handover惩罚。如果此时该任务超时，
            remain_time<0，修改状态为IDLE,该任务废弃。否则下载任务，如果下载到最后一个bit，存储任务的state,action,reward,next_state，调整任务状态为IDLE态。
        COMPUTE:
            T_ex+=1,计算一个时间帧，如果任务在此刻计算到最后一个比特，则代表任务计算完成，将该任务从本地CPU上拿下来，将车辆调整为IDLE态。如果此时该任务超时，remain_time<0，
            修改状态为IDLE,该任务废弃。
        OFFLOAD:
            T_tr_up+=1,卸载一个时间帧，首先判断当前连接的RSU是否和上一帧一样，如果不一样，则发生了handover，产生handover惩罚。如果此时该任务超时，remain_time<0，
            修改状态为IDLE,该任务废弃。否则卸载任务，如果卸载到最后一个bit，则会先将任务存在对应rsu的offload cache中。

    rsu step:
        rsu分为两个状态IDLE和ACTIVE态，rsu可以同时计算多个任务(5)，这里假设rsu task wait queue的容量无上限。


        对CPU中所有任务计算一个时间帧,T_ex+=1。如果有任务计算到最后一个bit，则向对应车辆发送一个可下载数据的message，并调整RSU为IDLE态。如果有任务发生超时，则将该任务废弃，
        同时发送超时message，并调整RSU为IDLE态。

        IDLE:

            从等待队列中，选取一个剩余时间最小的任务
                    如果该任务已经有卸载策略:
                        说明就该在这个rsu进行计算，直接送入计算队列中
                    否则
                        送入模型进行计算，产生卸载结果action，如果action[rsu_id]==1,说明就该在本地计算，直接送入计算队列中
                        否则将任务存在对应rsu的offload cache

    将缓存a和缓存b里的所有任务放到对应rsu的等待队列中




"""