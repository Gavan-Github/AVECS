import os
import csv
import math
import random
from random import randint

import wandb

from simulation2.data_structure.cache import TaskWaitCache
from simulation2.cyber_space.TaskModel import Task
import numpy as np

checkpoint_dir='../preprocess/NJUST_data/checkpoint'
# 目录路径
directory = '../preprocess/virtual_data'
class Car:
    IDLE = 0
    OFFLOADING=1
    COMPUTING=2
    DOWNLOADING=3

    def reset(self):
        self.u_t=0
        # 车辆状态
        self.STATE = Car.IDLE
        self.task_wait_queue.reset()
        self.task_download_queue.reset()
        self.task_offload_ok =[]
        #惩罚项
        self.punish = 0
        self.rd = 0
        # 当前执行的任务
        self.current_task = None
        #下一次任务到达时间
        self.next_task_arrival_time = random.expovariate(self.task_arrival_rate)

        # 重新加载位置信息
        self.location_data,self.rsu_conn_data, self.min_time_frame, self.max_time_frame=self.load_car_data(self.car_id,self.date)

    # 重置惩罚项
    def reset_reward_punish(self):
        self.rd = 0
        self.punish = 0

    def add_punish(self, value):
        self.punish += value

    def add_reward(self,value):
        self.rd+=value

    def AOI(self, time_frame):
        return time_frame - self.u_t

    def reward(self, time_frame):
        # r = 1 - (self.punish + self.AOI(time_frame)) / self.nu
        # r=-(self.punish + self.AOI(time_frame))
        r=self.rd-self.punish#-self.AOI(time_frame)
        self.reset_reward_punish()
        return r

    def __init__(self, car_id, date,state_dim=10,action_dim=2,
                b_up=10e6,       #上传带宽      Hz
                b_down=20e6,     #下载带宽      MHz
                g_0=1e-4,        #路径损失常数
                dis_star=1,      #参考距离      m
                N_0=3.981e-21,   #噪声功率      W/Hz
                p_up=0.1,        #上传功率      W
                p_down=0.1,      #下载功率      W
                theta=4,         #路径损失指数
                zeta=1,
                tau=3,
                f_range=
                [0.6e9,0.8e9,1.0e9,1.8e9],#车辆计算能力   cycles/s  rsu是80e9
                nu=20,
                offload_model="MAPPO"
                 ):

        self.b_up=b_up
        self.b_down=b_down
        self.g_0=g_0
        self.dis_star=dis_star
        self.N_0=N_0
        self.p_up=p_up
        self.p_down=p_down
        self.theta=theta
        self.nu=nu

        # 车辆名
        self.car_id = car_id
        # 车辆状态
        self.STATE = Car.IDLE

        # 车辆数据文件
        self.date=date
        # 车辆位置信息,与rsu连接信息
        self.location_data,self.rsu_conn_data, self.min_time_frame, self.max_time_frame=self.load_car_data(car_id,date)

        #车辆计算能力
        self.f_ex=f_range[randint(0,len(f_range)-1)]
        #执行功率
        self.p_ex=zeta*(self.f_ex)**tau

        # 初始化任务等待队列
        self.task_wait_queue=TaskWaitCache(max_capacity=2e9)  # 设定缓存最大容量为 2 Gb
        # 当前执行的任务
        self.current_task=None
        #当前连接的RSU
        self.connected_rsu=None


        self.task_download_queue=TaskWaitCache(max_capacity=200e9)#这个队列假设永远不满

        config = wandb.config
        self.byzantine_attack = config.byzantine_attack
        self.byzantine_defense = config.byzantine_defense
        self.task_arrival_rate = config.task_arrival_rate  # lambda 参数
        self.next_task_arrival_time = random.expovariate(self.task_arrival_rate)

        self.offload_model=offload_model

    def get_connected_rsu(self,mode='rsu'):
        if mode=='rsu_id':
            return self.connected_rsu.rsu_id
        else:
            return self.connected_rsu

    def load_car_data(self, car_id, date):
        file_path = os.path.join(directory,'process_data', date, car_id + '.csv')
        with open(file_path, 'r',encoding='utf-8') as f:
            reader = csv.reader(f)
            # 初始化位置信息数组和 RSU 连接信息数组
            location_data = {}
            rsu_conn_data = {}
            # 读取表头，跳过
            header = next(reader)
            for row in reader:
                # time_frame 作为索引访问数组
                time_frame = float(row[0])
                car_x, car_y = float(row[1]), float(row[2])
                # 填充位置数据和 RSU 连接数据
                location_data[time_frame] = [car_x, car_y]
                conn_rsu = [int(rsu_id) for rsu_id in row[4:]]  # 从第4列开始是 RSU 信息
                rsu_conn_data[time_frame] = conn_rsu

        time_frames = location_data.keys()
        return location_data, rsu_conn_data,min(time_frames), max(time_frames)

    def get_location(self,time_frame):
        if time_frame<=self.max_time_frame:
            return self.location_data[time_frame]
        else:
            return self.location_data[self.max_time_frame]

    def get_connectedRSUs(self,time_frame):
        return self.rsu_conn_data[time_frame]

    def start_offload(self,time_frame):
        #从任务等待队列中找到一个任务
        task=self.task_wait_queue.get_task_with_min_remaining_time(time_frame)
        # 必须要有任务
        if task:
            #把车辆当前任务修改为task
            self.current_task=task
            #记录车辆开始任务卸载
            self.current_task.start_offload_v2r(
                rsu=self.connected_rsu,
                time_frame=time_frame)
            #改变车辆状态
            self.STATE=Car.OFFLOADING

    def end_offload(self):
        self.connected_rsu.offload(self.current_task)
        self.current_task=None
        self.STATE=Car.IDLE

    def start_compute(self,time_frame):
        # 从任务等待队列中找到一个任务
        task = self.task_wait_queue.get_task_with_min_remaining_time(time_frame)
        #必须要有任务
        if task:
            #把车辆当前任务修改为task
            self.current_task=task
            # 记录车辆开始任务卸载
            self.current_task.start_local_compute(time_frame=time_frame)
            # 改变车辆状态
            self.STATE = Car.COMPUTING

    def end_compute(self):
        self.current_task.on_completed()
        # 更新ux  (t)
        if self.u_t < self.current_task.arrive_time:
            self.u_t = self.current_task.arrive_time

        self.add_reward(self.current_task.remaining_time())
        # 删除本地任务
        self.current_task = None
        # 任务进入空闲状态
        self.STATE = Car.IDLE

    def start_download(self, task):
        # 必须要有任务
        if task:
            self.current_task=task
            self.STATE=Car.DOWNLOADING

    def start_download(self):
        self.STATE = Car.DOWNLOADING

    def end_download(self,task):
        # 更新u(t)
        if self.u_t < task.arrive_time:
            self.u_t = task.arrive_time

        self.add_reward(task.remaining_time())
        task.on_completed()
        # 任务进入空闲状态
        self.STATE = Car.IDLE

    def connect_to(self,rsu):
        if rsu==None:
            return
        #要连接到新的rsu
        self.connected_rsu=rsu

    def distance_to(self,rsu,time_frame):
        [rsu_x,rsu_y]=rsu.get_location()
        [car_x,car_y]=self.get_location(time_frame)
        return math.sqrt((rsu_x-car_x)**2+(rsu_y-car_y)**2)

    #获得卸载速率
    def get_R_up(self,time_frame,rsu=None):
        if rsu==None:
            rsu=self.connected_rsu
        dis=self.distance_to(rsu,time_frame)
        R_up=self.b_up*math.log2(1+self.g_0*(self.dis_star/dis)**self.theta*self.p_up/(self.b_up*self.N_0))
        return R_up

    #获得下载速率
    def get_R_down(self,time_frame,rsu=None):
        if rsu == None:
            rsu = self.connected_rsu
        dis = self.distance_to(rsu, time_frame)
        R_down=self.b_down*math.log2(1+self.g_0*(self.dis_star/dis)**self.theta*self.p_down/(self.b_down*self.N_0))
        return R_down


    def compute_one_slot(self,time_slot):
        return self.current_task.compute_one_time_slot(time_slot)

    def generate_task(self, time_frame):
        task=Task.generate_task(self,time_frame)
        return task

    # 计算自己的质量
    def get_m(self):
        m=self.task_wait_queue.get_total_D_ex()/self.f_ex
        return max(m,0.5)

    def sense_state(self, time_frame, time_slot):
        #获得连接的rsu的one_shot编号
        rsu_one_shot_id=self.connected_rsu.get_rsu_id(mode="one_shot")
        # 计算车辆的质量
        m = self.get_m()
        # 获取任务缓存状态
        task_wait_queue_state = self.task_wait_queue.sense_state()
        # 下载队列缓存状态
        task_download_queue_state=self.task_download_queue.sense_state()
        # 获取前20个时间帧的位置（包括当前时间帧）
        location_history = []
        for i in range(20):
            # 计算每个时间帧的时间索引
            frame = time_frame - i * time_slot
            if frame < self.min_time_frame:
                break  # 如果时间帧超出最小帧范围，则停止
            # 获取该时间帧下的车辆位置
            loc = self.get_location(frame)
            location_history+=loc

        # 如果时间帧少于20个，则用[0, 0]填充
        while len(location_history) < 20*2:
            location_history=[location_history[0],location_history[1]]+location_history

        # 返回完整的状态信息
        return rsu_one_shot_id+[m]+location_history+ task_wait_queue_state+task_download_queue_state

    def drop_task(self,dropped_task=None):
        # 如果没有告诉你是哪些任务被丢弃了，那就说明丢弃的是正在计算的任务
        if dropped_task == None:
            self.current_task.on_dropped()
            self.current_task = None
            self.STATE = Car.IDLE
        else:
            dropped_task.on_dropped()
            self.add_punish(dropped_task.total_T())


    def drop_tasks(self,dropped_tasks):
        for task in dropped_tasks:
            task.on_dropped()
            self.add_punish(task.total_T())


    def send_message(self,task,message):
        if message=='download':
            self.task_download_queue.add_task(task)



