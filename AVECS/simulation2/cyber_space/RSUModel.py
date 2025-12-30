import wandb

directory = '../preprocess/virtual_data'
import os
import csv
from simulation2.data_structure.cache import TaskWaitCache, TaskComputeCache

class RSU:
    IDLE = 0
    ACTIVE = 1

    def set_rsu_count(self,count):
        self.rsu_count = count

    def get_rsu_id(self,mode="normal"):
        if mode=="one_shot":
            # 生成链接的rsu_id的one_shot编码
            k_star = [0] * self.rsu_count
            k_star[int(self.rsu_id)]=1
            return k_star
        else:
            return self.rsu_id

    def reset(self):
        self.u_t=0
        # 车辆状态
        self.STATE = RSU.IDLE
        self.task_wait_queue.reset()
        self.task_compute_queue.reset()
        self.task_offload_ok =[]
        #惩罚项
        self.punish = 0

        self.rd = 0
        # 重新加载位置信息
        self.car_conn_data,self.min_time_frame,self.max_time_frame=self.load_rsu_data(self.rsu_id,self.date)

    #重置惩罚项
    def reset_reward_punish(self):
        self.punish=0

    def add_punish(self,value):
        self.punish+=value

    def add_reward(self,value):
        self.rd+=value

    def AOI(self,time_frame):
        return (time_frame-self.u_t)

    def reward(self,time_frame):
        # r=1-(self.punish+self.AOI(time_frame))/self.nu
        # r=-(self.punish + self.AOI(time_frame))
        r=self.rd-self.punish-self.AOI(time_frame)
        self.reset_reward_punish()
        return r

    #判断自己下一时刻是否需要有action
    def is_action_masked(self):
        return self.STATE==RSU.IDLE and not self.task_wait_queue.empty()

    def __init__(self,rsu_id,rsu_x,rsu_y,date,
                 f_ex=10e9,     #车辆计算能力   cycles/s  rsu是80e9
                 max_compute_num=4,
                 reward_mode="AOI",
                 nu=20,
                 offload_model="MAPPO"
                 ):
        self.rsu_id = rsu_id
        self.STATE=RSU.IDLE
        self.rsu_x,self.rsu_y=rsu_x,rsu_y
        self.date=date
        self.car_conn_data,self.min_time_frame,self.max_time_frame=self.load_rsu_data(rsu_id,date)
        self.f_ex = f_ex
        self.nu=nu

        #惩罚项
        self.punish = 0
        #奖励项
        self.rd = 0

        # 初始化任务等待队列
        self.task_wait_queue = TaskWaitCache(max_capacity=20e9)  # 设定缓存最大容量为 20 Gb
        # 当前执行的任务
        self.task_compute_queue =TaskComputeCache(f_ex=self.f_ex,max_compute_num=max_compute_num)

        # 卸载到该rsu上的任务，会暂存在这里，等待所有rsu还有车辆step结束后集中上传到等待队列
        self.task_offload_ok = []
        #用于计算AOI的
        self.u_t = 0

        config = wandb.config
        self.byzantine_attack = config.byzantine_attack
        self.byzantine_defense = config.byzantine_defense
        self.offload_model=offload_model

    def offload(self,task):
        self.task_offload_ok.append(task)

    def add_task(self,task):
        # 检查任务是否可以被添加到等待队列中（容量限制）
        removed_tasks = self.task_wait_queue.add_task(task)
        #这些任务会对RSU做出惩罚
        for task in removed_tasks:
            #谁做了这个卸载，就惩罚谁
            if task.offload_rsu_v2r:
                # 谁做了这个卸载，就惩罚谁
                task.offload_rsu_v2r.drop_task(task)
            else:
                self.drop_task(task)

        if removed_tasks:
            print(f"任务队列已满，移除了以下任务: {removed_tasks}")

    def drop_task(self,dropped_task):
        self.add_punish(dropped_task.total_T())
        # 哪个车辆把任务卸载到边缘服务器上，就惩罚谁
        dropped_task.create_car.drop_task(dropped_task)

    def load_rsu_data(self,rsu_id,date):
        file_path = os.path.join(directory, f'process_data/{date}/car_in_rsu/{rsu_id}.csv')
        with open(file_path, 'r',encoding='utf-8') as f:
            reader = csv.reader(f)
            # 初始化位置信息数组
            car_conn_data = {}
            # 读取表头，跳过
            header = next(reader)
            for row in reader:
                # 时间帧
                time_frame = float(row[0])
                # 此时有多少辆车
                conn_car_num=int(row[1])
                # 车辆id
                conn_car_ids=row[2:]
                # 范围内有哪些车
                car_conn_data[time_frame]=conn_car_ids

        time_frames = car_conn_data.keys()
        return car_conn_data,min(time_frames), max(time_frames)

    def get_location(self):
        return [self.rsu_x,self.rsu_y]

    def get_connectedCars(self,time_frame):
        return self.car_conn_data[time_frame]

    # 计算自己的质量
    def get_m(self):
        m = self.task_wait_queue.get_total_D_ex() / self.f_ex
        return max(m, 0.5)

    def send_message(self,car,MESSAGE):
        car.get_message()

    def sense_state(self):
        #位置
        rsu_x,rsu_y=self.get_location()
        #计算能力
        f_ex=self.f_ex/1e9
        # 计算自己的的质量
        m = self.get_m()
        # 获取任务缓存状态
        task_wait_queue_state = self.task_wait_queue.sense_state()
        # 获取任务计算队列的状态
        task_compute_queue_state=self.task_compute_queue.sense_state()

        # 返回完整的状态信息
        return [rsu_x,rsu_y,f_ex,m] + task_wait_queue_state+task_compute_queue_state


    def start_edge_compute(self,task,time_frame):
        task.offload_time_r2r=time_frame
        # 送入计算队列中
        self.task_compute_queue.add_task(task)
        # 如果任务队列满了的话，说明rsu没有剩余资源去计算任务卸载了
        if (self.task_compute_queue.cache_full()):
            self.STATE = RSU.ACTIVE

    def complete_task(self,completed_tasks):
        for task in completed_tasks:
            # 更新u(t)
            if self.u_t < task.arrive_time:
                self.u_t = task.arrive_time
            self.add_reward(task.remaining_time())
            #通知车辆可以下载任务了
            task.create_car.send_message(task=task,message='download')