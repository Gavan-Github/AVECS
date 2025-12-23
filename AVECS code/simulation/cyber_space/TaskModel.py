import random

c_avg=500#cycles/bit
d_avg=1#1Mb
class Task:
    time_slot=0

    # 全局出现过的任务数量
    total_task_count = 0
    def on_created(self):
        Task.total_task_count += 1

    # 全局丢弃任务数量
    total_task_dropped_count = 0
    # 全局丢弃任务占有时延
    total_task_dropped_total_T = 0
    def on_dropped(self):
        if not self.dropped:
            self.dropped=True
            Task.total_task_dropped_count+=1
            Task.total_task_dropped_total_T+=self.total_T()

    #全局完成的任务数量
    total_task_completed_count = 0
    # 全局出现过的任务计算量
    total_task_completed_bits = 0
    # 全局计算完毕的任务总计算时延
    total_task_completed_total_T = 0
    def on_completed(self):
        Task.total_task_completed_count += 1
        Task.total_task_completed_bits+=self.D_ex#*1e-6#转换为Mbits
        Task.total_task_completed_total_T+=self.total_T()

    @staticmethod
    def get_Task_info(system_total_seconds):
        #总共出现的任务量
        total_task_count=Task.total_task_count
        #总共丢弃的任务比率
        completed_ratio=Task.total_task_completed_count/total_task_count
        #总共丢弃的任务比率
        dropped_ratio=Task.total_task_dropped_count/total_task_count
        #整个系统吞吐量
        throughput=Task.total_task_completed_bits/system_total_seconds/1e6#Mbps
        #平均计算时延
        avg_completed_delay=Task.total_task_completed_total_T/Task.total_task_completed_count*Task.time_slot
        #平均丢弃时延
        avg_dropedd_delay=Task.total_task_dropped_total_T/Task.total_task_dropped_count*Task.time_slot

        return completed_ratio,dropped_ratio,throughput,avg_completed_delay,avg_dropedd_delay


    @staticmethod
    def reset():
        Task.total_task_count = 0

        Task.total_task_dropped_count=0
        Task.total_task_dropped_total_T=0

        Task.total_task_completed_count = 0
        Task.total_task_completed_bits = 0
        Task.total_task_completed_total_T = 0


    def __init__(self,car,arrive_time,
                 C_range=[1,30],                #任务所需计算资源  cycles/bit
                 D_up_range=[150e6,300e6],      #任务上传数据量    bits
                 D_down_range=[100e6,200e6],    #任务下载数据量    bits
                 max_T_range=[30,60],           #任务最大容忍时延  s

                 #rsu 80
                 ):

        #由哪个车辆创建的
        self.create_car=car
        #到达时刻
        self.arrive_time=arrive_time
        #完成时刻
        self.complete_time=0

        #v2r卸载时刻
        self.offload_time_v2r=-1
        self.offload_time_r2r=-1

        # v2r卸载到的rsu
        self.offload_rsu_v2r = None

        #是否废弃
        self.dropped=False

        #策略
        self.action_v2r=0
        self.action_r2r=None

        #目前已经上传数据量
        self.D_up_cur=0
        #目前已经下载的数据量
        self.D_down_cur=0
        #目前以及计算的数据量
        self.D_ex_cur=0

        # 任务每bit所需计算资源
        self.C=random.randint(C_range[0],C_range[1])
        #卸载数据量
        self.D_up=random.randint(D_up_range[0],D_up_range[1])
        #下载数据量
        self.D_down=random.randint(D_down_range[0],D_down_range[1])
        #需要进行计算的数据量
        self.D_ex=self.C*self.D_up

        #最大容忍时延
        self.max_T=random.randint(max_T_range[0],max_T_range[1])
        #执行时延
        self.T_ex=0
        #传输时延
        self.T_tr_up=0
        self.T_tr_down=0
        #队列等待时延
        self.T_queue=0

    def start_offload_v2r(self,rsu,time_frame):
        #卸载的rsu
        self.offload_rsu_v2r = rsu
        #发生卸载的时间
        self.offload_time_v2r = time_frame
        #修改v2r策略
        self.action_v2r=1

    def start_download(self):
        pass

    def start_local_compute(self,time_frame):
        # 发生卸载的时间
        self.offload_time_v2r = time_frame
        # 修改v2r策略
        self.action_v2r = 0

    def start_r2r_offload(self,action_r2r,offload_r2r_rsu, time_frame):
        # 发生卸载的时间
        self.offload_time_r2r = time_frame
        # 修改v2r策略
        self.action_r2r = action_r2r
        offload_r2r_rsu.offload(self)

    def get_offload_rsu_v2r_id(self):
        return self.offload_rsu_v2r.rsu_id

    #生成一个任务
    @staticmethod
    def generate_task(car, current_time):
        tk=Task(car, current_time)
        tk.on_created()
        return tk

    #  R_up是卸载传输速率，bps
    def offload_one_time_slot(self,R_up,time_slot):
        if  self.D_up_cur>=self.D_up:
            return True
        self.T_tr_up+=time_slot
        self.D_up_cur+=R_up*time_slot
        return self.D_up_cur>=self.D_up,self.remaining_time()<0

    #  R_down是下载传输速率，bps
    def download_one_time_slot(self,R_down,time_slot):
        if  self.D_down_cur>=self.D_down:
            return True
        self.T_tr_down+=time_slot
        self.D_down_cur+=R_down*time_slot
        return self.D_down_cur>=self.D_down,self.remaining_time()<0

    # f_ex是智能体的计算能力 cycles/s
    def compute_one_time_slot(self,f_ex,time_slot):
        if  self.D_ex_cur>=self.D_ex:
            return True
        self.T_ex+=time_slot
        self.D_ex_cur+=f_ex*time_slot
        return self.D_ex_cur>=self.D_ex,self.remaining_time()<0

    #整体的时延
    def total_T(self):
        # 本地计算时延
        T_local = self.T_queue + self.T_ex
        # 边缘计算时延
        T_edge = self.T_queue + self.T_tr_up + self.T_ex + self.T_tr_down
        return (1-self.action_v2r)*T_local+self.action_v2r*T_edge

    def remaining_time(self,time_frame=None):
        if time_frame is None:
            return self.max_T-self.total_T()
        return self.max_T-(time_frame-self.arrive_time)

    def f_ex_estimate(self,time_frame):
        #剩余计算周期
        D_ex_rem=self.D_ex-self.D_ex_cur
        #产生这个任务的车辆与其现在相连的RSU之间的瞬时卸载速率
        R_down=self.create_car.get_R_down(time_frame)
        #还剩多少时间用于计算
        T_for_ex=self.remaining_time(time_frame)-self.D_down/R_down
        if T_for_ex>0:
            #不超时的话预估计算资源
            f_ex_estimate=D_ex_rem/T_for_ex
        else:
            #否则这个任务其实也没有必要进行计算了
            f_ex_estimate=-1

        return f_ex_estimate


    def __repr__(self):
        """方便打印查看缓存内容"""
        return str(self.__dict__)


