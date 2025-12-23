
class TaskWaitCache:
    def __init__(self, max_capacity):
        self.cache = []  # 存储 Task 对象的列表
        self.max_capacity = max_capacity  # 缓存的最大容量（单位：bits）
        self.current_size = 0  # 当前缓存占用的总容量
    def empty(self):
        return len(self.cache) == 0

    def get_cache(self):
        self.current_size = 0
        new_cache=self.cache
        self.cache=[]
        return new_cache

    def reset(self):
        self.cache=[]
        self.current_size=0

    def sense_state(self):
        # min_task = min(self.cache, key=lambda task: task.remaining_time()) if self.cache else None
        # min_task_remaining_time = min_task.remaining_time() if self.cache else 0
        #
        # sum_remaining_time = sum(map(lambda task: task.remaining_time(), self.cache)) if self.cache else 0
        #
        # # 计算剩余时间平均值
        # avg_remaining_time = (sum_remaining_time / len(self.cache)) if self.cache else 0
        #
        # return [min_task_remaining_time,sum_remaining_time, avg_remaining_time,len(self.cache)]
        min_task = min(self.cache, key=lambda task: task.remaining_time()) if self.cache else None
        min_task_D_ex = min_task.D_ex/1e7 if self.cache else 0
        min_task_D_up = min_task.D_up/1e6 if self.cache else 0
        min_task_D_down = min_task.D_down/1e6 if self.cache else 0
        min_task_remaining_time = min_task.remaining_time() if self.cache else 0

        sum_D_ex = sum(map(lambda task: task.D_ex/1e7, self.cache)) if self.cache else 0
        sum_D_up = sum(map(lambda task: task.D_up/1e6, self.cache)) if self.cache else 0
        sum_D_down = sum(map(lambda task: task.D_down/1e6, self.cache)) if self.cache else 0
        sum_remaining_time = sum(map(lambda task: task.remaining_time(), self.cache)) if self.cache else 0

        # 计算
        avg_D_ex = (sum_D_ex/1e7 / len(self.cache)) if self.cache else 0
        # 计算上传数据平均值
        avg_D_up = (sum_D_up/1e6 / len(self.cache)) if self.cache else 0
        # 计算下载数据平均值
        avg_D_down = (sum_D_down/1e6 / len(self.cache)) if self.cache else 0
        # 计算剩余时间平均值
        avg_remaining_time = (sum_remaining_time / len(self.cache)) if self.cache else 0

        return [min_task_D_ex, min_task_D_up, min_task_D_down, min_task_remaining_time,
               sum_D_ex, sum_D_up, sum_D_down, sum_remaining_time,
               avg_D_ex, avg_D_up, avg_D_down, avg_remaining_time,
               len(self.cache)]

    #总共有多少任务需要计算
    def get_total_D_ex(self):
        sum_D_ex=(sum(map(lambda task: task.D_ex, self.cache)) / len(self.cache)) if self.cache else 0
        return sum_D_ex

    def add_task(self, task):
        """向缓存中添加一个任务。如果缓存满了，移除剩余时间最小的任务直到有空间。"""
        # 任务的上传数据量占用的容量
        task_size = task.D_up

        # 先检查当前容量是否允许添加新任务
        removed_tasks = []  # 存储被移除的任务
        while self.current_size + task_size > self.max_capacity:
            if self.cache:
                # 找到剩余时间最小的任务
                min_remaining_time_task = min(self.cache, key=lambda t: t.remaining_time(t.arrive_time))
                self.cache.remove(min_remaining_time_task)  # 移除任务
                self.current_size -= min_remaining_time_task.D_up  # 更新当前容量
                removed_tasks.append(min_remaining_time_task)  # 记录移除的任务
            else:
                break  # 如果缓存为空，退出循环

        # 添加新任务
        self.cache.append(task)
        self.current_size += task_size  # 更新当前容量

        return removed_tasks  # 返回被移除的任务列表

    def get_task_with_min_remaining_time(self, current_time=None,mode='remove'):
        """获取并移除剩余时间最短的有效任务"""
        if not self.cache:
            return None  # 如果缓存为空，返回 None

        # 找出仍在容忍时间内的任务
        valid_tasks = [task for task in self.cache if task.remaining_time(current_time) > 0]

        # 如果有有效任务，找出剩余时间最短的任务
        if valid_tasks:
            min_task = min(valid_tasks, key=lambda task: task.remaining_time(current_time))
            if mode=='remove':
                self.cache.remove(min_task)  # 从缓存中移除任务
                self.current_size -= min_task.D_up  # 更新当前容量
            return min_task  # 返回被移除的任务

        return None  # 没有有效任务

    def step(self, time_frame, time_slot):
        """增加所有任务的 T_queue，并检测超出最大容忍时间的任务"""
        expired_tasks = []

        for task in self.cache:
            task.T_queue += time_slot  # 增加等待时延
            # 检查任务是否超出最大容忍时间
            if task.remaining_time(time_frame) <= 0:
                expired_tasks.append(task)  # 记录超出时间的任务

        # 移除超出最大容忍时间的任务
        for expired_task in expired_tasks:
            self.cache.remove(expired_task)  # 移除超期任务
            self.current_size -= expired_task.D_up  # 更新当前容量

        return expired_tasks  # 返回超出时间的任务列表

    def __repr__(self):
        """方便打印查看缓存内容"""
        return str([task.__dict__ for task in self.cache])

class TaskComputeCache:
    def __init__(self,f_ex, max_compute_num=4):
        self.f_ex = f_ex    #RSU的最大计算能力
        self.max_compute_num = max_compute_num  # 最多同时计算的任务数量
        self.cache = []  # 存放任务的列表，任务以 task 形式存入
        #需要的计算资源总量
        self.f_ex_total=0
        self.f_ex_estimate=[]

    def reset(self):
        self.cache = []  # 存放任务的列表，任务以 task 形式存入
        # 需要的计算资源总量
        self.f_ex_total = 0
        self.f_ex_estimate = []

    def compute_num(self):
        return len(self.cache)

    def cache_full(self):
        return len(self.cache)>=self.max_compute_num

    def add_task(self, task):
        """添加任务到计算缓存中。"""
        if self.cache_full():
            return
        #存入计算队列中
        self.cache.append(task)

    #计算出能给每个任务分配任务资源
    def get_f_ex_allocate(self,time_frame):
        #预估需要多少资源
        self.f_ex_estimate=[task.f_ex_estimate(time_frame) for task in self.cache]
        #总量
        self.f_ex_total=sum(self.f_ex_estimate)
        #计算出分配给所有任务的计算量
        f_ex_allocate=[f/self.f_ex_total*self.f_ex for f in self.f_ex_estimate]
        return f_ex_allocate

    def sense_state(self):
        # # 每个任务剩余时间
        # task_rem_time = [task.remaining_time() for task in self.cache]
        # task_rem_time += [0] * (self.max_compute_num - len(task_rem_time))
        #
        # return task_rem_time
        # 预估需要多少资源
        f_ex_estimate =[f/1e9 for f in self.f_ex_estimate]
        f_ex_estimate+=[0]*(self.max_compute_num-len(f_ex_estimate))
        # 剩余计算量
        D_ex_rem=[(task.D_ex-task.D_ex_cur)/1e7 for task in self.cache]
        D_ex_rem+=[0]*(self.max_compute_num-len(D_ex_rem))
        # 每个任务剩余时间
        task_rem_time=[task.remaining_time() for task in self.cache]
        task_rem_time += [0] * (self.max_compute_num - len(task_rem_time))

        return f_ex_estimate+D_ex_rem+task_rem_time

    def execute_tasks(self,time_frame,time_slot):
        #计算分配给每个任务分配的计算资源
        f_ex_allocate=self.get_f_ex_allocate(time_frame)

        """执行当前缓存中的任务并更新状态。"""
        completed_tasks = []
        overtime_tasks=[]
        for i, task in enumerate(self.cache):
            compute_ok,overtime=task.compute_one_time_slot(
                f_ex=f_ex_allocate[i],
                time_slot=time_slot
            )
            # 每个任务都执行一个时间片
            if compute_ok:
                completed_tasks.append(task)  # 任务完成，记录到完成列表中
            elif overtime:
                overtime_tasks.append(task) #任务超时，记录任务超时

        # 从缓存中移除完成和超时的任务
        self.cache = [ task for task in self.cache if task not in completed_tasks and task not in overtime_tasks]
        return completed_tasks,overtime_tasks

    def remove_completed_tasks(self):
        """清除完成的任务"""
        self.cache = [(p, t) for p, t in self.cache if not t.is_complete()]

    def __repr__(self):
        """方便打印查看缓存内容"""
        return str([task.__dict__ for task in self.cache])

#
# # 示例使用
# cache = TaskCache(max_capacity=1000 * 10 ** 6)  # 设定缓存最大容量为 1 Gb
#
# # 创建几个任务并添加到缓存中
# task1 = Task(car_id="car1", arrive_time=1)
# task2 = Task(car_id="car2", arrive_time=2)
# task3 = Task(car_id="car3", arrive_time=3)
#
# # 添加任务并检查移除的任务
# removed_tasks1 = cache.add_task(task1)
# removed_tasks2 = cache.add_task(task2)
# removed_tasks3 = cache.add_task(task3)
#
# # 打印当前缓存状态和移除的任务
# print("缓存内容:", cache)
# print("移除的任务:", removed_tasks1 + removed_tasks2 + removed_tasks3)
#
# # 增加所有任务的 T_queue，假设当前时间帧为 10，time_slot 为 5
# expired_tasks = cache.step(time_frame=10, time_slot=5)
# print("超出最大容忍时间的任务:", expired_tasks)
#
# # 获取并移除剩余时间最短的任务
# min_task = cache.get_task_with_min_remaining_time(current_time=10)
# print("移除的剩余时间最短的任务:", min_task)
# print("当前缓存状态:", cache)
