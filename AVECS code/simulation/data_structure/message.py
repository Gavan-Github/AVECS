class MESSAGE:
    TASK_DROPED=0
    TASK_DOWNLOAD=1
    def __init__(self,messagetype,task):
        #任务类型
        self.messagetype =messagetype
        #消息对应的任务
        self.task=task




