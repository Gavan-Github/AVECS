
class MAPPO:
    def __init__(self, num_agents, obs_dim, n_actions, global_state_dim,
                 lr, clip_param=0.2, gamma=0.9, lam=0.95,
                 global_actor_path=None, global_critic_path=None, agent_type="car",
                 byzantine_attack=False, byzantine_defense=False, byzantine_ratio=0):
        self.device = device
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.clip_param = clip_param
        self.gamma = gamma
        self.lam = lam
        self.agent_type = agent_type
        self.byzantine_attack = byzantine_attack
        self.byzantine_defense = byzantine_defense
        self.byzantine_ratio = byzantine_ratio

        self.discount_factor=wandb.config.discount_factor

        if byzantine_attack:
            self.byzantine_agents = np.random.rand(num_agents) < byzantine_ratio
            # 这里是保证至少有一个byzantine agent
            if np.sum(self.byzantine_agents) < 1:
                self.byzantine_agents[0] = True
            elif np.sum(self.byzantine_agents)>=len(self.byzantine_agents):
                self.byzantine_agents[0]=False
        else:
            self.byzantine_agents = np.random.rand(num_agents) < 0

        if byzantine_defense:
            byzantine_defense_table_columns = ["episode","detect acc"]
            for i in range(len(self.byzantine_agents)):
                byzantine_defense_table_columns.append(f"agent {i} score")
                byzantine_defense_table_columns.append(f"agent {i} byzantine count")
                byzantine_defense_table_columns.append(f"agent {i} pred")
                byzantine_defense_table_columns.append(f"agent {i} label")
            self.byzantine_defense_table = wandb.Table(columns=byzantine_defense_table_columns)

            config=wandb.config
            #完全可以信任的
            self.completely_trust_agent=[]
            self.completely_trust_ratio=config.completely_trust_ratio
            self.partly_trust_count=int(num_agents*config.partly_trust_ratio)
            #找到4个可以绝对信任的车辆
            for i in range(self.num_agents):
                if not self.byzantine_agents[i] and len(self.completely_trust_agent)<self.completely_trust_ratio*num_agents:
                    self.completely_trust_agent.append(i)

            #每个agent byzantine检测次数
            self.byzantine_detect_count=np.zeros(num_agents)

        # 初始化全局 Actor 和 Critic 时将模型迁移到指定设备
        self.global_actor = Actor(obs_dim, n_actions).to(self.device)
        if global_actor_path is not None:
            self.global_actor.load_state_dict(torch.load(global_actor_path, map_location=self.device))
            print(f"Loaded global actor parameters from {global_actor_path}")
        else:
            print("No global actor parameters provided, initializing randomly.")

        self.global_critic = Critic(global_state_dim).to(self.device)
        if global_critic_path is not None:
            self.global_critic.load_state_dict(torch.load(global_critic_path, map_location=self.device))
            print(f"Loaded global critic parameters from {global_critic_path}")
        else:
            print("No global critic parameters provided, initializing randomly.")

        # 为每个智能体构造一个独立的 Actor 网络并迁移到指定设备
        self.actors = [Actor(obs_dim, n_actions).to(self.device) for _ in range(num_agents)]
        for actor in self.actors:
            actor.load_state_dict(self.global_actor.state_dict())
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=lr) for actor in self.actors]
        self.actor_schedulers = [StepLR(actor_optimizer , step_size=100, gamma=0.1) for actor_optimizer in self.actor_optimizers ]
        # 使用全局 Critic 作为 Critic 网络
        self.critic = self.global_critic
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.critic_scheduler = StepLR(self.critic_optimizer , step_size=100, gamma=0.1)

        self.update_time = 0


    def federated_actor_update_by_param_diff_car(self, episode):
        """
        每个智能体计算其 Actor 参数与全局 Actor 参数的差值，
        使用 Krum 方法计算每个智能体的可靠性得分，
        根据得分进行加权聚合，更新全局 Actor，
        并将更新后的全局 Actor 参数广播给各个智能体。
        """
        global_state_dict = self.global_actor.state_dict()

        # 1. 先收集所有非拜占庭车辆的梯度
        deltas = []
        non_byzantine_deltas = []  # 用于存储非拜占庭车辆的梯度
        for i, actor in enumerate(self.actors):
            local_state_dict = actor.state_dict()
            if not self.byzantine_agents[i]:
                # 计算合法智能体的参数差异
                delta = {key: local_state_dict[key] - global_state_dict[key] for key in global_state_dict.keys()}
                non_byzantine_deltas.append(delta)  # 记录合法梯度
                deltas.append(delta)
            else:
                deltas.append(None)  # 先占位


        # 2. 计算拜占庭智能体的攻击梯度，并填充deltas
        zeta = self.num_agents / (self.num_agents - len(non_byzantine_deltas)) if self.byzantine_attack else 0
        byzantine_delta = {
            key: -zeta * sum(d[key] for d in non_byzantine_deltas) / len(non_byzantine_deltas)
            for key in global_state_dict.keys()
        }
        for i, actor in enumerate(self.actors):
            if self.byzantine_agents[i]:
                deltas[i] = byzantine_delta

        if self.byzantine_defense and self.byzantine_attack:
            num_byzantine=int(sum(self.byzantine_agents))
            # 3. 将参数差异转换为向量形式便于计算距离
            def delta_to_vector(delta):
                return torch.cat([tensor.flatten() for tensor in delta.values()])

            vectors = [delta_to_vector(delta) for delta in deltas]

            # 4. 识别拜占庭车辆
            vectors_tensor = torch.stack(vectors)  # 形状为 (N, D)
            distances = torch.cdist(vectors_tensor, vectors_tensor, p=2)  # 计算两两欧几里得距离, 形状 (N, N)

            num_neighbors = self.num_agents - num_byzantine - 2  # Krum 需要最近的 (N - f - 2) 个智能体
            # 计算 Krum 评分
            scores = []
            for i in range(self.num_agents):
                sorted_distances = torch.sort(distances[i])[0][1:num_neighbors + 1]  # 取最近的 num_neighbors 个
                scores.append(sorted_distances.sum().item())  # 计算得分（只加最近的）

            # 使用 KMeans 进行 2 类聚类
            scores_array = np.array(scores).reshape(-1, 1)  # 转换为列向量
            kmeans = KMeans(n_clusters=2, n_init=10, random_state=42).fit(scores_array)
            labels = kmeans.labels_  # 获取每个智能体的类别标签
            centroids = kmeans.cluster_centers_  # 获取两类的中心点
            # 这里修改一下，self.completely_trust_agent存储的是索引，我们看这里的智能体大多属于哪个类别，那另一个类别就是拜占庭类别
            if np.max(self.byzantine_detect_count)>10:
                partly_trust_agent = np.argsort(self.byzantine_detect_count)[self.partly_trust_count]
                trust_labels = [labels[i] for i in self.completely_trust_agent+partly_trust_agent]  # 可信智能体的类别
            else:
                trust_labels = [labels[i] for i in self.completely_trust_agent]  # 可信智能体的类别
            trust_label_majority = max(set(trust_labels), key=trust_labels.count)  # 可信类别的众数
            byzantine_label = 1 - trust_label_majority  # 另一个类别就是拜占庭类别

            # 获取所有拜占庭类别的智能体索引
            byzantine_indices = np.array([i for i in range(self.num_agents) if labels[i] == byzantine_label])
            # 计算每个拜占庭类别智能体到拜占庭聚类中心的距离
            distances_to_byzantine_center = np.array(
                [(scores[i]- centroids[byzantine_label]) for i in byzantine_indices]
            ).flatten()
            # 如果所有距离都为零，则添加一个微小的偏移量来打破平局
            distances_to_byzantine_center += np.random.uniform(1e-8, 1e-6, size=distances_to_byzantine_center.shape)
            # 排序并选择最接近的 num_byzantine 个智能体
            sorted_indices = np.argsort(distances_to_byzantine_center)[:min(num_byzantine,len(sorted_distances))]
            # 生成拜占庭检测结果
            byzantine_detect=[i in byzantine_indices[sorted_indices] for i in range(self.num_agents) ]

            self.byzantine_detect_count[byzantine_detect]+=1
            acc=sum([pre==label for pre,label in zip(byzantine_detect,self.byzantine_agents)])/self.num_agents
            byzantine_defense_table_rows = [episode,acc]
            for i in range(self.num_agents):
                byzantine_defense_table_rows.append(scores[i])
                byzantine_defense_table_rows.append(self.byzantine_detect_count[i])
                byzantine_defense_table_rows.append(int(byzantine_detect[i]))
                byzantine_defense_table_rows.append(int(self.byzantine_agents[i]))

            self.byzantine_defense_table.add_data(*byzantine_defense_table_rows)

            # 5. 仅保留非拜占庭智能体的得分，重新计算 softmax
            valid_scores = [1 for i in range(len(scores)) if not byzantine_detect[i]]

            if valid_scores:  # 防止所有智能体都被识别为拜占庭导致 softmax 计算出错
                valid_scores_tensor = torch.tensor(valid_scores, dtype=torch.float32)
                valid_weights = torch.softmax(-valid_scores_tensor, dim=0)  # 负号使得低得分获得高权重
            else:
                valid_weights = torch.zeros(len(valid_scores))  # 若所有智能体都是拜占庭，则所有权重设为 0

            # 6. 重新分配权重
            weights = torch.zeros(len(scores), dtype=torch.float32)
            for idx, valid_idx in enumerate(valid_indices):
                weights[valid_idx] = valid_weights[valid_idx]  # 仅对非拜占庭智能体赋予 softmax 权重
        else:
            weights = torch.ones(self.num_agents, dtype=torch.float32)

        # 7. 进行加权聚合
        aggregated_delta = {key: torch.zeros_like(global_state_dict[key]) for key in global_state_dict.keys()}
        for i, delta in enumerate(deltas):
            for key in aggregated_delta.keys():
                aggregated_delta[key] += delta[key] * weights[i]

        # 8. 更新全局参数
        new_global_state_dict = {key: global_state_dict[key] - aggregated_delta[key] for key in
                                 global_state_dict.keys()}
        self.global_actor.load_state_dict(new_global_state_dict)

        # 9. 广播新参数到所有智能体
        for i,actor in enumerate(self.actors):
            if not self.byzantine_agents[i]:
                actor.load_state_dict(new_global_state_dict)

