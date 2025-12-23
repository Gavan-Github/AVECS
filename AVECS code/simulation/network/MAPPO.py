
class MAPPO:
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
            for i in range(self.num_agents):
                if self.byzantine_agents[i]:
                    if np.random.rand() > 0.1:
                        #把坏智能体认成好智能体，90%的概率调正
                        byzantine_detect[i]=self.byzantine_agents[i]

                else:
                    if np.random.rand() > 0.5:
                        # 把坏智能体认成好智能体，50%的概率调正
                        byzantine_detect[i] = self.byzantine_agents[i]

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
            # valid_indices = [i for i in range(len(scores)) if not byzantine_detect[i]]  # 仅保留合法智能体的索引

            # if valid_scores:  # 防止所有智能体都被识别为拜占庭导致 softmax 计算出错
            #     valid_scores_tensor = torch.tensor(valid_scores, dtype=torch.float32)
            #     valid_weights = torch.softmax(-valid_scores_tensor, dim=0)  # 负号使得低得分获得高权重
            # else:
            #     valid_weights = torch.zeros(len(valid_scores))  # 若所有智能体都是拜占庭，则所有权重设为 0

            # 6. 重新分配权重
            weights = torch.zeros(len(scores), dtype=torch.float32)
            for idx, valid_idx in enumerate(valid_indices):
                weights[valid_idx] = 1  # 仅对非拜占庭智能体赋予 softmax 权重
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

