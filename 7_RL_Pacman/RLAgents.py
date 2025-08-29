from game import Agent
from game import Directions
from pacman import GameState
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple

# 定义常量
BUFFER_SIZE = 10000    # 经验回放缓冲区大小
BATCH_SIZE = 64        # 小批量训练大小
GAMMA = 0.99           # 折扣因子
TAU = 1e-3             # 目标网络软更新参数
LR = 1e-4              # 学习率
UPDATE_EVERY = 10       # 每隔多少步更新网络

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class DQNetwork(nn.Module):
    """Deep Q Network架构"""
    
    def __init__(self, state_size, action_size, seed=0):
        """初始化参数和构建模型"""
        super(DQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # 定义网络层
        self.fc1 = nn.Linear(state_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(128, action_size)
        
    def forward(self, state):
        x = F.relu(self.bn1(self.fc1(state)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        return self.fc4(x)

class ReplayBuffer:
    """固定大小的经验回放缓冲区"""
    
    def __init__(self, buffer_size, batch_size, seed=0):
        """初始化经验回放缓冲区"""
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        
    def add(self, state, action, reward, next_state, done):
        """添加新经验到缓冲区"""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    def sample(self):
        """随机抽取一批经验"""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """返回缓冲区当前大小"""
        return len(self.memory)

class RLAgent(Agent):
    """
    基于DQN的强化学习Pacman智能体
    """
    
    def __init__(self, index=0, training_mode=True, load_model='../models/7_RL_Pacman/pacman_rl_model.pth'):
        """初始化智能体"""
        super(RLAgent, self).__init__()
        
        self.index = index
        self.training_mode = training_mode  # 是否处于训练模式
        
        # 定义行动空间
        self.actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]
        self.action_size = len(self.actions)
        
        # 状态大小（根据特征提取函数确定）
        self.state_features = self._extract_features(None)
        self.state_size = len(self.state_features)
        print(self.state_size)
        
        # 初始化Q网络和目标Q网络
        self.qnetwork_local = DQNetwork(self.state_size, self.action_size).to(device)
        self.qnetwork_target = DQNetwork(self.state_size, self.action_size).to(device)
        
        # 如果提供了模型路径，加载模型
        if load_model and os.path.exists(load_model):
            self.qnetwork_local.load_state_dict(torch.load(load_model))
            self.qnetwork_target.load_state_dict(torch.load(load_model))
            print(f"Loaded model from {load_model}")
        
        # 初始化优化器
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        
        # 初始化经验回放缓冲区
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        
        # 初始化时间步
        self.t_step = 0
        
        # 探索参数（增加agent的探索能力）
        self.epsilon = 1.0 if not training_mode else 1.0  # 初始探索率
        self.epsilon_decay = 0.995                        # 探索率衰减
        self.epsilon_min = 0.05                           # 最小探索率
        
        # 跟踪上一个状态、动作和奖励
        self.last_state = None
        self.last_action = None
        self.last_score = 0
        
    def getAction(self, state):
        """根据当前游戏状态返回行动"""
        # 提取状态特征
        current_state = self._extract_features(state)
        
        # 获取合法动作
        legal = state.getLegalActions(self.index)
        
        # 如果是训练模式的第一次调用，初始化last_state
        if self.training_mode and self.last_state is None:
            self.last_state = current_state
            self.last_score = state.getScore()
        
        # 选择动作
        action_idx = self._select_action(current_state)
        move = self.actions[action_idx]
        
        # 如果选择的动作不合法，随机选择一个合法动作
        if move not in legal:
            if len(legal) > 0:
                move = random.choice(legal)
            else:
                move = Directions.STOP
        
        # 如果在训练模式下并且不是第一步
        if self.training_mode and self.last_action is not None:
            # 计算奖励（当前分数与上一步分数的差值）
            current_score = state.getScore()
            reward = current_score - self.last_score
            
            # 判断游戏是否结束
            done = state.isWin() or state.isLose()
            print(done)
            
            # 将经验添加到回放缓冲区（状态、动作、奖励、下一状态、是否结束）
            self.memory.add(self.last_state, 
                           self.actions.index(self.last_action), 
                           reward, 
                           current_state, 
                           done)
            
            # 定期学习
            self._learn()
            
            # 在游戏结束时进行额外的学习并保存模型参数
            if done:
                # if len(self.memory) > BATCH_SIZE:
                #     for _ in range(10):  # 多学习几次
                #         self._learn()
                self.final(state)
                
                return move
        
        # 更新状态
        self.last_state = current_state
        self.last_action = move
        self.last_score = state.getScore()
        
        return move
    
    def _select_action(self, state):
        """选择动作（epsilon贪婪策略）"""
        # 将状态转换为tensor        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # epsilon策略
        if random.random() > self.epsilon:
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()
            
            # 选择最佳动作
            return torch.argmax(action_values, dim=1).squeeze().cpu().numpy()
            
        else:
            # 随机选择动作
            return random.randrange(self.action_size)
    
    def _learn(self):
        """从经验中学习，更新Q网络"""
        # 增加时间步并检查是否应该学习
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step != 0 or len(self.memory) < BATCH_SIZE:
            return
        
        # 从经验回放缓冲区中采样
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences
        
        # 获取目标值
        self.qnetwork_target.eval()
        with torch.no_grad():
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        self.qnetwork_target.train()
        
        # 计算目标Q值
        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))
        
        # 获取当前Q值估计
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # 使用Huber损失函数
        loss = F.smooth_l1_loss(Q_expected, Q_targets)
        
        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 软更新目标网络
        self._soft_update(self.qnetwork_local, self.qnetwork_target)
        
        # 衰减探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def _soft_update(self, local_model, target_model):
        """软更新目标网络参数"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)
    
    def _extract_features(self, state):
        """从游戏状态提取特征向量"""
        if state is None:
            # 使用初始状态来确定特征维度
            layout_name = 'mediumClassic'  # 可以根据需要修改为其他布局
            from layout import getLayout
            layout = getLayout(layout_name)
            
            init_state = GameState()
            init_state.initialize(layout, 2)  # 假设有2个幽灵
            return self._extract_features(init_state)
        
        # 获取地图尺寸
        walls = state.getWalls()
        width, height = walls.width, walls.height
        
        # 创建一个二维矩阵表示地图状态
        # 0: blank, 1: walls, 2: food, 3: capsules, 4: Pacman, 5: ghosts
        grid_state = np.zeros((width, height))
        
        # 填充墙壁
        for x in range(width):
            for y in range(height):
                if walls[x][y]:
                    grid_state[x][y] = 1
        
        # 填充食物
        food = state.getFood()
        for x in range(width):
            for y in range(height):
                if food[x][y]:
                    grid_state[x][y] = 2
        
        # 填充胶囊
        capsules = state.getCapsules()
        for x, y in capsules:
            grid_state[int(x)][int(y)] = 3
        
        # 填充Pacman
        pacman_x, pacman_y = state.getPacmanPosition()
        grid_state[int(pacman_x)][int(pacman_y)] = 4
        
        # 填充幽灵
        ghost_states = state.getGhostStates()
        for ghost in ghost_states:
            ghost_x, ghost_y = ghost.getPosition()
            grid_state[int(ghost_x)][int(ghost_y)] = 5
        
        # 展平为一维向量
        grid_features = grid_state.flatten()
        
        # 添加额外的非空间特征
        
        # 1. 得分
        score_enc = np.array([state.getScore()])
        
        # 2. 剩余食物数量
        food_count_enc = np.array([state.getNumFood()])
        
        # 3. 剩余胶囊数量
        capsule_count_enc = np.array([len(capsules)])  # 假设最多4个胶囊

        # 将所有特征连接成一个向量
        features = np.concatenate([
            grid_features,          # 地图状态
            score_enc,              # 得分
            food_count_enc,         # 剩余食物数量
            capsule_count_enc,      # 剩余胶囊数量
        ])
        
        return features.astype(np.float32)
    
    def save_model(self, filename):
        """保存模型"""
        torch.save(self.qnetwork_local.state_dict(), filename)
        print(f"Model saved to {filename}")
    
    def final(self, state):
        """游戏结束时调用"""
        if self.training_mode:
            # 保存模型
            self.save_model('../models/7_RL_Pacman/pacman_rl_model.pth')


# 训练模式智能体
class TrainingRLAgent(RLAgent):
    def __init__(self, index=0):
        super(TrainingRLAgent, self).__init__(index, training_mode=True)

# 测试模式智能体（加载训练好的模型）
class TestedRLAgent(RLAgent):  
    def __init__(self, index=0):
        # 加载训练好的模型
        super(TestedRLAgent, self).__init__(index, training_mode=False, load_model='../models/7_RL_Pacman/pacman_rl_model.pth')
