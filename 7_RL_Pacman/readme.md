运行`python pacman.py`即可启动游戏，用方向键控制agent移动。
![example_image](./data/example_image.png)

python pacman.py -p TrainingRLAgent -n 20 --frameTime 0.001

待优化：
在实现RLAgent时，需要用到layout，ghost_num等参数，这里直接用了默认值，可以修改pacman代码更好地对齐。


这个项目乍一看有点复杂，实际上确实很复杂。不过我们只需要找到一些关键的代码和函数接口，在此基础上就可以实现我们的强化学习算法。

要基于此项目实现用强化学习控制agent，我们需要：
- 找到键盘传入action控制agent移动的代码，这是我们与环境交互的接口。
- 定义state，包括agent位置，ghosts位置，地图信息，food位置等。
- 执行action之后，要能得到next_state，reward，这些参数用来帮助agent学习。

之后我们就可以训练agent，使其能够根据当前时刻的state，选择最优的action来自动完成游戏，获得最大的score。


分析代码：
`pacman.py main()`:
runGames( **args )-->game = rules.newGame(...pacman, ghosts...)
```python
    def newGame( self, layout, pacmanAgent, ghostAgents, display, quiet = False, catchExceptions=False):
        agents = [pacmanAgent] + ghostAgents[:layout.getNumGhosts()]
        initState = GameState()
        initState.initialize( layout, len(ghostAgents) )
        game = Game(agents, display, self, catchExceptions=catchExceptions)
```
初始化game时传递了agents参数，在`game.py`第686行，通过`action = agent.getAction(observation)`来根据当前state选择action。pacaman和ghost实现了通用的框架，我们的目标是修改pacaman的getAction函数。
查找paman_agent的实现
`runFames(**args)`中传递的参数是`args = readCommand( sys.argv[1:] )`得到的。
```python
def readCommand( argv ):
    ...
    pacmanType = loadAgent(options.pacman, noKeyboard)
    pacman = pacmanType(**agentOpts)
    args['pacman'] = pacman

def loadAgent(pacman, nographics):
    ...
            if pacman in dir(module):
                # 观察传到这里的参数值
                print(f'pacman : {pacman}, module : {dir(module)}')
                return getattr(module, pacman)
```
输出结果：
![loadAgent](./data/loadAgent.png)
在当前目录中搜索以gents.py结尾的Python模块并检查其中是否存在名为pacman的代理类。 
pacman对应的agent类是`KeyboardAgent`, ghost对应的agent类是`RandomGhost`。
我们需要仿照这agent类的结构实现RLAgent，在运行游戏代码时指定`-p`参数即可调用RLAgent。
```python
def readCommand( argv ):
    ...
    parser.add_option('-p', '--pacman', dest='pacman',
                      help=default('the agent TYPE in the pacmanAgents module to use'),
                      metavar='TYPE', default='KeyboardAgent')
```

接下来分析状态，游戏中已经定义了比较完整的GameState（`game.py class GameStateData`），包括agent位置，地图信息，food位置和score等。
也可以在代码中print(state)观察
```txt
%%%%%%%%%%%%%%%%%%%%
%o...%........%....%
%.%%.%.%%%%%%.%.%%.%
%.%........G.....%.%
%.%.%%.%%G %%.%%.%.%
%......%    %......%
%.%.%%.%%%%%%.%%.%.%
%.%..............%.%
%.%%.%.%%%%%%.%.%%.%
%....%...<....%...o%
%%%%%%%%%%%%%%%%%%%%
Score: -7
```
这个state包含了当前游戏环境中全部信息，看到这种格式也很容易让人想到把状态表示成（多通道）二维向量，然后利用卷积神经网络学习。 
不过考虑到游戏逻辑比较简单，状态空间比较小，可以把不同的特征展平为一维向量再拼接
···python
def _extract_features(self, state):
    """从游戏状态提取特征向量"""
    
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
```