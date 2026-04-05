# 金融助手 Agent 强化学习环境 (Financial Assistant Agent RL Environment)

## 1. 项目简介

本项目旨在为一款集投资交易与个性化理财建议于一体的**金融助手 Agent** 设计并实现一个高性能、可扩展的强化学习 (RL) 环境。该环境遵循 **OpenAI Gymnasium** 标准接口，整合多模态数据输入，并提供精细化的奖励机制，以确保 Agent 能够在追求收益的同时有效管控风险。

## 2. 项目结构

```
financial_agent_rl/
├── env/
│   ├── financial_env.py          # 核心环境类 (基于 Gymnasium 标准接口)
│   ├── feature_engineering.py    # 特征工程模块
│   ├── reward.py                 # 奖励函数模块
│   ├── simulator.py              # 交易模拟引擎
│   ├── task_generator.py         # 任务生成器：支持多资产、多情景、多任务类型随机任务生成
│   └── scorer.py                 # 打分器：提供 episode 级别的综合评分报告
├── utils/
│   └── metrics.py                # 评估指标
├── examples/
│   ├── run_random_agent.py       # 随机 Agent 示例
│   ├── run_training.py           # 使用 Stable-Baselines3 训练 PPO Agent 示例
│   └── run_task_generator_demo.py # 演示任务生成器与打分器功能
├── data/                         # 存放数据
├── requirements.txt              # 依赖列表
└── README.md                     # 项目说明文档
```

## 3. 核心模块说明

- **`env/financial_env.py`**: 实现了 `gymnasium.Env` 接口。它整合了特征工程、交易模拟、奖励计算、任务生成和综合打分。
- **`env/task_generator.py`**: 负责为环境生成多样化的任务。支持股票分析、投资组合管理（如 401k 资产配置）和财务规划三种任务类型，包含市场情景识别（牛市/熊市/震荡市）和环境参数随机化。
- **`env/scorer.py`**: 在 episode 结束时提供综合评分。评分维度包括累计收益、夏普比率、最大回撤和风险控制能力，并支持根据任务难度和风险偏好进行评分调整。
- **`env/feature_engineering.py`**: 负责从原始金融数据中提取技术指标（RSI、MACD、布林带等）和模拟情感特征。
- **`env/reward.py`**: 定义了多目标复合奖励函数，考虑收益、波动率、交易成本和策略一致性。
- **`env/simulator.py`**: 模拟金融市场交易机制，包括持仓管理、佣金和滑点。

## 4. 安装依赖

```bash
pip install -r requirements.txt
```

## 5. 快速开始

### 5.1 运行任务生成器与打分器演示（推荐）

```bash
python examples/run_task_generator_demo.py
```

### 5.2 运行随机 Agent

```bash
python examples/run_random_agent.py
```

### 5.3 训练 PPO Agent

```bash
python examples/run_training.py
```

## 6. 如何接入自定义模型进行测试与训练

本环境遵循标准的 **Gymnasium** 接口，因此任何兼容 Gymnasium 的 RL 算法或自定义模型都可以轻松接入。以下介绍几种常见的接入方式。

### 6.1 基本接入流程

无论使用哪种模型，接入本环境的基本流程都是一致的：

```python
import sys
sys.path.append("path/to/financial_agent_rl")

from env.financial_env import FinancialAssistantEnv, generate_dummy_data
from env.task_generator import TaskGenerator, create_multi_asset_data

# 方式一：使用固定数据创建环境（适合快速测试）
df = generate_dummy_data(num_days=500)
env = FinancialAssistantEnv(df=df, initial_balance=10000)

# 方式二：使用任务生成器创建环境（推荐，每次 reset 自动生成新任务）
data_dict = create_multi_asset_data(num_assets=5, days=1000)
task_gen = TaskGenerator(
    data_dict=data_dict,
    min_window=100,
    max_window=300,
    risk_profiles=['conservative', 'moderate', 'aggressive']
)
env = FinancialAssistantEnv(task_generator=task_gen)
```

### 6.2 接入自定义模型进行测试

如果您已经有一个训练好的模型（无论是 RL 模型、规则引擎还是 LLM-based Agent），只需实现一个 `predict(observation)` 方法，即可接入环境进行测试：

```python
from env.financial_env import FinancialAssistantEnv, generate_dummy_data

# 创建环境
df = generate_dummy_data(num_days=500)
env = FinancialAssistantEnv(df=df, initial_balance=10000)

# ---- 定义您的自定义模型 ----
class MyCustomModel:
    def __init__(self):
        # 加载您的模型权重、配置等
        pass

    def predict(self, observation):
        """
        输入: observation (numpy array) - 环境的观测向量
              包含 7 个市场特征 + 3 个账户特征 = 10 维向量
              市场特征: [log_return, volatility_5, volatility_20, sma_ratio, rsi_14, bb_width, sentiment_score]
              账户特征: [balance, shares_held, net_worth]

        输出: action (numpy array, shape=(1,)) - 取值范围 [-1, 1]
              正值表示买入（值越大买入比例越高）
              负值表示卖出（绝对值越大卖出比例越高）
              接近 0 表示持有
        """
        import numpy as np
        # 示例：基于 RSI 的简单策略
        rsi = observation[4]  # RSI 指标（已标准化）
        if rsi < -1.0:        # RSI 过低 -> 超卖 -> 买入
            return np.array([0.5])
        elif rsi > 1.0:       # RSI 过高 -> 超买 -> 卖出
            return np.array([-0.5])
        else:
            return np.array([0.0])  # 持有

# ---- 运行测试 ----
model = MyCustomModel()
obs, info = env.reset()

done = False
total_reward = 0
while not done:
    action = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

# 查看测试结果
print(f"总奖励: {total_reward:.2f}")
print(f"最终净值: {info['net_worth']:.2f}")
print(f"交易次数: {info['trades_count']}")

# 如果使用了任务生成器，episode 结束时 info 中会包含评分报告
if "score_report" in info:
    report = info["score_report"]
    print(f"综合评分: {report['final_score']}")
    print(f"累计收益率: {report['metrics']['total_return']}")
    print(f"夏普比率: {report['metrics']['sharpe_ratio']}")
    print(f"最大回撤: {report['metrics']['max_drawdown']}")
```

### 6.3 接入 Stable-Baselines3 模型进行训练

本环境完全兼容 [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)，支持 PPO、A2C、SAC、TD3 等主流算法：

```python
from stable_baselines3 import PPO, A2C, SAC
from env.financial_env import FinancialAssistantEnv, generate_dummy_data

# 创建环境
df = generate_dummy_data(num_days=252 * 3)  # 3 年数据
env = FinancialAssistantEnv(df=df, initial_balance=10000)

# 选择算法并训练（以 PPO 为例）
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)
model.learn(total_timesteps=50000)

# 保存模型
model.save("my_financial_agent")

# 加载并测试
model = PPO.load("my_financial_agent")
obs, info = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

print(f"最终净值: {info['net_worth']:.2f}")
```

**更换算法只需替换一行代码：**

```python
# PPO
model = PPO("MlpPolicy", env, verbose=1)

# A2C
model = A2C("MlpPolicy", env, verbose=1)

# SAC（适合连续动作空间，推荐用于本环境）
model = SAC("MlpPolicy", env, verbose=1)

# TD3
model = TD3("MlpPolicy", env, verbose=1)
```

### 6.4 接入 PyTorch / TensorFlow 自定义神经网络

如果您使用 PyTorch 或 TensorFlow 编写了自定义的策略网络，可以按照以下方式接入：

```python
import torch
import torch.nn as nn
import numpy as np
from env.financial_env import FinancialAssistantEnv, generate_dummy_data

# ---- 定义 PyTorch 策略网络 ----
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim=10, action_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh()  # 输出范围 [-1, 1]，与动作空间匹配
        )

    def forward(self, x):
        return self.net(x)

# ---- 训练循环 ----
df = generate_dummy_data(num_days=500)
env = FinancialAssistantEnv(df=df, initial_balance=10000)

policy = PolicyNetwork(obs_dim=env.observation_space.shape[0], action_dim=1)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

num_episodes = 100
for episode in range(num_episodes):
    obs, info = env.reset()
    done = False
    log_probs = []
    rewards = []

    while not done:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        action = policy(obs_tensor)
        action_np = action.detach().numpy().flatten()

        obs, reward, terminated, truncated, info = env.step(action_np)
        rewards.append(reward)
        done = terminated or truncated

    # 简单的 REINFORCE 更新（示例）
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + 0.99 * G
        returns.insert(0, G)
    returns = torch.FloatTensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    if (episode + 1) % 10 == 0:
        print(f"Episode {episode+1}, Total Reward: {sum(rewards):.2f}, Net Worth: {info['net_worth']:.2f}")

# 保存模型
torch.save(policy.state_dict(), "my_policy.pth")

# 加载模型进行测试
policy.load_state_dict(torch.load("my_policy.pth"))
policy.eval()
```

### 6.5 接入 LLM-based Agent

如果您的模型是基于大语言模型（LLM）的 Agent，可以将环境的观测转换为文本提示，让 LLM 输出交易决策：

```python
import numpy as np
from env.financial_env import FinancialAssistantEnv, generate_dummy_data

# 假设您有一个 LLM 推理函数
def llm_predict(prompt: str) -> float:
    """
    调用 LLM 进行决策
    返回: float, 取值范围 [-1, 1]
    """
    # 这里替换为您的 LLM 调用逻辑
    # 例如: response = openai.ChatCompletion.create(...)
    # 解析 response 得到 action
    return 0.0  # 占位

def obs_to_prompt(obs, info):
    """将观测向量转换为自然语言提示"""
    prompt = f"""你是一个金融交易助手。请根据以下市场信息做出交易决策。

当前市场状态:
- 对数收益率: {obs[0]:.4f}
- 短期波动率(5日): {obs[1]:.4f}
- 长期波动率(20日): {obs[2]:.4f}
- 均线比率(SMA10/SMA30): {obs[3]:.4f}
- RSI(14): {obs[4]:.4f}
- 布林带宽度: {obs[5]:.4f}
- 市场情绪得分: {obs[6]:.4f}

账户状态:
- 可用余额: {obs[7]:.2f}
- 持仓数量: {obs[8]:.4f}
- 账户净值: {obs[9]:.2f}

请输出一个 -1 到 1 之间的数字作为交易决策:
- 正数表示买入（例如 0.5 表示用 50% 的余额买入）
- 负数表示卖出（例如 -0.3 表示卖出 30% 的持仓）
- 0 表示持有

请只输出数字，不要输出其他内容。"""
    return prompt

# ---- 运行测试 ----
df = generate_dummy_data(num_days=200)
env = FinancialAssistantEnv(df=df, initial_balance=10000)
obs, info = env.reset()

done = False
while not done:
    prompt = obs_to_prompt(obs, info)
    action_value = llm_predict(prompt)
    action = np.array([np.clip(action_value, -1, 1)])

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

print(f"最终净值: {info['net_worth']:.2f}")
```

### 6.6 使用不同任务类型进行测试

通过任务生成器，您可以针对不同的金融场景测试模型表现：

```python
from env.financial_env import FinancialAssistantEnv
from env.task_generator import TaskGenerator, create_multi_asset_data

# 创建带任务生成器的环境
data_dict = create_multi_asset_data(num_assets=5, days=1000)
task_gen = TaskGenerator(
    data_dict=data_dict,
    risk_profiles=['conservative', 'moderate', 'aggressive']
)
env = FinancialAssistantEnv(task_generator=task_gen)

# 测试不同任务类型
task_types = ["stock_analysis", "portfolio_management", "financial_planning"]

for task_type in task_types:
    obs, info = env.reset(options={"task_type": task_type})
    meta = env.current_task_meta
    print(f"\n任务类型: {task_type}")
    print(f"任务描述: {meta['description']}")

    done = False
    while not done:
        action = your_model.predict(obs)  # 替换为您的模型
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    # 查看该任务类型下的评分
    report = info["score_report"]
    print(f"综合评分: {report['final_score']}")
    print(f"累计收益率: {report['metrics']['total_return']}")
```

### 6.7 环境接口速查表

| 接口 | 说明 | 输入/输出 |
| :--- | :--- | :--- |
| `env.reset()` | 重置环境，开始新 episode | 返回 `(observation, info)` |
| `env.reset(options={"task_type": "..."})` | 指定任务类型重置 | 支持 `stock_analysis`, `portfolio_management`, `financial_planning` |
| `env.step(action)` | 执行一步交易 | 输入 `action: np.array(shape=(1,), range=[-1,1])`，返回 `(obs, reward, terminated, truncated, info)` |
| `env.observation_space` | 观测空间 | `Box(shape=(10,))`: 7 个市场特征 + 3 个账户特征 |
| `env.action_space` | 动作空间 | `Box(shape=(1,), low=-1, high=1)`: 正=买入, 负=卖出, 0=持有 |
| `env.current_task_meta` | 当前任务元数据 | 包含 `task_type`, `market_type`, `difficulty`, `description` 等 |
| `info["score_report"]` | Episode 结束时的评分报告 | 包含 `final_score`, `metrics`, `difficulty_multiplier` 等 |

## 7. 数据说明

目前环境支持通过 `TaskGenerator` 自动管理多资产数据。您可以通过 `create_multi_asset_data` 生成模拟数据，或接入真实数据源（如 `yfinance`）：

```python
import yfinance as yf

# 下载真实股票数据
tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
data_dict = {}
for ticker in tickers:
    data_dict[ticker] = yf.download(ticker, start="2020-01-01", end="2024-12-31")

# 使用真实数据创建任务生成器
task_gen = TaskGenerator(data_dict=data_dict)
env = FinancialAssistantEnv(task_generator=task_gen)
```

## 8. 许可证

本项目采用 MIT 许可证。
