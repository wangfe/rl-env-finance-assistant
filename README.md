# Financial Assistant Agent RL Environment

## 1. Introduction

This project implements a high-performance, extensible **Reinforcement Learning (RL) environment** for a **Financial Assistant Agent** that combines investment trading with personalized financial advice. The environment follows the **OpenAI Gymnasium** standard interface, integrates multi-modal data inputs (technical indicators, macro-economic data, news sentiment), and provides fine-grained reward mechanisms with **Curriculum Learning** support to ensure the agent can effectively manage risk while pursuing returns.

## 2. Project Structure

```
financial_agent_rl/
├── env/
│   ├── __init__.py                # Package initialization
│   ├── financial_env.py           # Core environment class (Gymnasium standard interface)
│   ├── feature_engineering.py     # Feature engineering module (macro + news support)
│   ├── reward.py                  # Reward function module
│   ├── simulator.py               # Trading simulation engine
│   ├── task_generator.py          # Task generator (difficulty-aware generation)
│   ├── scorer.py                  # Scorer: episode-level comprehensive scoring
│   └── curriculum_scheduler.py    # Curriculum learning scheduler (NEW)
├── utils/
│   └── metrics.py                 # Evaluation metrics
├── examples/
│   ├── run_random_agent.py        # Random agent example
│   ├── run_training.py            # PPO training with Stable-Baselines3 example
│   ├── run_task_generator_demo.py # Task generator and scorer demo
│   └── run_enhanced_demo.py       # Enhanced features demo: macro + news + curriculum (NEW)
├── data/                          # Data directory
├── requirements.txt               # Dependencies
└── README.md                      # Project documentation
```

## 3. Core Modules

### 3.1 Base Modules

- **`env/financial_env.py`**: Implements the `gymnasium.Env` interface. Integrates feature engineering, trading simulation, reward calculation, task generation, comprehensive scoring, and curriculum learning.
- **`env/feature_engineering.py`**: Extracts technical indicators (RSI, Bollinger Bands, etc.), macro-economic features, and news sentiment features from raw financial data.
- **`env/reward.py`**: Multi-objective composite reward function that considers returns, volatility, transaction costs, and strategy consistency.
- **`env/simulator.py`**: Simulates financial market trading mechanics, including position management, commissions, and slippage.
- **`env/task_generator.py`**: Generates diverse tasks for the environment, supporting stock analysis, portfolio management, and financial planning.
- **`env/scorer.py`**: Provides comprehensive scoring reports at the end of each episode.

### 3.2 Enhanced Modules (NEW)

- **`env/curriculum_scheduler.py`**: Curriculum learning scheduler that dynamically adjusts task difficulty based on the agent's historical performance, enabling progressive training.

## 4. Enhanced Features

### 4.1 Rich Environment Feedback

The environment now supports the fusion of three data sources:

| Data Type | Features | Description |
| :--- | :--- | :--- |
| Technical Indicators | log_return, volatility_5, volatility_20, sma_ratio, rsi_14, bb_width | Price momentum, moving averages, RSI, Bollinger Bands |
| News Sentiment | sentiment_score, news_volume, sentiment_momentum | News sentiment score, news volume, sentiment momentum |
| Macro-Economic | interest_rate_change, cpi_growth, unemployment_change | Interest rate change, CPI growth rate, unemployment rate change |

**How to enable:**

```python
from env.financial_env import FinancialAssistantEnv
from env.task_generator import (
    TaskGenerator, create_multi_asset_data,
    create_simulated_macro_data, create_simulated_news_data
)

# Create a task generator with macro and news data
data_dict = create_multi_asset_data(num_assets=3, days=500)
macro_data = create_simulated_macro_data(days=500)
news_data = create_simulated_news_data(days=500)

task_gen = TaskGenerator(
    data_dict=data_dict,
    macro_data=macro_data,
    news_data=news_data
)

# Enable macro and news features
env = FinancialAssistantEnv(
    task_generator=task_gen,
    include_macro=True,
    include_news=True
)
# Observation space dimensions: 12 (market features) + 3 (account features) = 15
```

### 4.2 Curriculum Learning

The curriculum learning scheduler automatically adjusts task difficulty based on the agent's performance:

**Difficulty Levels:**

| Difficulty Range | Label | Characteristics |
| :--- | :--- | :--- |
| 0.0 - 0.2 | Beginner | Bull market, low volatility, low transaction costs |
| 0.2 - 0.4 | Easy | Bull/sideways market, moderate volatility |
| 0.4 - 0.6 | Medium | Sideways market, standard parameters |
| 0.6 - 0.8 | Hard | Bear market, high volatility, high transaction costs |
| 0.8 - 1.0 | Expert | Bear market, extreme volatility, complex tasks |

**Usage:**

```python
from env.curriculum_scheduler import CurriculumScheduler

# Create the scheduler
scheduler = CurriculumScheduler(
    initial_difficulty=0.1,       # Start easy
    promotion_threshold=60.0,     # Promote if avg score > 60
    demotion_threshold=20.0,      # Demote if avg score < 20
    difficulty_step_up=0.1,       # Increase by 0.1 on promotion
    difficulty_step_down=0.05,    # Decrease by 0.05 on demotion
    exploration_rate=0.1          # 10% chance to explore new difficulty
)

# Create environment with curriculum learning
env = FinancialAssistantEnv(
    task_generator=task_gen,
    curriculum_scheduler=scheduler,
    include_macro=True,
    include_news=True
)

# Difficulty adjusts automatically during training
for episode in range(100):
    obs, info = env.reset()
    done = False
    while not done:
        action = your_model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    # Curriculum learning updates difficulty automatically at episode end
    if "curriculum_stats" in info:
        stats = info["curriculum_stats"]
        print(f"Current Difficulty: {stats['current_difficulty']:.2f} ({stats['difficulty_label']})")
```

### 4.3 Difficulty-Aware Task Generation

The task generator now supports a `target_difficulty` parameter to automatically adjust task parameters based on difficulty:

```python
# Generate tasks at specific difficulty levels
easy_task = task_gen.generate_task(task_type="stock_analysis", target_difficulty=0.1)
hard_task = task_gen.generate_task(task_type="stock_analysis", target_difficulty=0.8)

# Parameters affected by difficulty: market type preference, time window, transaction costs, initial balance
```

## 5. Installation

```bash
pip install -r requirements.txt
```

## 6. Quick Start

### 6.1 Run the Enhanced Features Demo (Recommended)

```bash
cd financial_agent_rl
python examples/run_enhanced_demo.py
```

This demo includes three parts:
1. **Rich Environment Feedback Demo**: Demonstrates macro-economic and news sentiment feature integration
2. **Curriculum Learning Demo**: Shows dynamic difficulty adjustment over 20 episodes
3. **Difficulty-Aware Task Generation Demo**: Shows task parameter differences across difficulty levels

### 6.2 Run the Task Generator and Scorer Demo

```bash
cd financial_agent_rl
python examples/run_task_generator_demo.py
```

### 6.3 Run the Random Agent

```bash
cd financial_agent_rl
python examples/run_random_agent.py
```

### 6.4 Train a PPO Agent

```bash
cd financial_agent_rl
python examples/run_training.py
```

## 7. Integrating Custom Models for Testing and Training

This environment follows the standard **Gymnasium** interface, so any Gymnasium-compatible RL algorithm or custom model can be easily integrated.

### 7.1 Basic Integration

```python
import sys
sys.path.append("path/to/financial_agent_rl")

from env.financial_env import FinancialAssistantEnv, generate_dummy_data
from env.task_generator import (
    TaskGenerator, create_multi_asset_data,
    create_simulated_macro_data, create_simulated_news_data
)
from env.curriculum_scheduler import CurriculumScheduler

# Option 1: Use fixed data (suitable for quick testing)
df = generate_dummy_data(num_days=500)
env = FinancialAssistantEnv(df=df, initial_balance=10000)

# Option 2: Use task generator + curriculum learning (recommended)
data_dict = create_multi_asset_data(num_assets=5, days=1000)
macro_data = create_simulated_macro_data(days=1000)
news_data = create_simulated_news_data(days=1000)

task_gen = TaskGenerator(
    data_dict=data_dict,
    macro_data=macro_data,
    news_data=news_data,
    risk_profiles=['conservative', 'moderate', 'aggressive']
)
scheduler = CurriculumScheduler(initial_difficulty=0.1)

env = FinancialAssistantEnv(
    task_generator=task_gen,
    curriculum_scheduler=scheduler,
    include_macro=True,
    include_news=True
)
```

### 7.2 Integrating a Custom Model for Testing

```python
class MyCustomModel:
    def predict(self, observation):
        """
        Input: observation (numpy array) - environment observation vector
               With all features enabled, this is a 15-dimensional vector:
               - 12 market features (technical indicators + news sentiment + macro-economic)
               - 3 account features (balance, shares_held, net_worth)

        Output: action (numpy array, shape=(1,)) - range [-1, 1]
        """
        import numpy as np
        rsi = observation[4]
        if rsi < -1.0:
            return np.array([0.5])
        elif rsi > 1.0:
            return np.array([-0.5])
        else:
            return np.array([0.0])

model = MyCustomModel()
obs, info = env.reset()
done = False
while not done:
    action = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

if "score_report" in info:
    report = info["score_report"]
    print(f"Final Score: {report['final_score']}")
```

### 7.3 Integrating Stable-Baselines3

```python
from stable_baselines3 import PPO, SAC

env = FinancialAssistantEnv(task_generator=task_gen, include_macro=True, include_news=True)

# PPO training
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)
model.learn(total_timesteps=50000)
model.save("my_financial_agent")

# SAC training (recommended for continuous action spaces)
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)
```

### 7.4 Integrating an LLM-based Agent

```python
def obs_to_prompt(obs, info):
    """Convert observation vector to a natural language prompt"""
    prompt = f"""You are a financial trading assistant. Please make a trading decision based on the following market information.

Current Market State:
- Log Return: {obs[0]:.4f}
- Short-term Volatility (5-day): {obs[1]:.4f}
- Long-term Volatility (20-day): {obs[2]:.4f}
- Moving Average Ratio (SMA10/SMA30): {obs[3]:.4f}
- RSI (14): {obs[4]:.4f}
- Bollinger Band Width: {obs[5]:.4f}
- Market Sentiment Score: {obs[6]:.4f}
- News Volume: {obs[7]:.4f}
- Sentiment Momentum: {obs[8]:.4f}
- Interest Rate Change: {obs[9]:.4f}
- CPI Growth Rate: {obs[10]:.4f}
- Unemployment Rate Change: {obs[11]:.4f}

Account State:
- Available Balance: {obs[12]:.2f}
- Shares Held: {obs[13]:.4f}
- Net Worth: {obs[14]:.2f}

Please output a number between -1 and 1 as your trading decision."""
    return prompt
```

### 7.5 Environment API Quick Reference

| API | Description | Input/Output |
| :--- | :--- | :--- |
| `env.reset()` | Reset the environment | Returns `(observation, info)` |
| `env.reset(options={"task_type": "..."})` | Reset with a specific task type | Supports `stock_analysis`, `portfolio_management`, `financial_planning` |
| `env.step(action)` | Execute one trading step | Input `action: np.array(shape=(1,), range=[-1,1])` |
| `env.observation_space` | Observation space | `Box(shape=(15,))`: 12 market + 3 account features |
| `env.action_space` | Action space | `Box(shape=(1,), low=-1, high=1)` |
| `env.current_task_meta` | Current task metadata | Contains `task_type`, `market_type`, `difficulty_score`, etc. |
| `info["score_report"]` | Scoring report | Contains `final_score`, `metrics`, etc. |
| `info["curriculum_stats"]` | Curriculum learning statistics | Contains `current_difficulty`, `promotions`, `demotions`, etc. |

## 8. Data Sources

The environment supports three data sources:

```python
# 1. Stock price data (required)
import yfinance as yf
data_dict = {}
for ticker in ["AAPL", "GOOGL", "MSFT"]:
    data_dict[ticker] = yf.download(ticker, start="2020-01-01", end="2024-12-31")

# 2. Macro-economic data (optional)
# Can use real data from FRED API, or use simulated data
from env.task_generator import create_simulated_macro_data
macro_data = create_simulated_macro_data(days=1000)

# 3. News sentiment data (optional)
# Can use real data from Finnhub/NewsAPI, or use simulated data
from env.task_generator import create_simulated_news_data
news_data = create_simulated_news_data(days=1000)
```

## 9. License

This project is licensed under the MIT License.
