import numpy as np
from bayes_opt import BayesianOptimization
from run_this import run_this

def evaluate_dqn(lr, capacity):
    return run_this(lr, int(capacity))

def bayesian_optimization(n_iter=25):
    # 定义参数范围
    pbounds = {
        'lr': (0.0001, 0.1),
        'capacity': (50, 2000)
    }

    # 初始化贝叶斯优化器
    optimizer = BayesianOptimization(
        f=evaluate_dqn,
        pbounds=pbounds,
        verbose=2,
        random_state=42
    )

    # 开始优化
    optimizer.maximize(
        init_points=5,  # 初始点数量
        n_iter=n_iter  # 迭代次数
    )

    # 获取最佳参数
    best_params = optimizer.max['params']
    best_params['capacity'] = int(best_params['capacity'])
    best_score = optimizer.max['target']

    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score}")

    return best_params, best_score

if __name__ == "__main__":
    best_params, best_score = bayesian_optimization()
