import numpy as np
import scipy
from typing import Tuple, List

class MPC():
    def __init__(self, agent, dt: float, horizon: int = 5, weights: list = [1, 1, 1, 1]) -> None:
        self.agent = agent
        self.dt = dt
        self.horizon = horizon
        self.w_target = weights[0]
        self.w_route = weights[1]
        self.w_linear_v = weights[2]
        self.w_angular_w = weights[3]

    def _find_nearest(self, position: np.ndarray) -> int:
        return np.sum(np.abs(self.route - position[:2]), axis=1).argmin()

    def _cost_function(self, action_seq: np.ndarray) -> float:
        states = np.zeros((self.horizon, 3), dtype=np.float32)
        states[0] = np.array(self.agent.predictor(np.array(self.agent.pose), self.dt * action_seq[0]))
        nearest = self._find_nearest(states[0])
        cost = (self.w_target * abs(self.route.shape[0] - nearest) + 
                self.w_route * pow(np.sum(np.abs(self.route[nearest] - states[0][:2])), 2) +
                np.exp(self.w_linear_v * abs(action_seq[0][0])) +
                np.exp(self.w_angular_w * abs(action_seq[0][1])))
        
        for i in range(1, self.horizon):
            states[i] = np.array(self.agent.predictor(states[i], self.dt * action_seq[i]))

            nearest = self._find_nearest(states[i])
            cost += (self.w_target * abs(self.route.shape[0] - nearest) + 
                     self.w_route * pow(np.sum(np.abs(self.route[nearest] - states[i][:2])), 2) +
                     np.exp(self.w_linear_v * abs(action_seq[i][0])) +
                     np.exp(self.w_angular_w * abs(action_seq[i][1])))

        return cost

    def calc_next_control(self) -> Tuple[float, float]:
        
        x0 = np.array([0] * 2 * self.horizon)
        bounds = tuple(zip(-0.5, 0.5))
        res = scipy.optimize.minimize(self._cost_function, x0, bounds=bounds)

        return res.x[0], res.x[1]

    def set_route(self, route: List[tuple]) -> None:
        self.route = np.array(route, dtype=np.float32)
