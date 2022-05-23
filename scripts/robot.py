import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Predictor(nn.Module):
    def __init__(self,
                 dim_observation: int = 3,
                 dim_action: int = 2,
                 dim_hidden: int = 5) -> None:
        super().__init__()

        self.dim_observation = dim_observation
        self.dim_action = dim_action
        self.dim_hidden = dim_hidden

        self.layers = nn.Sequential(
            nn.Linear(self.dim_observation + self.dim_action, self.dim_hidden),
            nn.ReLU(),
            nn.Linear(self.dim_hidden, self.dim_observation)
        )

    def forward(self, position: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat(position, action)
        x = self.layers(x) + x[:self.dim_observation]
        return x


class Robot():
    def __init__(self, predictor: nn.Module) -> None:
        """
        Pose = (x, y, th)
        """
        self.prev_pose = (None, None, None)
        self.pose = (None, None, None)
        self.predictor = predictor

        self._criterion = nn.HuberLoss(reduction='sum', delta=10.0)
        self._optimizer = optim.Adadelta(self.predictor.parameters())

    def set_pose(self, pose: tuple(float)) -> None:
        assert isinstance(pose, (tuple, list, np.ndarray))
        assert len(pose) == 3

        self.prev_pose = self.pose
        self.pose = tuple(map(float, pose))

    def transition_function(self, pose: tuple(float), action: tuple(float)):
        assert isinstance(pose, (tuple, list, np.ndarray))
        assert isinstance(action, (tuple, list, np.ndarray))
        assert len(pose) == 3
        assert len(action) == 2

        return self.predictor(pose, action)

    def update_model(self, prediction: torch.Tensor, true_value: torch.Tensor) -> None:
        self._optimizer.zero_grad()
        loss = self._criterion(prediction, true_value)
        loss.backward()
        self._optimizer.step()
