import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Predictor(nn.Module):
    def __init__(self,
                 dim_observation: int = 3,
                 dim_action: int = 2,
                 dim_hidden: int = 10) -> None:
        super().__init__()

        self._dim_observation = dim_observation
        self._dim_action = dim_action
        self._dim_hidden = dim_hidden

        self.layers = nn.Sequential(
            nn.Linear(self._dim_observation + self._dim_action, self._dim_hidden),
            nn.ReLU(),
            nn.Linear(self._dim_hidden, self._dim_observation)
        )

    def forward(self, position: torch.FloatTensor, action: torch.FloatTensor) -> torch.FloatTensor:
        x = torch.cat((position, action), )
        x = self.layers(x) + x[:self._dim_observation]
        return x


class Robot():
    def __init__(self, predictor: nn.Module = Predictor()) -> None:
        """
        Pose = (x, y, th)
        """
        self.prev_pose = torch.tensor([0, 0, 0], dtype=torch.float32)
        self.pose = torch.tensor([0, 0, 0], dtype=torch.float32)
        self.predictor = predictor

        self._criterion = nn.HuberLoss(reduction='sum', delta=10.0)
        self._optimizer = optim.Adadelta(self.predictor.parameters(), lr=1.0)

    def set_pose(self, pose: torch.FloatTensor) -> None:
        assert isinstance(pose, (tuple, list, np.ndarray, torch.FloatTensor))
        assert len(pose) == 3
        
        self.prev_pose = self.pose
        self.pose = pose.type(torch.float32) if isinstance(pose, torch.FloatTensor) else torch.tensor(pose, dtype=torch.float32)

    def transition_function(self, pose: torch.FloatTensor, action: torch.FloatTensor) -> torch.FloatTensor:
        return self.predictor(pose, action)

    def update_model(self, prediction: torch.FloatTensor, true_value: torch.FloatTensor) -> None:
        self._optimizer.zero_grad()
        loss = self._criterion(prediction, true_value)
        loss.backward()
        self._optimizer.step()
