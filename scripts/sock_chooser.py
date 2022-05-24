import numpy as np


class Sock_Chooser():
    
    def __init__(self, socks_array: np.integer) -> None:
        assert isinstance(socks_array, np.ndarray)

        # TODO: Hamiltonian path algorithm
        self.socks = socks_array
        self.is_taken = np.zeros(len(socks_array), dtype=bool)
    
    def _distance_metric(self, point_1: np.ndarray, point_2: np.ndarray) -> float:
        return sum(abs(point_1 - point_2))
    
    def _find_nearest(self, position: np.ndarray, taken: bool = False) -> int:
        dist = np.sum(abs(self.socks - position), axis=1)
        if taken:
            return np.argmin(dist)
        return np.argmin(dist[self.is_taken == False])

    def get_untaken(self) -> list[int]:
        return [*np.where(self.is_taken == False)[0]]

    def take_sock(self, position: np.ndarray) -> None:
        self.is_taken[self._find_nearest(position, taken=True)] = True

    def next_sock(self, position: np.ndarray) -> np.ndarray:
        return self.socks[self._find_nearest(position)]
