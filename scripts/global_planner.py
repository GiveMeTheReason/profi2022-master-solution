import heapq
import numpy as np


class Global_Planner():
    
    def __init__(self,
                 obstacle_map: np.integer,
                 target_radius: int = 0,
                 neighborhood: int = 8) -> None:
        assert neighborhood in (4, 8), "Neighborhood should be 4 or 8"

        # TODO: Delaunay triangulation: matrix -> graph
        self.map = obstacle_map.astype(bool)
        self.radius = target_radius
        self._shape_x = obstacle_map.shape[0]
        self._shape_y = obstacle_map.shape[1]
        self.neighborhood = neighborhood

    def _distance_metric(self, point_1: tuple[int, int], point_2: tuple[int, int]) -> float:
        return abs(point_1[0] - point_2[0]) + abs(point_1[1] - point_2[1])

    def _cost_function(self, position: tuple[int, int], neighbour: tuple[int, int]) -> int:
        if self._distance_metric(position, neighbour) == 1:
            return 5
        return 7

    def _heuristic_function(self, position: tuple[int, int], target: tuple[int, int]) -> int:
        x = abs(position[0] - target[0]
        y = abs(position[1] - target[1])
        if x + y == 0:
            return 0
        return int(5 * (x + y) - 7 * (x * y) / (x + y) - 1)

    def _get_neighbours(self, position: tuple[int, int]) -> list:
        if self.neighborhood == 8:
            return [(i, j)
                for i in range(position[0] - 1, position[0] + 2)
                for j in range(position[1] - 1, position[1] + 2)
                if (i > -1) and (j > -1) and
                   (i < self._shape_x) and (j < self._shape_y) and
                   (i != position[0] or j != position[1])]
        else:
            return [(i, j)
                for i in range(position[0] - 1, position[0] + 2)
                for j in range(position[1] - 1, position[1] + 2)
                if (i > -1) and (j > -1) and
                   (i < self._shape_x) and (j < self._shape_y) and
                   (abs(position[0] - i) + abs(position[1] - j) == 1)]

    def find_route(self, start: tuple[int, int], target: tuple[int, int]) -> tuple[int, list]:
        """
        A-star algorithm
        """
        assert isinstance(start, (tuple, list, np.ndarray))
        assert isinstance(target, (tuple, list, np.ndarray))
        assert len(start) == len(target) == 2

        start = tuple(map(int, start))
        target = tuple(map(int, target))

        costs: dict[tuple[int, int], int] = {}
        parents: dict[tuple[int, int], tuple[int, int]] = {}
        costs[start] = 0
        parents[start] = start

        queue = []
        heapq.heapify(queue)
        heapq.heappush(queue, (0, start))

        while queue:
            _, pos = heapq.heappop(queue)

            if self._distance_metric(pos, target) <= self.radius:
                route = [pos]
                pos = parents[pos]
                while pos != parents[pos]:
                    route.append(pos)
                    pos = parents[pos]
                route.reverse()
                return costs[pos], route

            for neighbour in self._get_neighbours(pos):
                if self.map[neighbour]:
                    continue
                new_cost = costs[pos] + self._cost_function(pos, neighbour)
                if new_cost < costs.get(neighbour, np.inf):
                    costs[neighbour] = new_cost
                    parents[neighbour] = pos
                    heapq.heappush(queue, (costs[neighbour] + self._heuristic_function(pos, target), neighbour))
        return -1, []
