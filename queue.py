class Queue:
    """A base class for maintaining a queue"""

    def __init__(self):
        # The queue
        self.elements = []

        # The parent of each element that has been inserted into the queue
        self.parents = {}

    def __len__(self):
        """Return the length of the queue"""
        return len(self.elements)

    def insert(self, x, parent):
        """Insert an element into the queue."""
        raise NotImplementedError

    def pop(self):
        """Remove and return the first element in the queue"""
        return self.elements.pop(0)

    def get_visited(self):
        """Return the list of elements that have been inserted into the queue"""
        return list(self.parents.keys())

    def get_path(self, xI, xG):
        """Trace back parents to return a path from xI to xG"""
        path = [xG]
        x = xG
        while x != xI:
            x = self.parents.get(x)
            if x is None:
                return []
            path.insert(0, x)
        return path

    def _is_visited(self, x):
        """Return whether x has been visited (i.e., added to the queue at some point)"""
        return x in self.parents

    def _add_element_at(self, ind, x, parent):
        """Add x to the queue at index ind. Also, update the parent of x."""
        self.elements.insert(ind, x)
        self.parents[x] = parent


class QueueBFS(Queue):
    """The queue that implements the insert function for BFS"""

    def insert(self, x, parent):
        # For BFS, do nothing for duplicate x
        if self._is_visited(x):
            return False

        # For BFS, an element is simply added to the end of the queue
        self._add_element_at(len(self.elements), x, parent)
        return True


class QueueDFS(Queue):
    """The queue that implements the insert function for DFS"""

    def insert(self, x, parent):
        # For DFS, do nothing for duplicate x
        if self._is_visited(x):
            return False

        # For DFS, an element is simply added to the beginning of the queue
        self._add_element_at(0, x, parent)
        return True


class QueueAstar(Queue):
    """The queue that implements the insert function for A*"""

    def __init__(self, cost_to_go_estimator):
        # For A*, we need to keep track of cost-to-come and estimated cost-to-go for each element
        super().__init__()
        self.cost_to_go_estimator = cost_to_go_estimator
        self.costs = {}

    def insert(self, x, parent, edge_cost=1):
        # The root has cost-to-come = 0, so its parent should have cost-to-come = -1
        parent_cost_to_come = -edge_cost
        if parent is not None:
            parent_cost_to_come = self.costs[parent][0]

        # Cost is a tuple (cost-to-come, cost-to-go)
        new_cost = (
            parent_cost_to_come + edge_cost,
            self.cost_to_go_estimator.get_lower_bound(x),
        )
        current_cost = self.costs.get(x)

        # Do nothing if the new cost is not smaller than the current cost
        if current_cost is not None and self._get_total_cost(
            current_cost
        ) <= self._get_total_cost(new_cost):
            return False

        # Resolve duplicate element by updating the cost of the element
        # To do this, we simply remove the element from the queue and re-insert it with the new cost
        if current_cost is not None:
            self.elements.remove(x)

        self._add_element_at(
            self._find_insert_index(self._get_total_cost(new_cost)), x, parent
        )
        self.costs[x] = new_cost
        return True

    def _get_total_cost(self, cost):
        """Return the sum of the element in cost"""
        return sum(cost)

    def _find_insert_index(self, cost):
        """Find the first index in the queue such that the cost of the corresponding element is greater than the given cost"""
        ind = 0
        while ind < len(self.elements):
            element_cost = self.costs[self.elements[ind]]
            if self._get_total_cost(element_cost) > cost:
                return ind
            ind += 1
        return ind
