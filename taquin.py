import threading
from threading import Barrier
import numpy as np
import heapq
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
import random
import copy
import time


class Message:
    def __init__(
        self,
        sender_id,
        current_position,
        next_position,
        priority,
    ):
        self.sender_id = sender_id
        self.current_position = current_position
        self.next_position = next_position
        self.priority = priority


class Agent(threading.Thread):
    def __init__(self, id, board):
        super().__init__()
        self.barrier: Barrier = board.barrier
        self.id = id
        self.board: Board = board
        self.current_position = self.board.get_agent_position(self.id)
        self.target_position = self.board.get_target_position(self.id)

    def run(self):
        while (not self.board.is_solved()) and (self.board.turn < self.board.max_turn):
            # Round
            with self.board.lock:
                self.board.end_turn = False
            self.board.messages = {
                key: None for key in self.board.agent_positions.keys()
            }
            self.barrier.wait()
            path = self.decide_next_move([])
            if path == None:
                path = self.current_position
            path: list
            self.broadcast(path[min(1, len(path) - 1)], len(path))
            self.barrier.wait()
            while not self.board.end_turn:
                self.handle_conflicts()
            self.barrier.wait()
            # Do the move
            self.board.move_agent(self.id, self.board.messages[self.id].next_position)
            self.current_position = self.board.messages[self.id].next_position
            self.barrier.wait()
            print(
                colored_text(
                    f"END TURN (turn {self.board.turn}): self {self.id}", "blue"
                )
            )

    def handle_conflicts(self):
        for sender, message in self.board.messages.items():
            if (
                (sender != self.id)
                and (
                    self.board.messages[self.id].next_position == message.next_position
                )
                and (self.board.messages[self.id].priority <= message.priority)
            ):
                self.recursive([message.next_position])
            # elif (
            #    (sender != self.id)
            #    and (self.current_position == message.next_position)
            #    and (
            #        message.current_position
            #        == self.board.messages[self.id].next_position
            #    )
            #    and (self.board.messages[self.id].priority == 1) and (message.priority == 1)
            # ):
            #    self.broadcast(self.board.messages[self.id].next_position, random.randint(1, self.board.size*2-1))
            elif (
                (sender != self.id)
                and (self.current_position == message.next_position)
                and (
                    message.current_position
                    == self.board.messages[self.id].next_position
                )
                and (self.board.messages[self.id].priority <= message.priority)
            ):
                print(
                    colored_text(
                        f"CROSSING (turn {self.board.turn}): self {self.id}, sender {message.sender_id}",
                        "red",
                    )
                )
                self.recursive([message.current_position])
                print(f"END CROSSING (turn {self.board.turn}): self {self.id}")
        with self.board.lock:
            if self.no_more_conflict():
                print(
                    colored_text(
                        f"CONFLICTS SOLVED (turn {self.board.turn}): self {self.id}, (current, next) : {[(message.current_position, message.next_position) for message in self.board.messages.values()]}",
                        "magenta",
                    )
                )
                self.board.end_turn = True

    def broadcast(self, position, priority):
        print(
            colored_text(
                f"BROADCAST (turn {self.board.turn}): self {self.id} to {position} priority {priority}",
                "yellow",
            )
        )
        message = Message(self.id, self.current_position, position, priority)
        with self.board.lock:
            self.board.messages[self.id] = message

    def move_to_neighbor(self):
        positions = self.board.available_positions(self.id)
        if positions:
            self.broadcast(random.choice(positions), 1)
            return None
        self.broadcast(self.current_position, self.board.size * 2)

    def recursive(self, obstacles):
        path = self.decide_next_move(obstacles)

        if path == None:
            # Indicate that doesn't want to move !
            if self.board.messages[self.id].priority == 2:
                self.move_to_neighbor()
                return None
            self.broadcast(self.current_position, self.board.size * 2)
            return None

        if len(path) == 1:  # Already at the target
            self.move_to_neighbor()
            return None

        for id, message in self.board.messages.items():
            if (id != self.id) and (path[1] == message.next_position):
                if len(path) <= message.priority:
                    # Loses the conflict !
                    obstacles.append(message.next_position)
                    self.recursive(obstacles)
                else:
                    # Win the conflict !
                    self.broadcast(path[1], len(path))
                return None
        # Conflicts for our agent has been solved
        self.broadcast(path[1], len(path))

        # We end the turn if there is no more conflicts in the board
        with self.board.lock:
            if self.no_more_conflict():
                print(
                    f"CONFLICTS SOLVED in recursive (turn {self.board.turn}): self {self.id}, (current, next) : {[(message.current_position, message.next_position) for message in self.board.messages.values()]}"
                )
                self.board.end_turn = True

    def no_more_conflict(self):
        moves = [message.next_position for message in self.board.messages.values()]
        return (len(set(moves)) == len(moves)) and (
            not self.check_exchanging_positions(self.board.messages)
        )

    def decide_next_move(self, obstacles):
        empty_board = np.zeros((self.board.size, self.board.size), dtype=int)
        for obstacle in obstacles:
            x, y = obstacle
            empty_board[obstacle] = 1
        return dijkstra(
            empty_board,
            self.current_position,
            self.target_position,
        )

    def check_exchanging_positions(self, board_messages):
        for key1, value1 in board_messages.items():
            for key2, value2 in board_messages.items():
                if key1 == key2:
                    continue  # skip comparing the same value to itself
                if (
                    value1.next_position == value2.current_position
                    and value1.current_position == value2.next_position
                ):
                    return True
        return False


class Board:
    def __init__(self, size: int, nb_agents: int, seed: int = 10, max_turn: int = 100):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.agent_positions = {}  # {id: (x, y)}
        self.target_positions = {}  # {id: (x, y)}
        self.agents = []
        self.end_turn = False
        self.move = 0
        self.barrier = Barrier(nb_agents)
        self.lock = threading.Lock()
        self.messages = {}
        self.turn = 0
        self.max_turn = max_turn
        for id, position in generate_config(nb_agents, size, seed):
            x, y = position
            self.board[x, y] = id
            self.agent_positions[id] = position
            self.target_positions[id] = (
                (id - 1) // size,
                (id - 1) % size,
            )
            self.agents.append(Agent(id, self))

        self.board_list = [copy.deepcopy(self.board)]

    def __call__(self) -> None:
        interactive_plotly_grid(self.board_list)

    def move_agent(self, id: int, new_position):
        print(
            colored_text(
                f"MOVE (turn {self.turn}): agent id, to {new_position}", "green"
            )
        )
        self.move += 1
        old_position = self.agent_positions[id]
        if self.board[old_position] == id:
            self.board[old_position] = 0
        self.board[new_position] = id
        self.agent_positions[id] = new_position
        if self.move == len(self.agents):
            self.turn += 1
            self.move = 0
            self.board_list.append(copy.deepcopy(self.board))

    def is_position_empty(self, position):
        return self.board[position] == 0

    def get_agent_position(self, id: int):
        return self.agent_positions.get(id)

    def get_target_position(self, id: int):
        return self.target_positions.get(id)

    def is_solved(self):
        for id, target_position in self.target_positions.items():
            if self.agent_positions[id] != target_position:
                return False
        return True

    def run(self):
        for agent in self.agents:
            agent.start()
        # Wait for agents to finish
        while not self.is_solved() and self.turn < self.max_turn:
            time.sleep(0.1)  # Sleep to prevent excessive CPU usage

        # Wait for agents to finish
        for agent in self.agents:
            agent.join()

    def get_neighbors(self, id):
        neighbors = []
        x, y = self.agent_positions[id]
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                neighbor_id = self.board[nx, ny]
                if neighbor_id != 0:
                    neighbors.append(neighbor_id)
        return neighbors

    def available_positions(self, id):  # new
        neighbors = []
        x, y = self.agent_positions[id]
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                neighbor_id = self.board[nx, ny]
                if neighbor_id == 0:
                    neighbors.append((nx, ny))
        return neighbors


def generate_config(i, n, seed):
    """This function will random sample positions of each agent the first time"""
    random.seed(seed)
    random_numbers = random.sample(range(1, (n * n) + 1), i)
    coordinates = set()

    while len(coordinates) < i:
        x = random.randint(0, n - 1)
        y = random.randint(0, n - 1)
        coordinates.add((x, y))

    result = [(num, coord) for num, coord in zip(random_numbers, coordinates)]

    return result


def dijkstra(grid, start, end):
    if (grid is None) or not start or not end:
        return None

    rows = len(grid)
    cols = len(grid[0])

    def in_bounds(cell):
        r, c = cell
        return 0 <= r < rows and 0 <= c < cols

    def passable(cell):
        r, c = cell
        return grid[r][c] == 0

    def neighbors(cell):
        r, c = cell
        possible_neighbors = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
        return [n for n in possible_neighbors if in_bounds(n) and passable(n)]

    def cost(current, next):
        return 1

    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = dict()
    cost_so_far = dict()
    came_from[start] = None
    cost_so_far[start] = 0

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == end:
            break

        for next in neighbors(current):
            new_cost = cost_so_far[current] + cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost
                heapq.heappush(frontier, (priority, next))
                came_from[next] = current

    if end not in came_from:
        return None

    path = [end]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    path.reverse()

    return path


def interactive_plotly_grid(grid_list):
    index = 0

    fig = go.FigureWidget(make_subplots(rows=1, cols=1, specs=[[{"type": "heatmap"}]]))
    plot_output = widgets.Output()
    index_label = widgets.Label()  # Create a label widget to display the index

    def show_grid():
        with plot_output:
            fig.data = []  # Remove previous heatmap trace
            fig.add_trace(
                go.Heatmap(
                    z=grid_list[index],
                    text=[
                        [str(val) if val != 0 else "" for val in row]
                        for row in grid_list[index]
                    ],
                    hoverinfo="text",
                    texttemplate="%{text}",
                    colorscale=[
                        [0, "grey"],
                        [1 / 3, "red"],
                        [2 / 3, "blue"],
                        [1, "green"],
                    ],
                    showscale=False,
                    xgap=3,
                    ygap=3,
                )
            )
            fig.update_layout(
                xaxis=dict(
                    showgrid=False, zeroline=False, ticks="", showticklabels=False
                ),
                yaxis=dict(
                    showgrid=False, zeroline=False, ticks="", showticklabels=False
                ),
                plot_bgcolor="white",
                margin=dict(t=20, b=20, l=20, r=20),
                width=400,
                height=400,
                autosize=False,
            )

            if index == len(grid_list) - 1:
                index_label.value = f"Last round {index}"
            else:
                index_label.value = f"Round {index}"

    def on_previous_button_clicked(b):
        nonlocal index
        if index > 0:
            index -= 1
            show_grid()

    def on_next_button_clicked(b):
        nonlocal index
        if index < len(grid_list) - 1:
            index += 1
            show_grid()

    def on_last_button_clicked(b):
        nonlocal index
        if index < len(grid_list) - 1:
            index = len(grid_list) - 1
            show_grid()

    previous_button = widgets.Button(description="Previous", button_style="primary")
    next_button = widgets.Button(description="Next", button_style="success")
    last_button = widgets.Button(description="Last", button_style="warning")

    previous_button.on_click(on_previous_button_clicked)
    next_button.on_click(on_next_button_clicked)
    last_button.on_click(on_last_button_clicked)

    buttons = widgets.HBox([previous_button, next_button, last_button])
    display(buttons)
    display(index_label)  # Display the index label in the user interface
    display(fig)
    show_grid()


def colored_text(text, color):
    color_codes = {
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "reset": "\033[0m",
    }

    if color not in color_codes:
        return text

    return f"{color_codes[color]}{text}{color_codes['reset']}"
