from math import floor

from .colors import Color


class Grid(object):
    """
    Create a discrete grid. Useful for measuring coverage.
    """
    def __init__(self, grid_size, world):
        """
        This class creates a grid of locations on top of a simulated world
        to monitor how much of the world has been visited. Each grid
        location is initally set to 0 to indicate that it is unvisited,
        and is updated by 1 every time that it is visited.

        The grid is stored in (x, y) or (col, row) order with origin at
        top left corner.

        The sequence traveled is the order of (x, y) of the first time
        visits to each grid location.

        Args:
            grid_size ((int, int)): the (width, height) of the grid
            world (World, or (int, int)): the world, or (width, height) of the world

        Examples:

            # Greate a 10x10 grid with world size (100, 200)
            >>> grid = Grid((10, 10), (100, 200))

            # Greate a 20x30 grid with world
            >>> grid = Grid((20, 30), world)
        """
        self.border_line_color = Color("white", 128)
        self.fill_color = Color("white", 64)
        self.grid_size = grid_size
        if isinstance(world, (list, tuple)):
            self.world = None
            self.world_size = world
        else:
            self.world = world
            self.world_size = (world.width, world.height)
        self.reset()

    def reset(self):
        self.sequence = []
        self.grid = []
        for i in range(self.grid_size[0]):
            self.grid.append([0] * self.grid_size[1])

    def get_sequence(self, pad_value=None, total_length=None):
        """
        Return the sequence traveled.
        """
        if pad_value is not None and total_length is not None:
            pad_count = total_length - len(self.sequence)
            return self.sequence + [pad_value] * pad_count
        else:
            return self.sequence

    def get_xy(self, x, y):
        """
        Given an (x, y) point in the world, return the
        (x, y) grid location.
        """
        x_size = self.world_size[0] / self.grid_size[0]
        y_size = self.world_size[1] / self.grid_size[1]
        col = floor(x / x_size)
        row = floor(y / y_size)
        return (col, row)

    def update(self, x, y):
        """
        In the simulator, the origin is at the top-left corner.
        Update the appropriate grid location.
        """
        row, col = self.get_xy(x, y)
        if self.grid[row][col] == 0:
            self.sequence.append((col, row))
        self.grid[row][col] += 1

    def show(self):
        """
        Print a representation of the grid.
        """
        for y in range(self.grid_size[1]):
            for x in range(self.grid_size[0]):
                print("%3d" % self.grid[x][y], end=" ")
            print()
        print()

    def draw(self):
        """
        Impose a grid on the world to show visited cells.
        Requires an aitk.robots.World be given to constructor.
        """
        self.world.canvas.clear()
        x_size = self.world.width // self.grid_size[0]
        y_size = self.world.height // self.grid_size[1]

        self.world.canvas.set_fill_style(self.fill_color)
        self.world.canvas.set_stroke_style(self.border_line_color)

        # Draw the vertical lines:
        for x in range(self.world.width // x_size):
            self.world.canvas.draw_line(x * x_size, 0, x * x_size, self.world.height)
        # Draw the horizontal lines:
        for y in range(self.world.height // y_size):
            self.world.canvas.draw_line(0, y * y_size, self.world.width, y * y_size)

        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                if self.grid[x][y] > 0:
                    self.world.canvas.draw_rect(
                        x * x_size, y * y_size, x_size, y_size)

    def analyze_visits(self):
        """Calculate the percentage of visited cells in the grid."""
        cells_visited = 0
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                if self.grid[x][y] > 0:
                    cells_visited += 1
        percent_visited = cells_visited / (self.grid_size[0] * self.grid_size[1])
        return percent_visited
