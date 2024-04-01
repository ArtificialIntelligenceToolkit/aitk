# -*- coding: utf-8 -*-
# ************************************************************
# aitk.robots: Python robot simulator
#
# Copyright (c) 2021 AITK Developers
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.robots
# ************************************************************

import glob
import io
import json
import math
import os
from collections import OrderedDict
from datetime import datetime, timedelta
from functools import wraps
import random

try:
    import numpy as np
except ImportError:
    np = None

from .config import get_aitk_search_paths
from .hit import Hit

PI_OVER_180 = math.pi / 180
PI_OVER_2 = math.pi / 2
ONE80_OVER_PI = 180 / math.pi
TWO_PI = math.pi * 2

MESSAGES = set()

try:
    _in_colab = 'google.colab' in str(get_ipython())
except Exception:
    _in_colab = False

try:
    from IPython.display import display
except ImportError:
    display = print


def in_colab():
    return _in_colab

class Food():
    def __init__(self, x, y, standard_deviation, state):
        self.x = x
        self.y = y
        self.standard_deviation = standard_deviation
        self.state = state

def print_once(message):
    if message not in MESSAGES:
        print(message)
        MESSAGES.add(message)

def compare(a, b):
    if a > b:
        return 1
    elif a < b:
        return -1
    else:
        return 0

def in_range(delta, current, stop_value):
    if delta < 0:
        return current >= stop_value
    elif delta > 0:
        return current <= stop_value
    else:
        return True

def normal_dist(x , mean , sd):
    prob_density = (math.pi * sd) * math.exp(-0.5 * ((x - mean) / sd) ** 2)
    return prob_density

def round_to(value, base):
    return base * round(value / base)

def uniform_angle(angle):
    return angle % TWO_PI

def degrees_to_world(degrees):
    return ((TWO_PI - (degrees * PI_OVER_180)) % TWO_PI)

def world_to_degrees(direction):
    return (((direction + TWO_PI) * -ONE80_OVER_PI) % 360)

def cast_ray(world, robot, x1, y1, a, maxRange,
             x2=None, y2=None, ignore_robots=None):
    # walls and robots
    hits = []
    if x2 is None:
        x2 = math.sin(a) * maxRange + x1
    if y2 is None:
        y2 = math.cos(a) * maxRange + y1

    for wall in world._walls:
        # never detect hit with yourself
        if robot is not None and wall.robot is robot:
            continue
        # ignore this robot:
        if ((ignore_robots is not None) and
            (wall.robot is not None) and
            (wall.robot in ignore_robots)):
            continue
        for line in wall.lines:
            p1 = line.p1
            p2 = line.p2
            pos = intersect_hit(x1, y1, x2, y2, p1.x, p1.y, p2.x, p2.y)
            if pos is not None:
                dist = distance(pos[0], pos[1], x1, y1)
                height = 1.0 if wall.robot is None else wall.robot.height
                color = wall.robot.color if wall.robot else wall.color
                boundary = len(wall.lines) == 1
                hits.append(
                    Hit(wall,
                        wall.robot,
                        height,
                        pos[0],
                        pos[1],
                        dist,
                        color,
                        x1,
                        y1,
                        boundary,
                        a,
                    )
                )

    hits.sort(
        key=lambda a: a.distance, reverse=True
    )  # further away first, back to front
    return hits

def rotate_around(x1, y1, length, angle):
    """
    Swing a line around a point.
    """
    return [x1 + length * math.cos(-angle),
            y1 - length * math.sin(-angle)]

def dot(v, w):
    x, y, z = v
    X, Y, Z = w
    return x * X + y * Y + z * Z


def length(v):
    x, y, z = v
    return math.sqrt(x * x + y * y + z * z)


def vector(b, e):
    x, y, z = b
    X, Y, Z = e
    return (X - x, Y - y, Z - z)


def unit(v):
    x, y, z = v
    mag = length(v)
    return (x / mag, y / mag, z / mag)


def scale(v, sc):
    x, y, z = v
    return (x * sc, y * sc, z * sc)


def add(v, w):
    x, y, z = v
    X, Y, Z = w
    return (x + X, y + Y, z + Z)


def ccw(ax, ay, bx, by, cx, cy):
    # counter clockwise
    return ((cy - ay) * (bx - ax)) > ((by - ay) * (cx - ax))


def intersect(ax, ay, bx, by, cx, cy, dx, dy):
    # Return true if line segments AB and CD intersect
    return ccw(ax, ay, cx, cy, dx, dy) != ccw(bx, by, cx, cy, dx, dy) and (
        ccw(ax, ay, bx, by, cx, cy) != ccw(ax, ay, bx, by, dx, dy)
    )


def coefs(p1x, p1y, p2x, p2y):
    A = p1y - p2y
    B = p2x - p1x
    C = p1x * p2y - p2x * p1y
    return [A, B, -C]


def intersect_coefs(L1_0, L1_1, L1_2, L2_0, L2_1, L2_2):
    D = L1_0 * L2_1 - L1_1 * L2_0
    if D != 0:
        Dx = L1_2 * L2_1 - L1_1 * L2_2
        Dy = L1_0 * L2_2 - L1_2 * L2_0
        x1 = Dx / D
        y1 = Dy / D
        return [x1, y1]
    else:
        return None


def intersect_hit(p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y):
    """
    Compute the intersection between two lines.
    """
    # http:##stackoverflow.com/questions/20677795/find-the-point-of-intersecting-lines
    L1 = coefs(p1x, p1y, p2x, p2y)
    L2 = coefs(p3x, p3y, p4x, p4y)
    xy = intersect_coefs(L1[0], L1[1], L1[2], L2[0], L2[1], L2[2])
    # now check to see on both segments:
    if xy:
        lowx = min(p1x, p2x) - 0.1
        highx = max(p1x, p2x) + 0.1
        lowy = min(p1y, p2y) - 0.1
        highy = max(p1y, p2y) + 0.1
        if (lowx <= xy[0] and xy[0] <= highx) and (lowy <= xy[1] and xy[1] <= highy):
            lowx = min(p3x, p4x) - 0.1
            highx = max(p3x, p4x) + 0.1
            lowy = min(p3y, p4y) - 0.1
            highy = max(p3y, p4y) + 0.1
            if lowx <= xy[0] and xy[0] <= highx and lowy <= xy[1] and xy[1] <= highy:
                return xy
    return None


def format_time(time):
    hours = time // 3600
    minutes = (time % 3600) // 60
    seconds = (time % 3600) % 60
    return "%02d:%02d:%05.2f" % (hours, minutes, seconds)


def load_world(filename=None):
    """
    worlds/
        test1/
            worlds/test1/w1.json
            worlds/test1/w2.json
        test2/
            worlds/test2/w1.json
        worlds/w1.json

    """
    from .world import World

    if filename is None:
        print("Searching for aitk.robots config files...")
        for path in get_aitk_search_paths():
            print("Directory:", path)
            files = sorted(
                glob.glob(os.path.join(path, "**", "*.json"), recursive=True),
                key=lambda filename: (filename.count("/"), filename),
            )
            if len(files) > 0:
                for fname in files:
                    basename = os.path.splitext(fname)[0]
                    print("    %r" % basename[len(path) :])
            else:
                print("    no files found")
    else:
        if not filename.endswith(".json"):
            filename += ".json"
        for path in get_aitk_search_paths():
            path_filename = os.path.join(path, filename)
            if os.path.exists(path_filename):
                print("Loading %s..." % path_filename)
                with open(path_filename) as fp:
                    contents = fp.read()
                    config = json.loads(contents)
                    config["filename"] = path_filename
                    world = World(**config)
                    return world
        print("No such world found: %r" % filename)
    return None


def load_image(filename, width=None, height=None):
    from PIL import Image

    pathname = find_resource(filename)
    if pathname is not None:
        image = Image.open(pathname)
        if width is not None and height is not None:
            image = image.resize((width, height))
        return image


def find_resource(filename=None):
    if filename is None:
        print("Searching for aitk.robots files...")
        for path in get_aitk_search_paths():
            files = sorted(glob.glob(os.path.join(path, "*.*")))
            print("Directory:", path)
            if len(files) > 0:
                for filename in files:
                    print("    %r" % filename)
            else:
                print("    no files found")
    else:
        for path in get_aitk_search_paths():
            path_filename = os.path.abspath(os.path.join(path, filename))
            if os.path.exists(path_filename):
                return path_filename
        print("No such file found: %r" % filename)
    return None


def image_to_png(image):
    with io.BytesIO() as fp:
        image.save(fp, format="png")
        return fp.getvalue()


def image_to_gif(image):
    # Leave fp opened
    from PIL import Image

    fp = io.BytesIO()
    image.save(fp, "gif")
    frame = Image.open(fp)
    return frame


def gallery(*images, border_width=1, background_color=(255, 255, 255)):
    """
    Construct a gallery of images
    """
    try:
        from PIL import Image
    except ImportError:
        print("gallery() requires Pillow, Python Image Library (PIL)")
        return

    gallery_cols = math.ceil(math.sqrt(len(images)))
    gallery_rows = math.ceil(len(images) / gallery_cols)

    size = images[0].size
    size = size[0] + (border_width * 2), size[1] + (border_width * 2)

    gallery_image = Image.new(
        mode="RGBA",
        size=(int(gallery_cols * size[0]), int(gallery_rows * size[1])),
        color=background_color,
    )

    for i, image in enumerate(images):
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        location = (
            int((i % gallery_cols) * size[0]) + border_width,
            int((i // gallery_cols) * size[1]) + border_width,
        )
        gallery_image.paste(image, location)
    return gallery_image


class arange:
    def __init__(self, start, stop, step):
        self.start = start
        self.stop = stop
        self.step = step

    def __iter__(self):
        current = self.start
        if self.step > 0:
            while current <= self.stop:
                yield current
                current += self.step
        else:
            while current >= self.stop:
                yield current
                current += self.step

    def __len__(self):
        return int(abs(self.stop - self.start) / abs(self.step))


def distance(x1, y1, x2=None, y2=None):
    """
    Takes either (x, y) from two points, or two points
    and returns the distance between them.
    """
    if x2 is None:
        if np is not None:
            # Works with more than two dimensions:
            a1 = np.array(x1)
            a2 = np.array(y1)
            return np.sqrt(np.sum((a1 - a2) ** 2))
        else:
            x1, y1, x2, y2 = x1[0], x1[1], y1[0], y1[1]

    d1 = (x1 - x2)
    d2 = (y1 - y2)
    return math.sqrt(d1 * d1 + d2 * d2)

def distance_point_to_line_3d(point, line_start, line_end):
    """
    Compute distance and location to closest point
    on a line segment in 3D.
    """
    line_vec = vector(line_start, line_end)
    point_vec = vector(line_start, point)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    point_vec_scaled = scale(point_vec, 1.0 / line_len)
    t = dot(line_unitvec, point_vec_scaled)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = scale(line_vec, t)
    # Optipization: assume 2D:
    dist = distance(nearest[0], nearest[1], point_vec[0], point_vec[1])
    nearest = add(nearest, line_start)
    return (dist, nearest)


def distance_point_to_line(point, line_start, line_end):
    return distance_point_to_line_3d(
        (point[0], point[1], 0),
        (line_start[0], line_start[1], 0),
        (line_end[0], line_end[1], 0),
    )


def json_dump(config, fp, sort_keys=True, indent=4):
    dumps(fp, config, sort_keys=sort_keys, indent=indent)


def dumps(fp, obj, level=0, sort_keys=True, indent=4, newline="\n", space=" "):
    if isinstance(obj, dict):
        if sort_keys:
            obj = OrderedDict({key: obj[key] for key in sorted(obj.keys())})
        fp.write(newline + (space * indent * level) + "{" + newline)
        comma = ""
        for key, value in obj.items():
            fp.write(comma)
            comma = "," + newline
            fp.write(space * indent * (level + 1))
            fp.write('"%s":%s' % (key, space))
            dumps(fp, value, level + 1, sort_keys, indent, newline, space)
        fp.write(newline + (space * indent * level) + "}")
    elif isinstance(obj, str):
        fp.write('"%s"' % obj)
    elif isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            fp.write("[]")
        else:
            fp.write(newline + (space * indent * level) + "[")
            # fp.write("[")
            comma = ""
            for item in obj:
                fp.write(comma)
                comma = ", "
                dumps(fp, item, level + 1, sort_keys, indent, newline, space)
            # each on their own line
            if len(obj) > 2:
                fp.write(newline + (space * indent * level))
            fp.write("]")
    elif isinstance(obj, bool):
        fp.write("true" if obj else "false")
    elif isinstance(obj, int):
        fp.write(str(obj))
    elif obj is None:
        fp.write("null")
    elif isinstance(obj, float):
        fp.write("%.7g" % obj)
    else:
        raise TypeError("Unknown object %r for json serialization" % obj)


class throttle(object):
    """
    Decorator that prevents a function from being called more than once every
    time period.
    To create a function that cannot be called more than once a minute:
        @throttle(minutes=1)
        def my_fun():
            pass
    """

    def __init__(self, seconds=0, minutes=0, hours=0):
        self.throttle_period = timedelta(seconds=seconds, minutes=minutes, hours=hours)
        self.time_of_last_call = datetime.min

    def __call__(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            now = datetime.now()
            time_since_last_call = now - self.time_of_last_call

            if time_since_last_call > self.throttle_period:
                self.time_of_last_call = now
                return fn(*args, **kwargs)

        return wrapper


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, item):
        if item == 0:
            return self.x
        elif item == 1:
            return self.y

    def __len__(self):
        return 2

    def __repr__(self):
        return "Point(%s,%s)" % (self.x, self.y)

    def copy(self):
        return Point(self.x, self.y)


class Line:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def __repr__(self):
        return "Line(%s,%s)" % (self.p1, self.p2)

class Grid:
    def __init__(self, width, height, step=10):
        self.width = width
        self.height = height
        self.step = step
        self.blocked = {}
        self.grid = self.spread([])
        self.need_update = True

    def clear_walls(self):
        self.blocked = {}
        self.need_update = True

    def block_area(self, x1, y1, x2, y2, box=True):
        if box:
            x_min = round_to(min(x1, x2), self.step)
            y_min = round_to(min(y1, y2), self.step)
            x_max = round_to(max(x1, x2), self.step)
            y_max = round_to(max(y1, y2), self.step)
            for x in range(x_min, x_max):
                for y in range(y_min, y_max):
                    self.blocked[(x,y)] = 1
        else:
            x_start = round_to(x1, self.step)
            x_stop = round_to(x2, self.step)
            y_start = round_to(y1, self.step)
            y_stop = round_to(y2, self.step)

            dx = compare(x_stop, x_start)
            dy = compare(y_stop, y_start)

            max_range = max(abs(x_stop - x_start),
                            abs(y_stop - y_start))

            if dx != 0 and dy != 0:
                x = x_start
                y = y_start
                while in_range(dx, x, x_stop) and in_range(dy, y, y_stop):
                    self.blocked[(round(x),round(y))] = 1
                    self.blocked[(round(x)-1,round(y))] = 1
                    self.blocked[(round(x),round(y)-1)] = 1
                    self.blocked[(round(x)+1,round(y))] = 1
                    self.blocked[(round(x),round(y)+1)] = 1
                    x += dx / max_range * abs(x_stop - x_start)
                    y += dy / max_range * abs(y_stop - y_start)

    def expand(self, visited, start, x, y, dist):
        retval = []
        for dx in [-self.step, 0, self.step]:
            for dy in [-self.step, 0, self.step]:
                if ((0 <= (x + dx) < self.width) and
                    (0 <= (y + dy) < self.height)):
                    if (x + dx, y + dy) not in self.blocked and (x + dx, y + dy) not in visited:
                        # artifacts on ordering:
                        #retval.append((x + dx, y + dy, dist + distance(0, 0, dx, dy)))
                        retval.append((x + dx, y + dy, dist + self.step))
        return retval

    def bfs(self, start):
        queue = [start + (0,)] # distance
        visited = {}
        while len(queue) > 0:
            x, y, dist = queue.pop(0)
            if (x,y) not in visited:
                queue.extend(self.expand(visited, start, x, y, dist))
                visited[(x,y)] = dist
        return visited

    def spread(self, points):
        # points are Food(x, y, standard_deviation, state)
        # Initialize grid with no values:
        grid = [[0 for y in range(self.height)]
                for x in range(self.width)]

        # For each point, search out from there:
        for food in points:
            if food.state == "off":
                continue
            # make sure we search from a spot on the
            # grid
            px = round_to(food.x, self.step)
            py = round_to(food.y, self.step)
            values = self.bfs((px, py))
            for x in range(0, self.width, self.step):
                for y in range(0, self.height, self.step):
                    dist = values.get((x,y), None)
                    if dist is not None:
                        value = self.weight(dist, food.standard_deviation)
                        for i in range(y, y + self.step):
                            for j in range(x, x + self.step):
                                if 0 <= i < self.height and 0 <= j < self.width:
                                    if (j, i) not in self.blocked:
                                        grid[j][i] += value
                                        grid[j][i] = min(grid[j][i], 1.0)
        return grid

    def weight(self, dist, sd=1):
        """
        Return a value between 1 and 0 where
        dist=0 gives 1, and as dist increases,
        the return value falls off in a normal
        distribution.
        """
        return (normal_dist(dist, 0, sd) / math.pi) / sd

    def update(self, blooms):
        """
        Update the grid with blooms of values
        to spread the values over the grid.
        blooms are a list of Food(x, y, standard_deviation, state).
        """
        if self.need_update:
            self.grid = self.spread(blooms)
            self.need_update = False

    def update_walls(self, walls):
        # update the grid to block smells
        self.clear_walls()
        for wall in walls:
            self.update_wall(wall)

    def update_wall(self, wall):
        self.need_update = True
        if wall.robot is None:
            if len(wall.lines) == 4: # box
                p1 = wall.lines[0].p1
                p3 = wall.lines[1].p2
                self.block_area(p1[0], p1[1], p3[0], p3[1], box=True)
            else: # line
                # FIXME: Includes boundary walls!
                for line in wall.lines:
                    self.block_area(
                        line.p1[0],
                        line.p1[1],
                        line.p2[0],
                        line.p2[1],
                        box=False)

    def get(self, x, y):
        """
        Get the reading at the grid point.
        """
        x = max(min(self.width - 1, int(x)), 0)
        y = max(min(self.height - 1, int(y)), 0)
        return self.grid[x][y]

    def get_image(self):
        """
        Get an image of the grid.
        """
        from PIL import Image
        import numpy as np

        data = np.array(self.grid).transpose() * 255
        data = data.astype(np.uint8)
        image = Image.fromarray(data, "L")
        return image
