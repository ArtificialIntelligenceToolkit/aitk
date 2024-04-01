# -*- coding: utf-8 -*-
# *************************************
# aitk.robots: Python robot simulator
#
# Copyright (c) 2020 Calysto Developers
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.robots
#
# *************************************

import json
import os
import threading
import time

from IPython.display import HTML as display_HTML, Image as display_Image, display
from ipywidgets import (
    Box,
    Button,
    FloatSlider,
    FloatText,
    HTML,
    HBox,
    Image,
    Label,
    Layout,
    Output,
    Text,
    Textarea,
    VBox,
)

from .utils import Point, arange, image_to_gif, image_to_png, distance
from .world import World

from aitk.utils import progress_bar

def make_attr_widget(obj, map, title, attrs, labels):
    box = VBox()
    children = []
    if title is not None:
        children.append(Label(value=title))
    for i, attr in enumerate(attrs):
        if hasattr(obj, attr) and isinstance(getattr(obj, attr), dict):
            widget = Textarea(description=labels[i])
        else:
            widget = Text(description=labels[i])
        map[labels[i]] = widget
        children.append(widget)
    box.children = children
    box.layout.border = "10px solid rgb(0 177 255)"
    return box


class Watcher:
    def __init__(self):
        """
        All watchers need a widget.
        """
        self.widget = None

    def draw(self):
        """
        Some watchers need to be told explicitly when to draw.
        """
        raise NotImplementedError("need to implement watcher.draw()")

    def update(self):
        """
        Some watchers need to be told explicitly when to update
        their state.
        """
        raise NotImplementedError("need to implement watcher.update()")

    def reset(self):
        """
        Some watchers hold state and need to be reset.
        """
        raise NotImplementedError("need to implement watcher.reset()")

    def _inject_css(self):
        css = HTML("<style>img.pixelated {image-rendering: pixelated;}</style>")
        display(css)

    def watch(self, **kwargs):
        """
        This method should return the widget associated with
        the watcher.

        Args:
            width (int): optional value to set width of watched item

        Note: you may also set values for other layout attributes,
            depending on details of watched widget.
        """
        self._inject_css()
        widget = self.get_widget(**kwargs)
        display(widget)

    def get_widget(self, **kwargs):
        self._inject_css()
        for key in kwargs:
            setattr(self.widget.layout, key, kwargs[key])
        return self.widget


class RobotWatcher(Watcher):
    def __init__(self, robot, size=100, show_robot=True, attributes="all"):
        super().__init__()
        self.robot = robot
        self.size = size
        self.show_robot = show_robot
        self.map = {}
        self.all_attrs = ["name", "x", "y", "a", "stalled", "tvx", "tva", "state"]
        if attributes == "all":
            self.visible = self.all_attrs[:]
        else:
            self.visible = attributes[:]
        self.labels = [
            "Name:",
            "X:",
            "Y:",
            "A:",
            "Stalled:",
            "Trans vel:",
            "Rotate vel:",
            "State:",
        ]
        widget = make_attr_widget(self.robot, self.map, None, self.all_attrs, self.labels)
        self.image = Image(layout=Layout(
            width="-webkit-fill-available",
            height="auto",
        ))
        self.image.add_class("pixelated")
        if not self.show_robot:
            self.image.layout.display = "none"
        widget.children = [self.image] + list(widget.children)

        self.widget = widget
        self.update()
        self.draw()

    def set_arguments(self, size=None, show_robot=None, attributes=None):
        if size is not None:
            self.size = size
        if show_robot is not None:
            self.show_robot = show_robot

        if show_robot is not None:
            if not self.show_robot:
                self.image.layout.display = "none"
            else:
                self.image.layout.display = "block"

        if attributes is not None:
            if attributes == "all":
                self.visible = self.all_attrs[:]
            else:
                self.visible = attributes[:]

    def draw(self):
        if self.robot.world is None:
            print("This robot is not in a world")
            return

        if self.show_robot:
            image = self.robot.get_image(size=self.size)
            self.image.value = image_to_png(image)
        for i in range(len(self.all_attrs)):
            attr_name = self.all_attrs[i]
            attr = getattr(self.robot, attr_name)
            if isinstance(attr, dict):
                string = json.dumps(attr, sort_keys=True, indent=2)
            else:
                string = str(attr)
            if attr_name in self.visible:
                self.map[self.labels[i]].value = string
                self.map[self.labels[i]].layout.display = "flex"
            else:
                self.map[self.labels[i]].layout.display = "none"

    def update(self):
        pass

    def reset(self):
        pass


class AttributesWatcher(Watcher):
    def __init__(self, obj, *attrs, title=None, labels=None, attributes="all"):
        super().__init__()
        self.obj = obj
        self.map = {}
        self.all_attrs = attrs
        if attributes == "all":
            self.visible = self.all_attrs[:]
        else:
            self.visible = attributes[:]
        self.title = title
        self.labels = labels
        if self.labels is None:
            self.labels = ["%s:" % attr for attr in attrs]
        self.widget = make_attr_widget(self.obj, self.map, title, attrs, labels)
        self.update()
        self.draw()

    def draw(self):
        for i in range(len(self.all_attrs)):
            attr_name = self.all_attrs[i]
            attr = getattr(self.obj, attr_name)
            if isinstance(attr, dict):
                string = json.dumps(attr, sort_keys=True, indent=2)
            else:
                string = str(attr)
            if attr_name in self.visible:
                self.map[self.labels[i]].value = string
                self.map[self.labels[i]].layout.display = "flex"
            else:
                self.map[self.labels[i]].layout.display = "none"

    def set_arguments(self, title=None, attributes=None):
        if title is not None:
            self.title = title
            self.widget.children[0].value = title

        if attributes is not None:
            if attributes == "all":
                self.visible = self.all_attrs[:]
            else:
                self.visible = attributes[:]

    def update(self):
        pass

    def reset(self):
        pass


class CameraWatcher(Watcher):
    def __init__(self, camera, **kwargs):
        super().__init__()
        self.camera = camera
        if "max_width" not in kwargs:
            kwargs["max_width"] = "100%"
        if "height" not in kwargs:
            kwargs["height"] = "auto"
        if "width" not in kwargs:
            kwargs["width"] = "500px"

        self.widget = Image(
            layout=Layout(
                **kwargs
            )
        )
        self.widget.layout.border = "10px solid rgb(0 177 255)"
        self.widget.add_class("pixelated")
        # Update and draw:
        self.draw()

    def draw(self):
        picture = self.camera.get_image()
        self.widget.value = image_to_png(picture)

    def update(self):
        pass

    def reset(self):
        pass


class _Player(threading.Thread):
    """
    Background thread for running a player.
    """

    def __init__(self, controller, time_wait=0.5):
        self.controller = controller
        threading.Thread.__init__(self)
        self.time_wait = time_wait
        self.can_run = threading.Event()
        self.can_run.clear()  # paused
        self.daemon = True  # allows program to exit without waiting for join

    def run(self):
        while True:
            self.can_run.wait()
            self.controller.goto("next")
            time.sleep(self.time_wait)

    def pause(self):
        self.can_run.clear()

    def resume(self):
        self.can_run.set()


class Recorder(Watcher):
    def __init__(self, world, play_rate=0.1):
        super().__init__()
        self.orig_world = world
        # Copy of the world for creating playback:
        self.states = []
        self.last_time = 0
        self.reset()
        # Copy items needed for playback
        for i in range(len(self.world._robots)):
            # Copy list references:
            self.world._robots[i].text_trace = self.orig_world._robots[i].text_trace
            self.world._robots[i].pen_trace = self.orig_world._robots[i].pen_trace
        self.widget = Player("Time:", self.goto, 0, play_rate)

    def draw(self):
        self.widget.update_length(len(self.states))

    def update(self):
        # Record the states from the real world:
        states = []
        for robot in self.orig_world._robots:
            states.append(
                (
                    robot.x,
                    robot.y,
                    robot.a,
                    robot.vx,
                    robot.vy,
                    robot.va,
                    robot.stalled,
                )
            )
        self.states.append(states)

    def reset(self):
        self.world = World(**self.orig_world.to_json())

    def get_widget(self, play_rate=0.0, **kwargs):
        self.widget.player.time_wait = play_rate
        self.set_kwargs(**kwargs)
        return self.widget

    def set_kwargs(self, **kwargs):
        for key in kwargs:
            if key == "width":
                width = kwargs["width"]
                if isinstance(width, int):
                    button_width = width
                else:
                    button_width = int("".join([ch for ch in width if ch.isnumeric()]))

                self.widget.button_layout.width = "%spx" % button_width
                self.widget.slider_layout.width = "%spx" % (button_width - 60)
            else:
                setattr(self.widget.button_layout, key, kwargs[key])
                setattr(self.widget.slider_layout, key, kwargs[key])

    def display(self, play_rate=0.0):
        self.widget.player.time_wait = play_rate
        display(self.widget)

    def get_trace(self, robot_index, current_index, trace_time):
        # return as [Point(x,y), direction]
        max_length = int(trace_time / 0.1)
        start_index = max(current_index - max_length, 0)
        points = [
            (Point(x, y), a)
            for (x, y, a, vx, vy, va, stalled) in [
                state[robot_index]
                for state in self.states[start_index : current_index + 1]
            ]
        ]
        retval = []
        min_dim = min(self.world.width, self.world.height) / 2
        for point,a in points:
            if len(retval) == 0 or distance(retval[-1][0].x, retval[-1][0].y, point.x, point.y) < min_dim:
                retval.append((point, a))
            else:
                retval.extend([None, (point, a)])
        return retval

    def goto(self, time):
        # place robots where they go in copy:
        if len(self.states) == 0:
            # Nothing has happened in world
            for i, orig_robot in enumerate(self.orig_world._robots):
                x, y, a, vx, vy, va, stalled = (
                    orig_robot.x,
                    orig_robot.y,
                    orig_robot.a,
                    orig_robot.vx,
                    orig_robot.vy,
                    orig_robot.va,
                    orig_robot.stalled,
                )
                self.world.robots[i]._set_pose(x, y, a, clear_trace=False)
                self.world.robots[i].vx = vx
                self.world.robots[i].vy = vy
                self.world.robots[i].va = va
                self.world.robots[i].stalled = stalled
                self.world.robots[i].trace[:] = []
            self.last_time = time
        else:
            index = round(time / 0.1)
            index = max(min(len(self.states) - 1, index), 0)
            for i, state in enumerate(self.states[index]):
                x, y, a, vx, vy, va, stalled = state
                self.world.robots[i]._set_pose(x, y, a, clear_trace=False)
                self.world.robots[i].vx = vx
                self.world.robots[i].vy = vy
                self.world.robots[i].va = va
                self.world.robots[i].stalled = stalled
                if self.world.robots[i].do_trace:
                    self.world.robots[i].trace = self.get_trace(
                        i, index, self.world.robots[i].max_trace_length
                    )
            if time >= self.last_time: # forward in time
                events = self.get_events(self.last_time, time)
                for event in events:
                    self.apply_event(event, forward=True)
            else: # backward
                events = reversed(self.get_events(time, self.last_time))
                for event in events:
                    self.apply_event(event, forward=False)
            self.last_time = time
        self.world.time = time
        if self.world.time == 0:
            # In case it changed:
            self.world._reset_ground_image()
        self.world.update()
        picture = self.world.get_image(copy=False)
        return picture

    def get_events(self, start_time, stop_time):
        if (len(self.orig_world._events) > 0 and
            (stop_time == self.orig_world._events[-1][0])):
            # On the very last step, show things that happened
            # after the beginning of step:
            stop_time += 0.1
        return [event for event in self.orig_world._events
                if start_time <= event[0] < stop_time]

    def apply_event(self, event, forward=True):
        # (time, "food-off", robot_index, food_index)
        # (time, "bulb-on", bulb_index)
        # (time, "bulb-off", bulb_index)
        # (time, "bulb-color", bulb_index, color)
        if event[1] == "bulb-off":
            bulb_index = event[2]
            if forward:
                self.world._bulbs[bulb_index].state = "off"
            else:
                self.world._bulbs[bulb_index].state = "on"
        elif event[1] == "bulb-on":
            bulb_index = event[2]
            if forward:
                self.world._bulbs[bulb_index].state = "on"
            else:
                self.world._bulbs[bulb_index].state = "off"
        elif event[1] == "bulb-color":
            bulb_index = event[2]
            previous_color = event[3]
            new_color = event[4]
            if forward:
                self.world._bulbs[bulb_index].color = new_color
            else:
                self.world._bulbs[bulb_index].color = previous_color
        elif event[1] == "food-off":
            robot_index = event[2]
            food_index = event[3]
            self.world._grid.need_update = True
            if forward:
                self.world._robots[robot_index].food_eaten += 1
                self.world._food[food_index].state = "off"
            else:
                self.world._robots[robot_index].food_eaten -= 1
                self.world._food[food_index].state = "on"
        elif event[1] == "beacon-on":
            if forward:
                self.world.beacon.state = "on"
            else:
                self.world.beacon.state = "off"
        elif event[1] == "beacon-off":
            if forward:
                self.world.beacon.state = "off"
            else:
                self.world.beacon.state = "on"

    def save_as(
        self,
        movie_name="aitk_movie",
        start=0,
        stop=None,
        step=0.1,
        loop=0,
        duration=100,
        mp4=True,
    ):
        """
        Save as animated gif and optionally mp4; show with controls.
        loop - 0 means continually
        duration - in MS
        """
        if stop is None:
            stop = len(self.states) * 0.1

        stop = min(stop, len(self.states) * 0.1)

        frames = []
        for time_step in progress_bar(arange(start, stop, step)):
            # Special function to load as gif, leave fp open
            picture = image_to_gif(self.goto(time_step))
            frames.append(picture)

        if frames:
            # First, save animated gif:
            frames[0].save(
                movie_name + ".gif",
                save_all=True,
                append_images=frames[1:],
                loop=loop,
                duration=duration,
            )
            if mp4:
                if os.path.exists(movie_name + ".mp4"):
                    os.remove(movie_name + ".mp4")
                retval = os.system(
                    """ffmpeg -i {0}.gif -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" {0}.mp4""".format(
                        movie_name
                    )
                )
                if retval != 0:
                    print(
                        "created animated gif, but error running ffmpeg; see console log message or use mp4=False in the future"
                    )


class Player(VBox):
    def __init__(self, title, function, length, play_rate=0.1):
        """
        function - takes a slider value and returns displayables
        """
        self.button_layout = Layout(width="600px")
        self.slider_layout = Layout(width="540px")
        self.player = _Player(self, play_rate)
        self.player.start()
        self.title = title
        self.function = function
        self.length = length
        self.output = Output()
        self.position_text = FloatText(value=0.0)
        self.total_text = Label(
            value="of %s" % round(self.length * 0.1, 1))
        controls = self.make_controls()
        super().__init__([controls, self.output])

    def update_length(self, length):
        self.length = length
        self.total_text.value = "of %s" % round(self.length * 0.1, 1)
        self.control_slider.max = round(max(self.length * 0.1, 0), 1)

    def goto(self, position):
        #### Position it:
        if position == "begin":
            self.control_slider.value = 0.0
        elif position == "end":
            self.control_slider.value = round(self.length * 0.1, 1)
        elif position == "prev":
            if self.control_slider.value - 0.1 < 0:
                self.control_slider.value = round(self.length * 0.1, 1)  # wrap around
            else:
                self.control_slider.value = round(
                    max(self.control_slider.value - 0.1, 0), 1
                )
        elif position == "next":
            if round(self.control_slider.value + 0.1, 1) > round(self.length * 0.1, 1):
                self.control_slider.value = 0  # wrap around
            else:
                self.control_slider.value = round(
                    min(self.control_slider.value + 0.1, self.length * 0.1), 1
                )
        self.position_text.value = round(self.control_slider.value, 1)

    def toggle_play(self, button):
        ## toggle
        if self.button_play.description == "Play":
            self.button_play.description = "Stop"
            self.button_play.icon = "pause"
            self.player.resume()
        else:
            self.button_play.description = "Play"
            self.button_play.icon = "play"
            self.player.pause()

    def make_controls(self):
        button_begin = Button(icon="fast-backward")
        button_prev = Button(icon="backward")
        button_next = Button(icon="forward")
        button_end = Button(icon="fast-forward")
        self.button_play = Button(icon="play", description="Play")
        self.control_buttons = HBox(
            [
                button_begin,
                button_prev,
                #self.position_text,
                button_next,
                button_end,
                self.button_play,
            ],
            layout=self.button_layout
        )
        self.control_slider = FloatSlider(
            description=self.title,
            continuous_update=False,
            min=0.0,
            step=0.1,
            max=max(round(self.length * 0.1, 1), 0.0),
            value=0.0,
            readout_format=".1f",
            style={"description_width": "initial"},
            layout=self.slider_layout,
        )
        ## Hook them up:
        button_begin.on_click(lambda button: self.goto("begin"))
        button_end.on_click(lambda button: self.goto("end"))
        button_next.on_click(lambda button: self.goto("next"))
        button_prev.on_click(lambda button: self.goto("prev"))
        self.button_play.on_click(self.toggle_play)
        self.control_slider.observe(self.update_slider_control, names="value")
        controls = VBox(
            [
                HBox(
                    [self.control_slider, self.total_text], layout=Layout(height="40px")
                ),
                self.control_buttons,
            ],
            layout=Layout(width="100%"),
        )
        controls.on_displayed(lambda widget: self.initialize())
        return controls

    def initialize(self):
        """
        Setup the displayer ids to map results to the areas.
        """
        results = self.function(self.control_slider.value)
        if not isinstance(results, (list, tuple)):
            results = [results]
        self.displayers = [display(x, display_id=True) for x in results]

    def update_slider_control(self, change):
        """
        If the slider changes the value, call the function
        and update display areas.
        """
        if change["name"] == "value":
            self.position_text.value = self.control_slider.value
            self.output.clear_output(wait=True)
            results = self.function(self.control_slider.value)
            if not isinstance(results, (list, tuple)):
                results = [results]
            for i in range(len(self.displayers)):
                self.displayers[i].update(results[i])
