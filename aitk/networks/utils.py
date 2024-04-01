# -*- coding: utf-8 -*-
# ******************************************************
# aitk.networks: Keras model wrapper with visualizations
#
# Copyright (c) 2021 Douglas S. Blank
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.networks
#
# ******************************************************

import base64
import html
import inspect
import io
import math

import numpy as np
from PIL import Image


class Line:
    """
    Properties of a line

    Arguments:
        pointA (array) [x,y]: coordinates
        pointB (array) [x,y]: coordinates

    Returns:
        Line: { length: l, angle: a }
    """

    def __init__(self, pointA, pointB):
        lengthX = pointB[0] - pointA[0]
        lengthY = pointB[1] - pointA[1]
        self.length = math.sqrt(math.pow(lengthX, 2) + math.pow(lengthY, 2))
        self.angle = math.atan2(lengthY, lengthX)


def get_layer_name(layer):
    from tensorflow.python.framework.ops import Tensor
    from tensorflow.keras.models import Model

    if isinstance(layer, Tensor):
        m = Model(inputs=layer, outputs=layer)
        return m.layers[0].name
    else:
        return layer.name


def get_error_colormap():
    return "FIXME"


def minimum(seq):
    """
    Find the minimum value in seq.

    Arguments:
        seq (list) - sequence or matrix of numbers

    Returns:
        The minimum value in list or matrix.

    >>> minimum([5, 2, 3, 1])
    1
    >>> minimum([[5, 2], [3, 1]])
    1
    >>> minimum([[[5], [2]], [[3], [1]]])
    1
    """
    try:
        seq[0][0]
        return min([minimum(v) for v in seq])
    except Exception:
        return np.array(seq).min()


def maximum(seq):
    """
    Find the maximum value in seq.

    Arguments:
        seq (list) - sequence or matrix of numbers

    Returns:
        The maximum value in list or matrix.

    >>> maximum([0.5, 0.2, 0.3, 0.1])
    0.5
    >>> maximum([[0.5, 0.2], [0.3, 0.1]])
    0.5
    >>> maximum([[[0.5], [0.2]], [[0.3], [0.1]]])
    0.5
    """
    try:
        seq[0][0]
        return max([maximum(v) for v in seq])
    except Exception:
        return np.array(seq).max()


def make_input_from_shape(shape):
    from tensorflow.keras.layers import Input

    if isinstance(shape, list):
        input_shape = [bank[1:] for bank in shape]
    else:
        input_shape = shape[1:]
    return Input(input_shape, name="input")


def find_path(from_layer, to_layer_name):
    """
    Breadth-first search to find shortest path
    from from_layer to to_layer_name.

    Returns None if there is no path.
    """
    # No need to put from_layer.name in path:
    from_layer.path = []
    queue = [from_layer]
    while len(queue) > 0:
        current = queue.pop()
        if current.name == to_layer_name:
            return current.path
        else:
            # expand:
            for node in current.outbound_nodes:
                layer = node.outbound_layer
                layer.path = current.path + [layer.name]
                queue.append(layer)
    return None


def gather_nodes(layers):
    nodes = []
    for layer in layers:
        for node in layer.inbound_nodes:
            if node not in nodes:
                nodes.append(node)

        for node in layer.outbound_nodes:
            if node not in nodes:
                nodes.append(node)
    return nodes

#def topological_sort_connections(input_layers, connections):
#    layer_list = input_layers[:]
#    while not done:
#        for connection in connections:

def topological_sort(layers):
    """
    Given a keras model and list of layers, produce a topological
    sorted list, from input(s) to output(s).
    """
    nodes = topological_sort_nodes(layers)
    layer_list = []
    for node in nodes:
        if hasattr(node.inbound_layers, "__iter__"):
            for layer in node.inbound_layers:
                if layer not in layer_list:
                    layer_list.append(layer)
        else:
            if node.inbound_layers not in layer_list:
                layer_list.append(node.inbound_layers)

        if node.outbound_layer not in layer_list:
            layer_list.append(node.outbound_layer)
    return layer_list


def topological_sort_nodes(layers):
    """
    Given a keras model and list of layers, produce a topological
    sorted list, from input(s) to output(s).
    """
    # Initilize all:
    nodes = gather_nodes(layers)
    for node in nodes:
        node.visited = False
    stack = []
    for node in reversed(nodes):
        if not node.visited:
            visit_node(node, stack)
    return reversed(stack)


def visit_node(node, stack):
    """
    Utility function for topological_sort.
    """
    node.visited = True
    if node.outbound_layer:
        for subnode in node.outbound_layer.outbound_nodes:
            if not subnode.visited:
                visit_node(subnode, stack)
    stack.append(node)


def scale_output_for_image(vector, minmax, truncate=False):
    """
    Given an activation name (or something else) and an output
    vector, scale the vector.
    """
    return rescale_numpy_array(vector, minmax, (0, 255), "uint8", truncate=truncate,)


def rescale_numpy_array(a, old_range, new_range, new_dtype, truncate=False):
    """
    Given a numpy array, old min/max, a new min/max and a numpy type,
    create a new numpy array that scales the old values into the new_range.

    >>> import numpy as np
    >>> new_array = rescale_numpy_array(np.array([0.1, 0.2, 0.3]), (0, 1), (0.5, 1.), float)
    >>> ", ".join(["%.2f" % v for v in new_array])
    '0.55, 0.60, 0.65'
    """
    assert isinstance(old_range, (tuple, list)) and isinstance(new_range, (tuple, list))
    old_min, old_max = old_range
    if a.min() < old_min or a.max() > old_max:
        if truncate:
            a = np.clip(a, old_min, old_max)
        else:
            raise Exception("array values are outside range %s" % (old_range,))
    new_min, new_max = new_range
    old_delta = float(old_max - old_min)
    new_delta = float(new_max - new_min)
    if old_delta == 0:
        return ((a - old_min) + (new_min + new_max) / 2).astype(new_dtype)
    else:
        return (new_min + (a - old_min) * new_delta / old_delta).astype(new_dtype)


def svg_to_image(svg, config):
    try:
        import cairosvg
    except ImportError as exc:
        raise Exception("cairosvg is required to convert svg to an image") from exc

    if isinstance(svg, bytes):
        pass
    elif isinstance(svg, str):
        svg = svg.encode("utf-8")
    else:
        raise Exception("svg_to_image takes a str, rather than %s" % type(svg))

    image_bytes = cairosvg.svg2png(bytestring=svg)
    image = Image.open(io.BytesIO(image_bytes))
    if "background_color" in config:
        # create a blank image, with background:
        red = int(config["background_color"][1:3], 16)
        green = int(config["background_color"][3:5], 16)
        blue = int(config["background_color"][5:7], 16)
        background_color = (red, green, blue, 255)
        canvas = Image.new("RGBA", image.size, background_color)
        try:
            canvas.paste(image, mask=image)
        except Exception:
            canvas = None  # fails on images that don't need backgrounds
        if canvas:
            return canvas
        else:
            return image
    else:
        return image


def get_templates(config):
    # Define the SVG strings:
    image_svg = """<rect x="{{rx}}" y="{{ry}}" width="{{rw}}" height="{{rh}}" style="fill:none;stroke:{{border_color}};stroke-width:{{border_width}}"/><image id="{id}_{{name}}" class="{{class_id}}" x="{{x}}" y="{{y}}" height="{{height}}" width="{{width}}" preserveAspectRatio="none" image-rendering="optimizeSpeed" xlink:href="{{image}}"><title>{{tooltip}}</title></image>""".format(
        **config
    )
    line_svg = """<line x1="{{x1}}" y1="{{y1}}" x2="{{x2}}" y2="{{y2}}" stroke="{{arrow_color}}" stroke-width="{arrow_width}"><title>{{tooltip}}</title></line>""".format(
        **config
    )
    arrow_svg = """<line x1="{{x1}}" y1="{{y1}}" x2="{{x2}}" y2="{{y2}}" stroke="{{arrow_color}}" stroke-width="{arrow_width}" marker-end="url(#arrow)"><title>{{tooltip}}</title></line>""".format(
        **config
    )
    curve_svg = """" stroke="{{arrow_color}}" stroke-width="{arrow_width}" marker-end="url(#arrow)" fill="none" />""".format(
        **config
    )
    arrow_rect = """<rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" style="fill:white;stroke:none"><title>{tooltip}</title></rect>"""
    label_svg = """<text x="{x}" y="{y}" font-family="{font_family}" font-size="{font_size}" text-anchor="{text_anchor}" fill="{font_color}" alignment-baseline="central" {transform}>{label}</text>"""
    svg_head = """<svg id='{id}' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' image-rendering="pixelated" width="{top_width}px" height="{top_height}px" style="background-color: {background_color}">
 <g {transform}>
  <svg viewBox="0 0 {viewbox_width} {viewbox_height}" width="{width}px" height="{height}px">
    <defs>
        <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="{arrow_color}" />
        </marker>
    </defs>"""
    templates = {
        "image_svg": image_svg,
        "line_svg": line_svg,
        "arrow_svg": arrow_svg,
        "arrow_rect": arrow_rect,
        "label_svg": label_svg,
        "head_svg": svg_head,
        "curve": curve_svg,
    }
    return templates


def image_to_uri(img_src):
    # Convert to binary data:
    b = io.BytesIO()
    try:
        img_src.save(b, format="gif")
    except Exception:
        return ""
    data = b.getvalue()
    data = base64.b64encode(data)
    if not isinstance(data, str):
        data = data.decode("latin1")
    return "data:image/gif;base64,%s" % html.escape(data)


def controlPoint(config, current, previous_point, next_point, reverse=False):
    """
    # Position of a control point
    # I:  - current (array) [x, y]: current point coordinates
    #     - previous (array) [x, y]: previous point coordinates
    #     - next (array) [x, y]: next point coordinates
    #     - reverse (boolean, optional): sets the direction
    # O:  - (array) [x,y]: a tuple of coordinates
    """
    # When 'current' is the first or last point of the array
    # 'previous' or 'next' don't exist.
    # Replace with 'current'
    p = previous_point or current
    n = next_point or current

    #  // Properties of the opposed-line
    o = Line(p, n)

    # // If is end-control-point, add PI to the angle to go backward
    angle = o.angle + (math.pi if reverse else 0)
    length = o.length * config["smoothing"]

    # // The control point position is relative to the current point
    x = current[0] + math.cos(angle) * length
    y = current[1] + math.sin(angle) * length
    return (x, y)


def bezier(config, points, index):
    """
    Create the bezier curve command

    Arguments:
        points: complete array of points coordinates
        index: index of 'point' in the array 'a'

    Returns:
        String of current path
    """
    current = points[index]
    if index == 0:
        return "M %s,%s " % (current[0], current[1])
    # start control point
    prev1 = points[index - 1] if index >= 1 else None
    prev2 = points[index - 2] if index >= 2 else None
    next1 = points[index + 1] if index < len(points) - 1 else None
    cps = controlPoint(config, prev1, prev2, current, False)
    # end control point
    cpe = controlPoint(config, current, prev1, next1, True)
    return "C %s,%s %s,%s, %s,%s " % (
        cps[0],
        cps[1],
        cpe[0],
        cpe[1],
        current[0],
        current[1],
    )


def svgPath(config, points):
    """
    // Render the svg <path> element
    // I:  - points (array): points coordinates
    //     - command (function)
    //       I:  - point (array) [x,y]: current point coordinates
    //           - i (integer): index of 'point' in the array 'a'
    //           - a (array): complete array of points coordinates
    //       O:  - (string) a svg path command
    // O:  - (string): a Svg <path> element
    """
    # build the d attributes by looping over the points
    return '<path d="' + (
        "".join([bezier(config, points, i) for i in range(len(points))])
    )


def render_curve(start, struct, end_svg, config):
    """
    Collect and render points on the line/curve.
    """
    points = [
        (start["x2"], start["y2"]),
        (start["x1"], start["y1"]),
    ]  # may be direct line
    start["drawn"] = True
    for (template, dict) in struct:
        # anchor! come in pairs
        if (
            (template == "curve")
            and (not dict["drawn"])
            and points[-1] == (dict["x2"], dict["y2"])
        ):
            points.append((dict["x1"], dict["y1"]))
            dict["drawn"] = True
    end_html = end_svg.format(**start)
    if len(points) == 2:  # direct, no anchors, no curve:
        svg_html = (
            """<path d="M {sx} {sy} L {ex} {ey} """.format(
                **{
                    "sx": points[-1][0],
                    "sy": points[-1][1],
                    "ex": points[0][0],
                    "ey": points[0][1],
                }
            )
            + end_html
        )
    else:  # construct curve, at least 4 points
        points = list(reversed(points))
        svg_html = svgPath(config, points) + end_html
    return svg_html


def get_argument_bindings(function, args, kwargs):
    signature = inspect.signature(function)
    binding = signature.bind(*args, **kwargs)

    # Set default values for missing values:
    binding.apply_defaults()
    ignore_param_list = ["self"]
    # Side-effect, remove ignored items:
    [
        binding.arguments.pop(item)
        for item in ignore_param_list
        if item in binding.arguments
    ]
    # Returns OrderedDict:
    return binding.arguments


def is_keras_tensor(item):
    """
    Wrapper around a stupid Keras function that
    crashes when it can't handle something, like an int.
    """
    import tensorflow.keras.backend as K

    try:
        return K.is_keras_tensor(item)
    except Exception:
        return False
