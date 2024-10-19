# -*- coding: utf-8 -*-
# ***********************************************************
# aitk.utils: Python utils for AI
#
# Copyright (c) 2021 AITK Developers
#
# https://github.com/ArtificialIntelligenceToolkit/aitk.utils
#
# ***********************************************************

import base64
import html
import io
import math
import os
import sys

try:
    from IPython.display import clear_output, display
except ImportError:
    clear_ouput = None


def array_to_image(array, colormap=None, channels="last", minmax=None):
    """
    Convert an array to a Python Image.

    Args:
        array (sequence): list, or array, of numbers
        colormap (str): optional, the name of the colormap to use
        channels (str): optional, "first" or "last"
        minmax (sequence of 2 numbers): optional, range of numbers
            to scale from, or None to compute dynamically
    """
    try:
        import numpy
        import PIL.Image
    except ImportError as exc:
        raise Exception(
            "The Python libraries PIL and numpy are required for converting an array into an image"
        ) from exc

    array = numpy.array(array)

    if channels == "first" and len(array.shape) == 3:
        array = numpy.moveaxis(array, 0, -1)

    if len(array.shape) == 3:
        if array.shape[-1] == 1:
            array = array.reshape((array.shape[0], array.shape[1]))
        elif array.shape[0] == 1:
            array = array.reshape((array.shape[1], array.shape[2]))
    elif len(array.shape) == 1:
        array = numpy.array([array])

    if minmax is None:
        minmax = array.min(), array.max()

    if colormap is not None:
        try:
            from matplotlib import cm
        except ImportError as exc:
            raise Exception(
                "The Python library matplotlib is required to use a colormap"
            ) from exc

        ## Need to be in range (0,1) for colormapping:
        if minmax[0] != minmax[1]:
            array.clip(*minmax)
        else:
            minmax = [minmax[0] - 1, minmax[1] + 1]
        array = rescale_array(array, minmax, (0, 1), "float")
        try:
            cm_hot = cm.get_cmap(colormap)
            array = cm_hot(array)
        except Exception:
            print("WARNING: invalid colormap; ignored")

        minmax = (0, 1)
        mode = "RGBA"
    else:
        mode = None

    array = rescale_array(array, minmax, (0, 255), "uint8")
    image = PIL.Image.fromarray(array, mode)
    return image


def rescale_array(array, old_range, new_range, dtype):
    """
    Given a numpy array in an old_range, rescale it
    into new_range, and make it an array of dtype.
    """
    try:
        import numpy
    except ImportError as exc:
        raise Exception(
            "The Python library numpy is required for rescaling array"
        ) from exc

    old_min, old_max = old_range
    if array.min() < old_min or array.max() > old_max:
        ## truncate:
        array = numpy.clip(array, old_min, old_max)
    new_min, new_max = new_range
    old_delta = float(old_max - old_min)
    new_delta = float(new_max - new_min)
    if old_delta == 0:
        return ((array - old_min) + (new_min + new_max) / 2).astype(dtype)
    else:
        return (new_min + (array - old_min) * new_delta / old_delta).astype(dtype)


def image_to_data(img_src, format="PNG"):
    # Convert to binary data:
    b = io.BytesIO()
    img_src.save(b, format=format)
    data = b.getvalue()
    data = base64.b64encode(data)
    if not isinstance(data, str):
        data = data.decode("latin1")
    return "data:image/%s;base64,%s" % (format, html.escape(data))


def gallery(
    images,
    labels="{index}",
    border_width=1,
    background_color=(255, 255, 255),
    return_type="display",
    clear=True,
    gallery_shape=None,
):
    """
    Construct a gallery (grid) of images. Can return an HTML table of images
    or a single Image.

    Args:
        images (sequence of Image or an array): sequence of PIL Images or arrays.
            If arrays, they will be processed by array_to_image()
        labels (str or sequence): optional, default "{index}"
        border_width (int): border around images
        background_color (list or tuple of 3 int): optional, represents RGB.
        return_type (str): "display" or None
        clear (bool): if format is None, then clear the display output
        gallery_shape (sequence of 2 int): optional, (cols, rows). One of
            row or col may be None, in order to set one of the dimensions.

    The labels argument can be a string containing the following special patterns
    that will be replaced:

    * "{index}" - the 0-based index of the item
    * "{count}" - index + 1
    * "{row}" - the row of the item
    * "{col}" - the column of the item

    If labels is not None, then gallery() will create a single image. If return_type
    is None then it will return the image rather than display it.

    If clear is True, then the output will be cleared before displaying.
    """
    try:
        import PIL.Image
        from IPython.display import HTML
    except ImportError:
        print("gallery() requires Pillow (Python Image Library) and IPython")
        return

    if len(images) == 0:
        return None

    if (gallery_shape is None) or (
        len(gallery_shape) == 2
        and (gallery_shape[0] is None)
        and (gallery_shape[1] is None)
    ):
        gallery_cols = math.ceil(math.sqrt(len(images)))
        gallery_rows = math.ceil(len(images) / gallery_cols)
    else:
        gallery_cols, gallery_rows = gallery_shape
        if gallery_cols is None:
            gallery_cols = math.ceil(len(images) / gallery_rows)
        elif gallery_rows is None:
            gallery_rows = math.ceil(len(images) / gallery_cols)

    # check that all images are images:
    _images = []
    for index in range(len(images)):
        image = images[index]
        if not isinstance(image, PIL.Image.Image):
            image = array_to_image(image)
        _images.append(image)
    images = _images

    if labels is None:
        size = images[0].size
        size = size[0] + (border_width * 2), size[1] + (border_width * 2)

        gallery_image = PIL.Image.new(
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
        output = gallery_image
    else:
        if isinstance(labels, str):
            label_pattern = labels
            labels = [label_pattern for i in range(len(images))]

        table = "<table>"
        index = 0
        for row in range(gallery_rows):
            table += '<tr style="padding: %dpx">' % border_width
            for col in range(gallery_cols):
                if index < len(labels):
                    label = str(labels[index]).format(
                        **{"count": index + 1, "index": index, "row": row, "col": col}
                    )
                    table += '<td style="text-align: center; padding: %dpx">%s<br/>' % (
                        border_width,
                        label,
                    )
                    table += '<img src="%s" alt="%s" title="%s"></img>' % (
                        image_to_data(images[index]),
                        label,
                        label,
                    )
                    table += "</td>"
                else:
                    table += "<td></td>"
                index += 1
            table += "</tr>"
        table += "</table>"
        output = HTML(table)

    if return_type == "display":
        if clear:
            if clear_output is not None:
                clear_output(wait=True)

        display(output)
    else:
        return output


def progress_bar(range, show_progress=True, progress_type="tqdm"):
    """
    Wrap a range/iter in a progress bar (or not).
    """
    try:
        import tqdm
        import tqdm.notebook
    except ImportError:
        tqdm = None

    if progress_type is None or tqdm is None or show_progress is False:
        return range
    elif progress_type == "tqdm":
        return tqdm.tqdm(range)
    elif (progress_type == "notebook") and (sys.platform != "emscripten"):
        return tqdm.notebook.tqdm(range)
    else:
        return range


def images_to_movie(
    *frames, movie_name="aitk_movie", start=0, stop=None, loop=0, duration=100, mp4=True
):
    """
    Save as animated gif and optionally mp4; show with controls.
    loop - 0 means continually
    duration - in MS
    """
    if stop is None:
        stop = len(frames) * 0.1

    stop = min(stop, len(frames) * 0.1)

    # First, save animated gif:
    frames[0].save(
        movie_name + ".gif",
        save_all=True,
        append_images=frames[1:],
        loop=loop,
        duration=duration,
    )
    if mp4:
        retval = os.system(
            """ffmpeg -y -v quiet -nostats -hide_banner -loglevel error -i {0}.gif -movflags faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" {0}.mp4""".format(
                movie_name
            )
        )
        if retval != 0:
            print(
                "created animated gif, but error running ffmpeg; see console log message or use mp4=False in the future"
            )
