# coding=utf-8
# Author: Andreas Christian Mueller <t3kcit@gmail.com>
#
# (c) 2012
# Modified by: Paul Nechifor <paul@nechifor.net>
#
# License: MIT

from __future__ import division

import base64
import colorsys
import io
import os
import re
import sys
import warnings
from operator import itemgetter
from random import Random
from xml.sax import saxutils

import matplotlib
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFilter
from PIL import ImageFont

from .query_integral_image import query_integral_image
from .tokenization import unigrams_and_bigrams, process_tokens

FILE = os.path.dirname(__file__)
FONT_PATH = os.environ.get('FONT_PATH', os.path.join(FILE, 'DroidSansMono.ttf'))
STOPWORDS = set(map(str.strip, open(os.path.join(FILE, 'stopwords')).readlines()))


class IntegralOccupancyMap(object):
    """
        积分图: 是一个用于记录单词位置的数据结构。

        https://python123.io/tutorials/word_cloud
    """

    def __init__(self, height, width, mask):
        self.height = height
        self.width = width
        # 这段代码是用于计算二值掩模图像（mask）的积分图像（integral），
        # 其中积分图像的每个像素值表示对应像素及其左上角区域内的所有像素值之和。
        if mask is not None:
            # the order of the cumsum's is important for speed ?!
            # 通过 np.cumsum 方法计算出掩模图像的积分图像。np.cumsum 函数是用于计算累积和的函数，它可以在给定轴向上计算每个元素及其之前所有元素的和。
            # 这里使用了两个 np.cumsum 函数，第一个函数在水平方向上计算积分值，第二个函数在垂直方向上计算积分值。
            self.integral = np.cumsum(np.cumsum(255 * mask, axis=1),
                                      axis=0).astype(np.uint32)
        else:
            # 则积分图像被初始化为全零矩阵，其大小为 (height, width)。这种情况通常用于计算没有掩模的原始图像的积分图像。
            self.integral = np.zeros((height, width), dtype=np.uint32)

    # 用于查找可以放置单词的新位置
    def sample_position(self, size_x, size_y, random_state):
        """
            随机找到一个 可以放单词的位置  (ps:cython 提升效率)
        """
        return query_integral_image(self.integral, size_x, size_y,
                                    random_state)

    def update(self, img_array, pos_x, pos_y):
        """
            插入并更新积分图数据
        """
        # 首先，代码通过 np.cumsum 函数计算出从给定位置开始的部分积分图像（partial_integral）。
        # 这里的 img_array 是原始图像的像素数组，通过对其进行累加操作可以快速计算出积分图像中指定位置及其以下行和右侧列的所有像素值之和。
        partial_integral = np.cumsum(np.cumsum(img_array[pos_x:, pos_y:],
                                               axis=1), axis=0)
        # paste recomputed part into old image
        # if x or y is zero it is a bit annoying
        # 接下来，代码将重新计算的部分积分图像插入到旧的积分图像中。这里需要考虑两种情况：
        if pos_x > 0:
            # 需要将重新计算的部分积分图像与旧积分图像中对应位置及其以上行和右侧列的像素值相加。
            if pos_y > 0:
                # 这里通过使用 self.integral[pos_x - 1, pos_y:] - self.integral[pos_x - 1, pos_y - 1] 来计算旧积分图像中对应区域的像素值总和。
                partial_integral += (self.integral[pos_x - 1, pos_y:]
                                     - self.integral[pos_x - 1, pos_y - 1])
            else:
                # 需要将重新计算的部分积分图像与旧积分图像中对应位置及其左侧列的像素值相加。
                partial_integral += self.integral[pos_x - 1, pos_y:]
        if pos_y > 0:
            # 这里通过使用 self.integral[pos_x:, pos_y - 1][:, np.newaxis] 来计算旧积分图像中对应区域的像素值总和。
            partial_integral += self.integral[pos_x:, pos_y - 1][:, np.newaxis]

        # 最后，代码将更新后的积分图像存储在 self.integral 中。
        # 通过更新积分图像，可以在常数时间内计算任意大小和位置的矩形内的像素值总和，这对于一些计算机视觉算法非常重要。
        self.integral[pos_x:, pos_y:] = partial_integral


def random_color_func(word=None, font_size=None, position=None,
                      orientation=None, font_path=None, random_state=None):
    """
    这段代码的作用是随机生成一种 HSL（色相、饱和度、亮度）颜色，并将其以字符串的形式返回。
    """
    #
    if random_state is None:
        random_state = Random()
    return "hsl(%d, 80%%, 50%%)" % random_state.randint(0, 255)


class colormap_color_func(object):
    """
    创建一个颜色映射对象（colormap），用于将标量数据映射到对应的颜色值。
    -------
    >>> WordCloud(color_func=colormap_color_func("magma"))
    """

    def __init__(self, colormap):
        """
            生成一个Colormap对象。
            colormap: 用于对一组数据进行颜色映射的对象，它将数据映射到一系列颜色上，可以用于在可视化中显示数据。
        """

        import matplotlib.pyplot as plt

        self.colormap = plt.cm.get_cmap(colormap)

    def __call__(self, word, font_size, position, orientation,
                 random_state=None, **kwargs):
        """
        随机的 RGB 颜色
        前4个参数未使用
        """
        if random_state is None:
            random_state = Random()
        r, g, b, _ = np.maximum(0, 255 * np.array(self.colormap(
            random_state.uniform(0, 1))))
        return "rgb({:.0f}, {:.0f}, {:.0f})".format(r, g, b)


def get_single_color_func(color):
    """
        color => rgb 或 hsv => rgb
    >>> color_func1 = get_single_color_func('deepskyblue')
    >>> color_func2 = get_single_color_func('#00b4d2')
    """
    old_r, old_g, old_b = ImageColor.getrgb(color)
    rgb_max = 255.
    h, s, v = colorsys.rgb_to_hsv(old_r / rgb_max, old_g / rgb_max,
                                  old_b / rgb_max)

    def single_color_func(word=None, font_size=None, position=None,
                          orientation=None, font_path=None, random_state=None):
        """
        随机颜色生成 HSV 转成 rgb
        """
        if random_state is None:
            random_state = Random()
        r, g, b = colorsys.hsv_to_rgb(h, s, random_state.uniform(0.2, 1))
        return 'rgb({:.0f}, {:.0f}, {:.0f})'.format(r * rgb_max, g * rgb_max,
                                                    b * rgb_max)

    return single_color_func


class WordCloud(object):
    r"""Word cloud object for generating and drawing.

    Parameters
    ----------
    font_path : string
        Font path to the font that will be used (OTF or TTF).
        Defaults to DroidSansMono path on a Linux machine. If you are on
        another OS or don't have this font, you need to adjust this path.

    width : int (default=400)
        Width of the canvas.

    height : int (default=200)
        Height of the canvas.

    prefer_horizontal : float (default=0.90)
        The ratio of times to try horizontal fitting as opposed to vertical.
        If prefer_horizontal < 1, the algorithm will try rotating the word
        if it doesn't fit. (There is currently no built-in way to get only
        vertical words.)

    mask : nd-array or None (default=None)
        If not None, gives a binary mask on where to draw words. If mask is not
        None, width and height will be ignored and the shape of mask will be
        used instead. All white (#FF or #FFFFFF) entries will be considerd
        "masked out" while other entries will be free to draw on. [This
        changed in the most recent version!]

    contour_width: float (default=0)
        If mask is not None and contour_width > 0, draw the mask contour.

    contour_color: color value (default="black")
        Mask contour color.

    scale : float (default=1)
        Scaling between computation and drawing. For large word-cloud images,
        using scale instead of larger canvas size is significantly faster, but
        might lead to a coarser fit for the words.

    min_font_size : int (default=4)
        Smallest font size to use. Will stop when there is no more room in this
        size.

    font_step : int (default=1)
        Step size for the font. font_step > 1 might speed up computation but
        give a worse fit.

    max_words : number (default=200)
        The maximum number of words.

    stopwords : set of strings or None
        The words that will be eliminated. If None, the build-in STOPWORDS
        list will be used. Ignored if using generate_from_frequencies.

    background_color : color value (default="black")
        Background color for the word cloud image.

    max_font_size : int or None (default=None)
        Maximum font size for the largest word. If None, height of the image is
        used.

    mode : string (default="RGB")
        Transparent background will be generated when mode is "RGBA" and
        background_color is None.

    relative_scaling : float (default='auto')
        Importance of relative word frequencies for font-size.  With
        relative_scaling=0, only word-ranks are considered.  With
        relative_scaling=1, a word that is twice as frequent will have twice
        the size.  If you want to consider the word frequencies and not only
        their rank, relative_scaling around .5 often looks good.
        If 'auto' it will be set to 0.5 unless repeat is true, in which
        case it will be set to 0.

        .. versionchanged: 2.0
            Default is now 'auto'.

    color_func : callable, default=None
        Callable with parameters word, font_size, position, orientation,
        font_path, random_state that returns a PIL color for each word.
        Overwrites "colormap".
        See colormap for specifying a matplotlib colormap instead.
        To create a word cloud with a single color, use
        ``color_func=lambda *args, **kwargs: "white"``.
        The single color can also be specified using RGB code. For example
        ``color_func=lambda *args, **kwargs: (255,0,0)`` sets color to red.

    regexp : string or None (optional)
        Regular expression to split the input text into tokens in process_text.
        If None is specified, ``r"\w[\w']+"`` is used. Ignored if using
        generate_from_frequencies.

    collocations : bool, default=True
        Whether to include collocations (bigrams) of two words. Ignored if using
        generate_from_frequencies.


        .. versionadded: 2.0

    colormap : string or matplotlib colormap, default="viridis"
        Matplotlib colormap to randomly draw colors from for each word.
        Ignored if "color_func" is specified.

        .. versionadded: 2.0

    normalize_plurals : bool, default=True
        Whether to remove trailing 's' from words. If True and a word
        appears with and without a trailing 's', the one with trailing 's'
        is removed and its counts are added to the version without
        trailing 's' -- unless the word ends with 'ss'. Ignored if using
        generate_from_frequencies.

    repeat : bool, default=False
        Whether to repeat words and phrases until max_words or min_font_size
        is reached.

    include_numbers : bool, default=False
        Whether to include numbers as phrases or not.

    min_word_length : int, default=0
        Minimum number of letters a word must have to be included.

    collocation_threshold: int, default=30
        Bigrams must have a Dunning likelihood collocation score greater than this
        parameter to be counted as bigrams. Default of 30 is arbitrary.

        See Manning, C.D., Manning, C.D. and Schütze, H., 1999. Foundations of
        Statistical Natural Language Processing. MIT press, p. 162
        https://nlp.stanford.edu/fsnlp/promo/colloc.pdf#page=22

    Attributes
    ----------
    ``words_`` : dict of string to float
        Word tokens with associated frequency.

        .. versionchanged: 2.0
            ``words_`` is now a dictionary

    ``layout_`` : list of tuples ((string, float), int, (int, int), int, color))
        Encodes the fitted word cloud. For each word, it encodes the string, 
        normalized frequency, font size, position, orientation, and color.
        The frequencies are normalized by the most commonly occurring word.
        The color is in the format of 'rgb(R, G, B).'

    Notes
    -----
    Larger canvases make the code significantly slower. If you need a
    large word cloud, try a lower canvas size, and set the scale parameter.

    The algorithm might give more weight to the ranking of the words
    than their actual frequencies, depending on the ``max_font_size`` and the
    scaling heuristic.
    """

    def __init__(self, font_path=None, width=400, height=200, margin=2,
                 ranks_only=None, prefer_horizontal=.9, mask=None, scale=1,
                 color_func=None, max_words=200, min_font_size=4,
                 stopwords=None, random_state=None, background_color='black',
                 max_font_size=None, font_step=1, mode="RGB",
                 relative_scaling='auto', regexp=None, collocations=True,
                 colormap=None, normalize_plurals=True, contour_width=0,
                 contour_color='black', repeat=False,
                 include_numbers=False, min_word_length=0, collocation_threshold=30):
        # 这段代码的作用是对字体路径和颜色函数进行检查和设置。
        # 如果没有指定字体路径，则使用默认路径 FONT_PATH。
        # 如果没有指定颜色函数 color_func 且也没有指定颜色映射 colormap，则代码检查 matplotlib 的版本号。
        # 如果版本号低于 2.5，则使用颜色映射 hsv，否则使用 viridis。
        #
        # 这段代码的意图是为了在没有指定颜色函数或颜色映射时提供一个默认的颜色方案。
        # 在旧版本的 matplotlib 中，hsv 色彩映射常用于可视化中，而在新版本中，viridis 成为了 matplotlib 的默认色彩映射。
        # 由于这段代码的作用域是在一个 WordCloud 类中，因此这些默认值将在生成词云时使用。
        if font_path is None:
            font_path = FONT_PATH
        if color_func is None and colormap is None:
            version = matplotlib.__version__
            if version[0] < "2" and version[2] < "5":
                colormap = "hsv"
            else:
                colormap = "viridis"
        self.colormap = colormap
        self.collocations = collocations
        self.font_path = font_path
        self.width = width
        self.height = height
        self.margin = margin
        self.prefer_horizontal = prefer_horizontal
        self.mask = mask
        self.contour_color = contour_color
        self.contour_width = contour_width
        self.scale = scale
        self.color_func = color_func or colormap_color_func(colormap)
        self.max_words = max_words
        self.stopwords = stopwords if stopwords is not None else STOPWORDS
        self.min_font_size = min_font_size
        self.font_step = font_step
        self.regexp = regexp
        if isinstance(random_state, int):
            random_state = Random(random_state)
        self.random_state = random_state
        self.background_color = background_color
        self.max_font_size = max_font_size
        self.mode = mode

        if relative_scaling == "auto":
            if repeat:
                relative_scaling = 0
            else:
                relative_scaling = .5

        if relative_scaling < 0 or relative_scaling > 1:
            raise ValueError("relative_scaling needs to be "
                             "between 0 and 1, got %f." % relative_scaling)
        self.relative_scaling = relative_scaling
        if ranks_only is not None:
            warnings.warn("ranks_only is deprecated and will be removed as"
                          " it had no effect. Look into relative_scaling.",
                          DeprecationWarning)
        self.normalize_plurals = normalize_plurals
        self.repeat = repeat
        self.include_numbers = include_numbers
        self.min_word_length = min_word_length
        self.collocation_threshold = collocation_threshold

        # Override the width and height if there is a mask
        if mask is not None:
            self.width = mask.shape[1]
            self.height = mask.shape[0]

    def fit_words(self, frequencies):
        """Create a word_cloud from words and frequencies.

        Alias to generate_from_frequencies.

        Parameters
        ----------
        frequencies : dict from string to float
            A contains words and associated frequency.

        Returns
        -------
        self
        """
        return self.generate_from_frequencies(frequencies)

    def generate_from_frequencies(self, frequencies, max_font_size=None):  # noqa: C901
        """
            对词频字典进行处理, 按照词频从高到低排序
            保证至少有一个随机器
            处理遮罩图像
            获取宽高
            生成积分图
            处理背景图片（灰度）
            计算单词字体 max值
            计算词频 并圈定 要使用的单词
            根据词频计算字体大小 并绘制
        """
        frequencies = sorted(frequencies.items(), key=itemgetter(1), reverse=True)
        # 如果字典中没有任何词频信息，则会抛出 ValueError 异常。
        if len(frequencies) <= 0:
            raise ValueError("We need at least 1 word to plot a word cloud, "
                             "got %d." % len(frequencies))
        # 并限制了最大的词数
        frequencies = frequencies[:self.max_words]

        # largest entry will be 1 // 最大的条目是1
        # x计算词频的最大值，并将所有词的频率除以该最大值，从而使频率在0和1之间；
        max_frequency = float(frequencies[0][1])

        frequencies = [(word, freq / max_frequency)
                       for word, freq in frequencies]

        # 保证至少有一个随机器
        if self.random_state is not None:
            random_state = self.random_state
        else:
            random_state = Random()

        # 处理 遮罩图像(如果有的话)
        if self.mask is not None:
            boolean_mask = self._get_bolean_mask(self.mask)
            width = self.mask.shape[1]
            height = self.mask.shape[0]
        else:
            boolean_mask = None
            height, width = self.height, self.width
        # 使用高度、宽度和布尔掩码创建一个 积分图 对象
        occupancy = IntegralOccupancyMap(height, width, boolean_mask)

        # create image
        # 这段代码创建了一张灰度图像，使用PIL库中的Image模块创建了一个新的大小为(width, height)的灰度图像img_grey，
        img_grey = Image.new("L", (width, height))
        # 然后通过ImageDraw模块创建一个可绘制的对象draw来绘制图像。
        draw = ImageDraw.Draw(img_grey)
        # numpy库中的asarray函数被用来将图像转换为numpy数组。
        img_array = np.asarray(img_grey)
        # 接下来定义了四个空列表变量font_sizes, positions, orientations, colors，
        # 用来存储每个词汇在生成的词云中的字体大小、位置、方向和颜色等信息。
        font_sizes, positions, orientations, colors = [], [], [], []

        # 最后定义了一个变量last_freq并初始化为1.
        last_freq = 1.

        # 这段代码用于设置单词的 最大 字体，如果没有提供，则使用默认值。
        if max_font_size is None:
            # if not provided use default font_size
            max_font_size = self.max_font_size

        # 如果还是没有默认值，则会尝试通过尝试用前两个单词生成一个词云并计算字体大小。
        if max_font_size is None:
            # figure out a good font size by trying to draw with
            # just the first two words
            # 如果只有一个单词，则字体大小设置为画布高度。
            if len(frequencies) == 1:
                # we only have one word. We make it big!
                font_size = self.height
            else:
                # 通过尝试用前两个单词生成一个词云并计算字体大小。
                self.generate_from_frequencies(dict(frequencies[:2]),
                                               max_font_size=self.height)
                # find font sizes
                sizes = [x[1] for x in self.layout_]
                try:
                    # 如果计算大小失败，则会引发ValueError。
                    font_size = int(2 * sizes[0] * sizes[1]
                                    / (sizes[0] + sizes[1]))
                # quick fix for if self.layout_ contains less than 2 values
                # on very small images it can be empty
                except IndexError:
                    try:
                        font_size = sizes[0]
                    except IndexError:
                        raise ValueError(
                            "Couldn't find space to draw. Either the Canvas size"
                            " is too small or too much of the image is masked "
                            "out.")
        else:
            # 如果提供了max_font_size，则直接使用该值。
            font_size = max_font_size

        # we set self.words_ here because we called generate_from_frequencies
        # above... hurray for good design?
        # 这段代码用于处理词频信息。
        # 首先将词频信息转化为一个字典，其中词语为键，词频为值，保存在self.words_属性中。
        self.words_ = dict(frequencies)

        # 如果设置了repeat=True并且词频信息的长度小于max_words，则会在词频信息的基础上添加重复的词语以填充至max_words个词语。
        if self.repeat and len(frequencies) < self.max_words:
            # pad frequencies with repeating words.
            # 这行代码的意思是计算需要重复几次原始词频列表才能填充满指定的 max_words 个单词，然后将结果存储在变量 times_extend 中。
            times_extend = int(np.ceil(self.max_words / len(frequencies))) - 1
            # get smallest frequency
            # 这段代码是用来扩充词频列表以便填满整个词云
            # 首先，将原始单词频率列表复制一份到 frequencies_org 变量中。
            frequencies_org = list(frequencies)
            # 使用 downweight 变量存储最小频率，即频率列表中最后一个元素的频率。
            downweight = frequencies[-1][1]
            # 然后使用一个循环，该循环次数为 times_extend，即需要添加到频率列表中的单词数量。  其中 i 从 0 开始递增，表示单词重复的次数。
            for i in range(times_extend):
                # 对于 frequencies_org 列表中的每个单词和其对应的频率，都将它们与一个下降权重（即 downweight ** (i + 1)）相乘，
                # 将生成的单词和频率元组列表添加到 frequencies 列表中，最终得到包含 max_words 个单词的频率列表。
                frequencies.extend([(word, freq * downweight ** (i + 1))
                                    for word, freq in frequencies_org])

        # start drawing grey image
        for word, freq in frequencies:

            # 首先，通过遍历每个单词和它的频率来进行操作，如果当前单词频率为0，则跳过它。
            if freq == 0:
                continue
            # select the font size
            # 接下来是选择字体大小的步骤，这里采用了相对缩放（relative_scaling）的方式。
            # relative_scaling表示单词频率和字体大小之间的相对比例，
            rs = self.relative_scaling
            if rs != 0:
                # 设置为0则所有单词使用相同的字体大小，
                # 设置为1则单词大小与频率呈线性关系，freq/last_freq的比值是当前单词的相对频率。
                font_size = int(round((rs * (freq / float(last_freq))
                                       + (1 - rs)) * font_size))

            # 随机单词的方向，是横向或纵向。
            if random_state.random() < self.prefer_horizontal:
                # orientation 是一个变量，用于保存单词的方向，如果为 None，则单词以横向排列；
                # 如果为 Image.ROTATE_90，则单词以纵向排列。
                orientation = None
            else:
                orientation = Image.ROTATE_90

            # 这段代码实现了一个寻找单词在词云图中的位置的过程。
            # 它采用了一个循环的方法，不断尝试不同的字体大小和方向，直到找到单词在图中的位置，或者字体大小小于最小字体大小。
            #
            tried_other_orientation = False
            while True:
                # try to find a position
                # 创建一个指定字体、指定字号的字体对象
                font = ImageFont.truetype(self.font_path, font_size)
                # transpose font optionally
                # 将一个字体进行翻转或旋转，生成一个新的字体对象。TransposedFont是PIL库中的一个类，用于创建具有新方向或布局的字体。
                transposed_font = ImageFont.TransposedFont(
                    font, orientation=orientation)
                # get size of resulting text
                # 通过在给定字体下计算文本 word 渲染的像素大小得到 box大小
                box_size = draw.textsize(word, font=transposed_font)
                # find possible places using integral image:
                # 使用积分图 找到可用的文本区域。
                result = occupancy.sample_position(box_size[1] + self.margin,
                                                   box_size[0] + self.margin,
                                                   random_state)
                if result is not None or font_size < self.min_font_size:
                    # either we found a place or font-size went too small
                    break
                # 没找到合适单词位置时继续循环
                # if we didn't find a place, make font smaller
                # but first try to rotate!
                if not tried_other_orientation and self.prefer_horizontal < 1:
                    # 旋转90度
                    orientation = (Image.ROTATE_90 if orientation is None else
                                   Image.ROTATE_90)
                    tried_other_orientation = True
                else:
                    # 字体变小
                    font_size -= self.font_step
                    orientation = None

            if font_size < self.min_font_size:
                # we were unable to draw any more
                break

            # 这段代码是在确定每个单词的位置、旋转角度、字体大小、颜色等信息后，
            # 使用draw.text方法在图像上绘制单词。
            x, y = np.array(result) + self.margin // 2
            # actually draw the text
            draw.text((y, x), word, fill="white", font=transposed_font)
            positions.append((x, y))
            orientations.append(orientation)
            font_sizes.append(font_size)

            # 这段代码是在确定每个单词的位置、旋转角度、字体大小、颜色等信息后，使用draw.text方法在图像上绘制单词。
            colors.append(self.color_func(word, font_size=font_size,  # font_sizes 列表存储单词的字体大小，
                                          position=(x, y),  # positions 列表存储单词的左上角坐标，
                                          orientation=orientation,  # orientations 列表存储单词的旋转角度，
                                          random_state=random_state,
                                          font_path=self.font_path))
            # recompute integral image
            # 这段代码的作用是根据设置的参数将图像转化为一个数组
            if self.mask is None:
                img_array = np.asarray(img_grey)
            else:
                img_array = np.asarray(img_grey) + boolean_mask
            # recompute bottom right
            # the order of the cumsum's is important for speed ?!
            # 更新 积分图 数据
            occupancy.update(img_array, x, y)
            last_freq = freq

        self.layout_ = list(zip(frequencies, font_sizes, positions,
                                orientations, colors))
        return self

    def process_text(self, text):
        """
            从给定的文本中提取单词并进行预处理，包括过滤长度过短的单词、去除停用词、统计单词出现次数等操作。
        """

        # 只用正则查询全部的 文字(ps:单个字母、数字或下划线)
        flags = (re.UNICODE if sys.version < '3' and type(text) is unicode  # noqa: F821
                 else 0)
        pattern = r"\w[\w']*" if self.min_word_length <= 1 else r"\w[\w']+"
        regexp = self.regexp if self.regexp is not None else pattern

        words = re.findall(regexp, text, flags)
        # remove 's
        words = [word[:-2] if word.lower().endswith("'s") else word
                 for word in words]
        # 设置了不包括数字（include_numbers=False），则去除words中的数字。
        if not self.include_numbers:
            words = [word for word in words if not word.isdigit()]
        # 设置了最小单词长度（min_word_length不为0），则去除words中长度小于min_word_length的单词。
        if self.min_word_length:
            words = [word for word in words if len(word) >= self.min_word_length]

        # 如果设置了停用词（stopwords不为空），则去除words中的停用词。
        stopwords = set([i.lower() for i in self.stopwords])
        # 如果设置了二元组合（collocations=True），则统计words中单词和二元组（两个单词组成的组合）的出现次数。
        if self.collocations:
            word_counts = unigrams_and_bigrams(words, stopwords, self.normalize_plurals, self.collocation_threshold)
        else:
            # remove stopwords
            words = [word for word in words if word.lower() not in stopwords]
            word_counts, _ = process_tokens(words, self.normalize_plurals)

        return word_counts

    def generate_from_text(self, text):
        """
            从文本中生成词云
        """
        words = self.process_text(text)
        self.generate_from_frequencies(words)
        return self

    def generate(self, text):
        """
            从文本中生成词云
        """
        return self.generate_from_text(text)

    def _check_generated(self):
        """Check if ``layout_`` was computed, otherwise raise error."""
        if not hasattr(self, "layout_"):
            raise ValueError("WordCloud has not been calculated, call generate"
                             " first.")

    def to_image(self):
        """
            将文字画到图片上
        """
        self._check_generated()
        if self.mask is not None:
            width = self.mask.shape[1]
            height = self.mask.shape[0]
        else:
            height, width = self.height, self.width

        # 首先确定宽高并 创建一个空白的图片
        img = Image.new(self.mode, (int(width * self.scale),
                                    int(height * self.scale)),
                        self.background_color)

        draw = ImageDraw.Draw(img)
        # 循环依次处理每一个词汇，获取该词汇的字体、位置、方向和颜色信息，将该词汇使用指定的字体和颜色在 img 上绘制出来。
        for (word, count), font_size, position, orientation, color in self.layout_:
            font = ImageFont.truetype(self.font_path,
                                      int(font_size * self.scale))
            transposed_font = ImageFont.TransposedFont(
                font, orientation=orientation)
            pos = (int(position[1] * self.scale),
                   int(position[0] * self.scale))
            draw.text(pos, word, fill=color, font=transposed_font)

        return self._draw_contour(img=img)

    def recolor(self, random_state=None, color_func=None, colormap=None):
        """
        重新上色
        """

        # 检查随机数生成器是 单例的
        if isinstance(random_state, int):
            random_state = Random(random_state)
        self._check_generated()

        # 确认 color_func 颜色函数的存在
        if color_func is None:
            if colormap is None:
                color_func = self.color_func
            else:
                color_func = colormap_color_func(colormap)
        # 列表中的每一个元素都是一个元组，包括单词频率、字体大小、位置、旋转角度和颜色等信息。
        self.layout_ = [(word_freq, font_size, position, orientation,
                         color_func(word=word_freq[0], font_size=font_size,
                                    position=position, orientation=orientation,
                                    random_state=random_state,
                                    font_path=self.font_path))
                        for word_freq, font_size, position, orientation, _
                        in self.layout_]
        return self

    def to_file(self, filename):
        """Export to image file.

        Parameters
        ----------
        filename : string
            Location to write to.

        Returns
        -------
        self
        """

        img = self.to_image()
        img.save(filename, optimize=True)
        return self

    def to_array(self):
        """Convert to numpy array.

        Returns
        -------
        image : nd-array size (width, height, 3)
            Word cloud image as numpy matrix.
        """
        return np.array(self.to_image())

    def __array__(self):
        """Convert to numpy array.

        Returns
        -------
        image : nd-array size (width, height, 3)
            Word cloud image as numpy matrix.
        """
        return self.to_array()

    def to_svg(self, embed_font=False, optimize_embedded_font=True, embed_image=False):
        """Export to SVG.

        Font is assumed to be available to the SVG reader. Otherwise, text
        coordinates may produce artifacts when rendered with replacement font.
        It is also possible to include a subset of the original font in WOFF
        format using ``embed_font`` (requires `fontTools`).

        Note that some renderers do not handle glyphs the same way, and may
        differ from ``to_image`` result. In particular, Complex Text Layout may
        not be supported. In this typesetting, the shape or positioning of a
        grapheme depends on its relation to other graphemes.

        Pillow, since version 4.2.0, supports CTL using ``libraqm``. However,
        due to dependencies, this feature is not always enabled. Hence, the
        same rendering differences may appear in ``to_image``. As this
        rasterized output is used to compute the layout, this also affects the
        layout generation. Use ``PIL.features.check`` to test availability of
        ``raqm``.

        Consistant rendering is therefore expected if both Pillow and the SVG
        renderer have the same support of CTL.

        Contour drawing is not supported.

        Parameters
        ----------
        embed_font : bool, default=False
            Whether to include font inside resulting SVG file.

        optimize_embedded_font : bool, default=True
            Whether to be aggressive when embedding a font, to reduce size. In
            particular, hinting tables are dropped, which may introduce slight
            changes to character shapes (w.r.t. `to_image` baseline).

        embed_image : bool, default=False
            Whether to include rasterized image inside resulting SVG file.
            Useful for debugging.

        Returns
        -------
        content : string
            Word cloud image as SVG string
        """

        # TODO should add option to specify URL for font (i.e. WOFF file)

        # Make sure layout is generated
        self._check_generated()

        # Get output size, in pixels
        if self.mask is not None:
            width = self.mask.shape[1]
            height = self.mask.shape[0]
        else:
            height, width = self.height, self.width

        # Get max font size
        if self.max_font_size is None:
            max_font_size = max(w[1] for w in self.layout_)
        else:
            max_font_size = self.max_font_size

        # Text buffer
        result = []

        # Get font information
        font = ImageFont.truetype(self.font_path, int(max_font_size * self.scale))
        raw_font_family, raw_font_style = font.getname()
        # TODO properly escape/quote this name?
        font_family = repr(raw_font_family)
        # TODO better support for uncommon font styles/weights?
        raw_font_style = raw_font_style.lower()
        if 'bold' in raw_font_style:
            font_weight = 'bold'
        else:
            font_weight = 'normal'
        if 'italic' in raw_font_style:
            font_style = 'italic'
        elif 'oblique' in raw_font_style:
            font_style = 'oblique'
        else:
            font_style = 'normal'

        # Add header
        result.append(
            '<svg'
            ' xmlns="http://www.w3.org/2000/svg"'
            ' width="{}"'
            ' height="{}"'
            '>'
                .format(
                width * self.scale,
                height * self.scale
            )
        )

        # Embed font, if requested
        if embed_font:
            # Import here, to avoid hard dependency on fonttools
            import fontTools
            import fontTools.subset

            # Subset options
            options = fontTools.subset.Options(

                # Small impact on character shapes, but reduce size a lot
                hinting=not optimize_embedded_font,

                # On small subsets, can improve size
                desubroutinize=optimize_embedded_font,

                # Try to be lenient
                ignore_missing_glyphs=True,
            )

            # Load and subset font
            ttf = fontTools.subset.load_font(self.font_path, options)
            subsetter = fontTools.subset.Subsetter(options)
            characters = {c for item in self.layout_ for c in item[0][0]}
            text = ''.join(characters)
            subsetter.populate(text=text)
            subsetter.subset(ttf)

            # Export as WOFF
            # TODO is there a better method, i.e. directly export to WOFF?
            buffer = io.BytesIO()
            ttf.saveXML(buffer)
            buffer.seek(0)
            woff = fontTools.ttLib.TTFont(flavor='woff')
            woff.importXML(buffer)

            # Create stylesheet with embedded font face
            buffer = io.BytesIO()
            woff.save(buffer)
            data = base64.b64encode(buffer.getbuffer()).decode('ascii')
            url = 'data:application/font-woff;charset=utf-8;base64,' + data
            result.append(
                '<style>'
                '@font-face{{'
                'font-family:{};'
                'font-weight:{};'
                'font-style:{};'
                'src:url("{}")format("woff");'
                '}}'
                '</style>'
                    .format(
                    font_family,
                    font_weight,
                    font_style,
                    url
                )
            )

        # Select global style
        result.append(
            '<style>'
            'text{{'
            'font-family:{};'
            'font-weight:{};'
            'font-style:{};'
            '}}'
            '</style>'
                .format(
                font_family,
                font_weight,
                font_style
            )
        )

        # Add background
        if self.background_color is not None:
            result.append(
                '<rect'
                ' width="100%"'
                ' height="100%"'
                ' style="fill:{}"'
                '>'
                '</rect>'
                    .format(self.background_color)
            )

        # Embed image, useful for debug purpose
        if embed_image:
            image = self.to_image()
            data = io.BytesIO()
            image.save(data, format='JPEG')
            data = base64.b64encode(data.getbuffer()).decode('ascii')
            result.append(
                '<image'
                ' width="100%"'
                ' height="100%"'
                ' href="data:image/jpg;base64,{}"'
                '/>'
                    .format(data)
            )

        # For each word in layout
        for (word, count), font_size, (y, x), orientation, color in self.layout_:
            x *= self.scale
            y *= self.scale

            # Get text metrics
            font = ImageFont.truetype(self.font_path, int(font_size * self.scale))
            (size_x, size_y), (offset_x, offset_y) = font.font.getsize(word)
            ascent, descent = font.getmetrics()

            # Compute text bounding box
            min_x = -offset_x
            max_x = size_x - offset_x
            max_y = ascent - offset_y

            # Compute text attributes
            attributes = {}
            if orientation == Image.ROTATE_90:
                x += max_y
                y += max_x - min_x
                transform = 'translate({},{}) rotate(-90)'.format(x, y)
            else:
                x += min_x
                y += max_y
                transform = 'translate({},{})'.format(x, y)

            # Create node
            attributes = ' '.join('{}="{}"'.format(k, v) for k, v in attributes.items())
            result.append(
                '<text'
                ' transform="{}"'
                ' font-size="{}"'
                ' style="fill:{}"'
                '>'
                '{}'
                '</text>'
                    .format(
                    transform,
                    font_size * self.scale,
                    color,
                    saxutils.escape(word)
                )
            )

        # TODO draw contour

        # Complete SVG file
        result.append('</svg>')
        return '\n'.join(result)

    def _get_bolean_mask(self, mask):
        """Cast to two dimensional boolean mask."""
        if mask.dtype.kind == 'f':
            warnings.warn("mask image should be unsigned byte between 0"
                          " and 255. Got a float array")
        if mask.ndim == 2:
            boolean_mask = mask == 255
        elif mask.ndim == 3:
            # if all channels are white, mask out
            boolean_mask = np.all(mask[:, :, :3] == 255, axis=-1)
        else:
            raise ValueError("Got mask of invalid shape: %s" % str(mask.shape))
        return boolean_mask

    def _draw_contour(self, img):
        """Draw mask contour on a pillow image."""
        if self.mask is None or self.contour_width == 0:
            return img

        mask = self._get_bolean_mask(self.mask) * 255
        contour = Image.fromarray(mask.astype(np.uint8))
        contour = contour.resize(img.size)
        contour = contour.filter(ImageFilter.FIND_EDGES)
        contour = np.array(contour)

        # make sure borders are not drawn before changing width
        contour[[0, -1], :] = 0
        contour[:, [0, -1]] = 0

        # use gaussian to change width, divide by 10 to give more resolution
        radius = self.contour_width / 10
        contour = Image.fromarray(contour)
        contour = contour.filter(ImageFilter.GaussianBlur(radius=radius))
        contour = np.array(contour) > 0
        contour = np.dstack((contour, contour, contour))

        # color the contour
        ret = np.array(img) * np.invert(contour)
        if self.contour_color != 'black':
            color = Image.new(img.mode, img.size, self.contour_color)
            ret += np.array(color) * contour

        return Image.fromarray(ret)
