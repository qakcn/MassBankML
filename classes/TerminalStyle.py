# Copyright 2023 qakcn
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

################################################################################
## TerminalStyle - stylize text for pretty terminal output.                   ##
##                                                                            ##
## Author: qakcn                                                              ##
## Email: qakcn@hotmail.com                                                   ##
## Version: 2.1                                                               ##
## Date: 2023-12-05                                                           ##
################################################################################

# Change logs
# - 1.0 First version
# - 2.0 Add more ANSI color and RGB color support
# - 2.1 Add unsusal ANSI styles support, and complete the document

"""TerminalStyle - stylize text for pretty terminal output

Two ways to use this class: class method and instance method. A style can be 
called by name to stylizing text. All valid styles names can be printed by 
calling class method print_all_styles().

Usage:
    >>> from TerminalStyle import TS
    >>> TS.p(TS.green("Hello, world!")) # class method
    >>> TS("Hello, world!").green().p() # instance method
    >>> TS.p(TS("Hello, world!").green()) # mixed method

    >>> TS.green("Hello, world!").p() # this will failed since the value that a 
    class method returns should be a string or no returns.
"""

if __name__ == "__main__":
    raise SystemExit('This script is not meant to be run directly')

class TSMeta(type):
    """Terminal Style Meta Class"""
    def __getattr__(cls, name):
        def method(text):
            return cls._stylize(name, text)
        return method

class TS(metaclass=TSMeta):
    """TS - Terminal Styles, a class to stylize text for terminal output
    
    Class attributes:
    _simple_style (dict) -- simple styles name and code
    _unual_style (dict) -- unusual styles name and code
    _html_colors (dict) -- HTML color name and hex string

    Class methods:
    _style_str(style) -- return ANSI escape string
    _stylize(style, text) -- stylize text with style
    _indent(text, level, spaces) -- indent text
    _print(text, **kwargs) -- print text with styles
    _inline_print(text, end, **kwargs) -- print text with styles inline
    register_printer(printer) -- register a printer function
    print_all_styles(table) -- print all styles supported

    Wildcard class methods:
    <style>(text) -- stylize text with style

    Instance attributes:
    original_text (str) -- original text
    formated_text (str) -- formated text

    Instance methods:
    stylize(name) -- stylize text with style
    indent(level, spaces) -- indent text
    print(**kwargs) -- print text with styles
    inline_print(end, **kwargs) -- print text with styles inline

    Wildcard instance methods:
    <style>() -- stylize text with style

    Class or instance methods:
    p(**kwargs) -- shortcut for TS._print() or TS().print()
    ip(**kwargs) -- shortcut for TS._inline_print() or TS().inline_print()
    """
    
    _simple_style = dict(
        reset = 0,
        bold = 1,
        faint = 2,
        italic = 3,
        underline = 4,
        blink = 5,
        reverse = 7,
        concealed = 8,
        strike = 9,
        overline = 53,

        #colors
        black = 30,
        red = 31,
        green = 32,
        yellow = 33,
        blue = 34,
        magenta = 35,
        cyan = 36,
        white = 37,
        black_bg = 40,
        red_bg = 41,
        green_bg = 42,
        yellow_bg = 43,
        blue_bg = 44,
        magenta_bg = 45,
        cyan_bg = 46,
        white_bg = 47,

        bright_black = 90,
        bright_red = 91,
        bright_green = 92,
        bright_yellow = 93,
        bright_blue = 94,
        bright_magenta = 95,
        bright_cyan = 96,
        bright_white = 97,
        bright_black_bg = 100,
        bright_red_bg = 101,
        bright_green_bg = 102,
        bright_yellow_bg = 103,
        bright_blue_bg = 104,
        bright_magenta_bg = 105,
        bright_cyan_bg = 106,
        bright_white_bg = 107,
    )

    _unual_style = dict(
        rapid_blink = 6,
        pri_font = 10,
        alt_font_1 = 11,
        alt_font_2 = 12,
        alt_font_3 = 13,
        alt_font_4 = 14,
        alt_font_5 = 15,
        alt_font_6 = 16,
        alt_font_7 = 17,
        alt_font_8 = 18,
        alt_font_9 = 19,
        fraktur = 20,
        double_underline = 21,
        normal_color = 22,
        not_italic = 23,
        not_underline = 24,
        not_blink = 25,
        prop_spacing = 26,
        not_reverse = 27,
        not_concealed = 28,
        not_strike = 29,

        # color 30-37, 40-47, 90-97, 100-107 are already defined in _simple_style

        # commented due to being implemented in other way:
        # fg_color = 38, # 38;5;#m or 38;2;#;#;#m
        default_fg_color = 39,
        # commented due to being implemented in other way:
        # bg_color = 48, # 48;5;#m or 48;2;#;#;#m
        default_bg_color = 49,
        not_prop_spacing = 50,
        framed = 51,
        encircled = 52,
        not_framed_or_encircled = 54,
        not_overline = 55,
        ideogram_underline = 60,
        ideogram_double_underline = 61,
        ideogram_overline = 62,
        ideogram_double_overline = 63,
        ideogram_stress_marking = 64,
        not_ideogram = 65,
        superscript = 73,
        subscript = 74,
        not_super_or_sub = 75,
    )

    _html_colors = {
        "aliceblue": "f0f8ff",
        "antiquewhite": "faebd7",
        "aqua": "00ffff",
        "aquamarine": "7fffd4",
        "azure": "f0ffff",
        "beige": "f5f5dc",
        "bisque": "ffe4c4",
        "black": "000000",
        "blanchedalmond": "ffebcd",
        "blue": "0000ff",
        "blueviolet": "8a2be2",
        "brown": "a52a2a",
        "burlywood": "deb887",
        "cadetblue": "5f9ea0",
        "chartreuse": "7fff00",
        "chocolate": "d2691e",
        "coral": "ff7f50",
        "cornflowerblue": "6495ed",
        "cornsilk": "fff8dc",
        "crimson": "dc143c",
        "cyan": "00ffff",
        "darkblue": "00008b",
        "darkcyan": "008b8b",
        "darkgoldenrod": "b8860b",
        "darkgray": "a9a9a9",
        "darkgreen": "006400",
        "darkkhaki": "bdb76b",
        "darkmagenta": "8b008b",
        "darkolivegreen": "556b2f",
        "darkorange": "ff8c00",
        "darkorchid": "9932cc",
        "darkred": "8b0000",
        "darksalmon": "e9967a",
        "darkseagreen": "8fbc8f",
        "darkslateblue": "483d8b",
        "darkslategray": "2f4f4f",
        "darkturquoise": "00ced1",
        "darkviolet": "9400d3",
        "deeppink": "ff1493",
        "deepskyblue": "00bfff",
        "dimgray": "696969",
        "dodgerblue": "1e90ff",
        "feldspar": "d19275",
        "firebrick": "b22222",
        "floralwhite": "fffaf0",
        "forestgreen": "228b22",
        "fuchsia": "ff00ff",
        "gainsboro": "dcdcdc",
        "ghostwhite": "f8f8ff",
        "gold": "ffd700",
        "goldenrod": "daa520",
        "gray": "808080",
        "green": "008000",
        "greenyellow": "adff2f",
        "honeydew": "f0fff0",
        "hotpink": "ff69b4",
        "indianred": "cd5c5c",
        "indigo": "4b0082",
        "ivory": "fffff0",
        "khaki": "f0e68c",
        "lavender": "e6e6fa",
        "lavenderblush": "fff0f5",
        "lawngreen": "7cfc00",
        "lemonchiffon": "fffacd",
        "lightblue": "add8e6",
        "lightcoral": "f08080",
        "lightcyan": "e0ffff",
        "lightgoldenrodyellow": "fafad2",
        "lightgrey": "d3d3d3",
        "lightgreen": "90ee90",
        "lightpink": "ffb6c1",
        "lightsalmon": "ffa07a",
        "lightseagreen": "20b2aa",
        "lightskyblue": "87cefa",
        "lightslateblue": "8470ff",
        "lightslategray": "778899",
        "lightsteelblue": "b0c4de",
        "lightyellow": "ffffe0",
        "lime": "00ff00",
        "limegreen": "32cd32",
        "linen": "faf0e6",
        "magenta": "ff00ff",
        "maroon": "800000",
        "mediumaquamarine": "66cdaa",
        "mediumblue": "0000cd",
        "mediumorchid": "ba55d3",
        "mediumpurple": "9370d8",
        "mediumseagreen": "3cb371",
        "mediumslateblue": "7b68ee",
        "mediumspringgreen": "00fa9a",
        "mediumturquoise": "48d1cc",
        "mediumvioletred": "c71585",
        "midnightblue": "191970",
        "mintcream": "f5fffa",
        "mistyrose": "ffe4e1",
        "moccasin": "ffe4b5",
        "navajowhite": "ffdead",
        "navy": "000080",
        "oldlace": "fdf5e6",
        "olive": "808000",
        "olivedrab": "6b8e23",
        "orange": "ffa500",
        "orangered": "ff4500",
        "orchid": "da70d6",
        "palegoldenrod": "eee8aa",
        "palegreen": "98fb98",
        "paleturquoise": "afeeee",
        "palevioletred": "d87093",
        "papayawhip": "ffefd5",
        "peachpuff": "ffdab9",
        "peru": "cd853f",
        "pink": "ffc0cb",
        "plum": "dda0dd",
        "powderblue": "b0e0e6",
        "purple": "800080",
        "red": "ff0000",
        "rosybrown": "bc8f8f",
        "royalblue": "4169e1",
        "saddlebrown": "8b4513",
        "salmon": "fa8072",
        "sandybrown": "f4a460",
        "seagreen": "2e8b57",
        "seashell": "fff5ee",
        "sienna": "a0522d",
        "silver": "c0c0c0",
        "skyblue": "87ceeb",
        "slateblue": "6a5acd",
        "slategray": "708090",
        "snow": "fffafa",
        "springgreen": "00ff7f",
        "steelblue": "4682b4",
        "tan": "d2b48c",
        "teal": "008080",
        "thistle": "d8bfd8",
        "tomato": "ff6347",
        "turquoise": "40e0d0",
        "violet": "ee82ee",
        "violetred": "d02090",
        "wheat": "f5deb3",
        "white": "ffffff",
        "whitesmoke": "f5f5f5",
        "yellow": "ffff00",
        "yellowgreen": "9acd32",
    }

    _printer = print

    @classmethod
    def _style_str(cls, style: str) -> str:
        """Return ANSI escape string
        
        Arguments:
        style (str) -- style name

        Returns:
        str: ANSI escape code
        """
        tpl = "\033[{fob}{code}m"
        style = style.lower()
        if style in cls._simple_style:
            code = cls._simple_style[style]
            fob = ""
        elif style in cls._unual_style:
            code = cls._unual_style[style]
            fob = ""
        elif style.startswith("bg_") or style.startswith("fg_"):
            if style.startswith("fg_"):
                fob = "38;"
            elif style.startswith("bg_"):
                fob = "48;"
            color = style[3:]
            if color.isdigit() and int(color) in range(0, 256):
                code = f"5;{color}"
            elif color in cls._html_colors:
                return cls._style_str(style[:3] + "rgb_" + cls._html_colors[color])
            elif color.startswith("rgb_"):
                rgb = color[4:10]
                if len(rgb) == 6 and all([s in "01234567890abcdefABCDEF" for s in rgb]):
                    r = int(rgb[0:2], 16)
                    g = int(rgb[2:4], 16)
                    b = int(rgb[4:6], 16)
                    code = f"2;{r};{g};{b}"
                else:
                    raise ValueError("RGB value error, should be 6-digit hex.")
            else:
                raise ValueError("Color not supported.")
        else:
            raise ValueError("Style not supported.")
        return tpl.format(fob=fob, code=code)
    
    @classmethod
    def _stylize(cls, style: str, text) -> str:
        """Stylize text with style
        
        Arguments:
        style (str) -- style name
        text (str or convertable to str) -- text to stylize

        Returns:
        TS: stylized text
        """
        text = str(text)
        reset = cls._style_str('reset')
        if style == reset:
            return text
        if isinstance(text, str) and text.endswith(reset):
            text = text[:-4]
        start = cls._style_str(style)
        text = start + text + reset
        return text
    
    @classmethod
    def _indent(cls, text, level: int = 1, spaces: str = " "*4) -> str:
        """Indent text

        Arguments:
        text (str or convertable to str) -- text to indent
        level (int) -- indent level
        spaces (str) -- spaces for each level

        Returns:
        str: indented text
        """
        return spaces*level + str(text)
    
    @classmethod
    def _print(cls, text, **kwargs) -> None:
        """Print text with styles
        
        Arguments:
        text (str or convertable to str) -- text to print
        other arguments -- arguments for print function
        """
        ts = TS(text)
        ts.print(**kwargs)
    
    @classmethod
    def _inline_print(cls, text, end: str = "", **kwargs) -> None:
        """Print text with styles inline

        Arguments:
        text (str or convertable to str) -- text to print
        end (str) -- end string
        other arguments -- arguments for print function
        """
        cls._print(text, end=end, **kwargs)
    
    @classmethod
    def register_printer(cls, printer: callable) -> None:
        """Register a printer function

        Arguments:
        printer (callable) -- printer function
        """
        cls._printer = printer

    @classmethod
    def print_all_styles(cls, table="usual") -> None:
        """Print all styles supported
        
        Arguments:
        table (str or list or tuple) -- styles to print, "all" for all styles, "usual" for usual styles, or a list or tuple of one or more of "basic", "color", "rgb", "html" and "unusual".
        """
        if isinstance(table, str):
            if table == "all":
                table = ("basic", "color", "rgb", "html", "unusual")
            elif table == "usual":
                table = ("basic", "color", "rgb", "html")
            else:
                table = (table,)
        elif not isinstance(table, (list, tuple)):
            raise TypeError("table should be a string, list or tuple.")
        if "basic" in table:
            TS._print("+"*80)
            TS._print("")
            (TS("These are ") + TS("basic styles").bold() + ", use them by " + TS("name").blue() + TS(".\nAfter ") + TS("reverse").yellow() + " is \"" + TS("concealed").cyan() + "\", invisible because it's concealed.").print()
            TS._print("")
            for style in cls._simple_style:
                if style == "reset":
                    continue
                else:
                    if style in  ("blink", "overline", "black",  "black_bg", "blue", "blue_bg", "bright_black", "bright_black_bg", "bright_blue", "bright_blue_bg"):
                        TS._print("")
                    TS(f"{style:^20}").stylize(style).inline_print()
            TS._print("")
            TS._print("")

        if "color" in table:
            TS._print("+"*80)
            TS._print("")
            (TS("These are ") + TS("color codes").bold() + ", use them by " + TS("fg_<code>").blue() + " for foreground color or " + TS("bg_<code>").blue() + "\nfor background color. 0-15 are as same as those colors in basic styles.").print()
            TS._print("")

            for i in range(0, 16):
                if i != 0 and i % 8 == 0:
                    TS._print("")
                TS(f"{i:^5}").stylize(f"bg_{i}").inline_print()
            TS._print("")

            for i in range(16,256):
                if i-16 != 0 and (i-16) % 12 == 0:
                    TS._print("")
                TS(f"{i:^5}").stylize(f"bg_{i}").inline_print()
            TS._print("")
            TS._print("")

        if "rgb" in table:
            TS._print("+"*80)
            TS._print("")
            (TS("RGB hex color").bold() + " can be used by " + TS("fg_rgb_######").blue() + " or " + TS("bg_rgb_######").blue() + ", " + TS("######").yellow() + " is an \n8-digit hex, 2 for " + TS("red").fg_red() + ", 2 for " + TS("green").fg_lime() + " and 2 for " + TS("blue").fg_blue() + ". Here are some examples.").print()
            TS._print("")

            (TS(" fg_rgb_8fbc8f ").fg_rgb_8fbc8f() + "|" +
            TS(" fg_rgb_CD853F ").fg_rgb_CD853F() + "|" +
            TS(" fg_rgb_DdA0dD ").fg_rgb_DDA0DD() + "|" +
            TS(" fg_rgb_00fF7F ").fg_rgb_00ff7f()).print()

            (TS(" bg_rgb_8fbc8f ").bg_rgb_8fbc8f() + " " +
            TS(" bg_rgb_CD853F ").bg_rgb_CD853F() + " " +
            TS(" bg_rgb_DdA0dD ").bg_rgb_DDA0DD() + " " +
            TS(" bg_rgb_00fF7F ").bg_rgb_00ff7f()).print()
            TS._print("")

        if "html" in table:
            TS._print("+"*80)
            TS._print("")
            (TS("HTML color name").bold() + " can be used by " + TS("fg_<name>").blue() + " or " + TS("bg_<name>").blue() + ".\nYou can find them here:\n<" + TS("https://developer.mozilla.org/en-US/docs/Web/CSS/named-color>").fg_blue() +">.").print()
            TS._print("")

            (TS(" fg_Chocolate ").fg_Chocolate() + "|" +
            TS(" fg_Magenta ").fg_Magenta() + "|" +
            TS(" fg_Turquoise ").fg_Turquoise() + "|" +
            TS(" fg_AliceBlue ").fg_AliceBlue() + "|" +
            TS(" fg_Moccasin ").fg_Moccasin()
            ).print()

            (TS(" bg_Chocolate ").bg_Chocolate() + " " +
            TS(" bg_Magenta ").bg_Magenta() + " " +
            TS(" bg_Turquoise ").bg_Turquoise()  + " " +
            TS(" bg_AliceBlue ").bg_AliceBlue() + " " +
            TS(" bg_Moccasin ").bg_Moccasin()).print()

            TS._print("")

        if "unusual" in table:
            TS._print("+"*80)
            TS._print("")
            (TS("These are ") + TS("unusual styles").bold() + ", use them by " + TS("name").blue() + ".\nThese styles may not be fully supported by your terminal.").print()

            TS._print("")

            cnt = 0
            for style in cls._unual_style:
                if cnt != 0 and cnt % 3 == 0:
                    TS._print("")
                TS(f"{style:^25}").stylize(style).inline_print()
                cnt += 1
            TS._print("")
        TS._print("+"*80)

    def __init__(self, text, joint: str = " "):
        if isinstance(text, TS):
            self.original_text = text.original_text
            self.formated_text = text.formated_text
        else:
            if isinstance(text, (list, tuple, set)):
                text = joint.join([str(i) for i in text])
            self.formated_text = self.original_text = str(text)

    def stylize(self, name: str):
        """Stylize text with style
        
        Arguments:
        name (str) -- style name

        Returns:
        TS: with text stylized
        """
        self.formated_text = TS._stylize(name, self.formated_text)
        return self
    
    def indent(self, level: int = 1, spaces: str = " "*4):
        """Indent text
        
        Arguments:
        level (int) -- indent level
        spaces (str) -- spaces for each level

        Returns:
        TS: with text indented
        """
        self.formated_text = TS._indent(self.formated_text, level, spaces)
        return self
    
    def add(self, text, style: str = "reset", joint: str = " "):
        return self + TS(text, joint).stylize(style)
    
    def __getattr__(self, name: str):
        def method():
            return self.stylize(name)
        return method
    
    def __add__(self, other):
        other = TS(other)
        self.formated_text += other.formated_text
        self.original_text += other.original_text
        return self
    
    def __str__(self):
        return self.formated_text
    
    def print(self, **kwargs) -> None:
        """Print text with styles
        
        Arguments:
        other arguments -- arguments for print function
        """
        TS._printer(self.formated_text, **kwargs)
    
    def inline_print(self, end: str = "", **kwargs) -> None:
        """Print text with styles inline
        
        Arguments:
        end (str) -- end string
        other arguments -- arguments for print function
        """
        self.print(end=end, **kwargs)

    def p(cos, **kwargs) -> None:
        """Shortcut for TS._print() when called as class method, or TS().print() when called as instance method."""
        if isinstance(cos, TS):
            cos.print(**kwargs)
        else:
            TS._print(cos, **kwargs)

    def ip(cos, **kwargs) -> None:
        """Shortcut for TS._inline_print() when called as class method, or TS().inline_print() when called as instance method."""
        if isinstance(cos, TS):
            cos.inline_print(**kwargs)
        else:
            TS._inline_print(cos, **kwargs)
