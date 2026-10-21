"""Font utilities.

License
-------

Copyright (c) 2026 Shinei Arakawa
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import pathlib
import typing

DEFAULT_FONT_GOTHIC_PATH: typing.Final[pathlib.Path] = pathlib.Path(__file__).parent / 'fonts' / 'Roboto-Medium.ttf'
DEFAULT_CODE_FONT_PATH: typing.Final[pathlib.Path] = pathlib.Path(__file__).parent / 'fonts' / 'Hack-Regular.ttf'
