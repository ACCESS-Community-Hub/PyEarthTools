# Copyright Commonwealth of Australia, Bureau of Meteorology 2024.
# This software is provided under license 'as is', without warranty 
# of any kind including, but not limited to, fitness for a particular 
# purpose. The user assumes the entire risk as to the use and 
# performance of the software. In no event shall the copyright holder 
# be held liable for any claim, damages or other liability arising 
# from the use of the software.

"""

"""

from edit.data.archive import register_archive

import edit_archive_template

from edit_archive_template.template import BaseTemplate
from edit_archive_template.structured import StructuredTemplate


register_archive("Template")(edit_archive_template)

__version__ = "2024.04.01"
