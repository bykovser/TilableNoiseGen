# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

bl_info = {
    "name": "TilableNoiseGen",
    "author": "BykovSer",
    "version": (1, 4),
    "blender": (4, 3, 0),
    "location": "Image Editor > N Panel > Noise Tools",
    "description": "Generates procedural noise patterns and connects to shaders",
    "category": "Material",
}

# Import the main module
from . import procedural_noise_generator


def register():
    procedural_noise_generator.register()

def unregister():
    procedural_noise_generator.unregister()