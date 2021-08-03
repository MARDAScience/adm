# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2020, Marda Science LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import tensorflow as tf

# urls = [
# "https://github.com/dbuscombe-usgs/aom/releases/download/0.0.1/20181006_VA_to_Oregon_Inlet_1m_DSM_adjExt_cog.tif",
# "https://github.com/dbuscombe-usgs/aom/releases/download/0.0.1/20181006_VA_to_Oregon_Inlet_RGBAvg_1m_NODATA_NAD83_2011_UTM18_cog.tif",
# "https://github.com/dbuscombe-usgs/aom/releases/download/0.0.1/20181007_Hattaras_Inlet_to_Ocracoke_Inlet_1m_DSM_adjExt_cog.tif",
# "https://github.com/dbuscombe-usgs/aom/releases/download/0.0.1/20181007_Hattaras_Inlet_to_Ocracoke_Inlet_RGBAvg_1m_NODATA_NAD83_2011_UTM18_cog.tif"
# ]

urls = [
# "https://github.com/dbuscombe-usgs/adm/releases/download/v0.0.1/clip_20181006_Hatteras_Inlet_to_Ocracoke_Inlet_1m_UTM18N_NAVD88_cog.tif",
# "https://github.com/dbuscombe-usgs/adm/releases/download/v0.0.1/clip_20181006_Ocracoke_Inlet_to_Ophelia_Inlet_1m_UTM18N_NAVD88_cog.tif",
# "https://github.com/dbuscombe-usgs/adm/releases/download/v0.0.1/clip_20181006_Ophelia_Inlet_to_Beaufort_Inlet_1m_UTM18N_NAVD88_cog.tif",
# "https://github.com/dbuscombe-usgs/adm/releases/download/v0.0.1/clip_20181006_VA_to_Oregon_Inlet_1m_UTM18N_NAVD88_cog.tif",
# "https://github.com/dbuscombe-usgs/adm/releases/download/v0.0.1/clip_20181007_Beaufort_Inlet_to_Bogue_Inlet_1m_UTM18N_NAVD88_cog.tif",
# "https://github.com/dbuscombe-usgs/adm/releases/download/v0.0.1/clip_20181007_Bogue_Inlet_to_New_River_Inlet_1m_UTM18N_NAVD88_cog.tif",
# "https://github.com/dbuscombe-usgs/adm/releases/download/v0.0.1/clip_20181007_Hatteras_Inlet_to_Ocracoke_Inlet_1m_UTM18N_NAVD88_cog.tif",
# "https://github.com/dbuscombe-usgs/adm/releases/download/v0.0.1/clip_20181007_Ocracoke_Inlet_to_Ophelia_Inlet_1m_UTM18N_NAVD88_cog.tif",
# "https://github.com/dbuscombe-usgs/adm/releases/download/v0.0.1/clip_20181007_Ophelia_Inlet_to_Beaufort_Inlet_1m_UTM18N_NAVD88_cog.tif",
# "https://github.com/dbuscombe-usgs/adm/releases/download/v0.0.1/clip_20181007_Oregon_Inlet_to_Hatteras_Inlet_1m_UTM18N_NAVD88_cog.tif",
# "https://github.com/dbuscombe-usgs/adm/releases/download/v0.0.1/clip_20181008_Hatteras_Inlet_to_Ocracoke_Inlet_1m_UTM18N_NAVD88_cog.tif",
# "https://github.com/dbuscombe-usgs/adm/releases/download/v0.0.1/clip_20181008_VA_to_Oregon_Inlet_1m_UTM18N_NAVD88_cog.tif",
"https://github.com/dbuscombe-usgs/adm/releases/download/v0.0.1/clip_20181008_Oregon_Inlet_to_Hatteras_Inlet_1m_UTM18N_NAVD88_cog.tif",
"https://github.com/dbuscombe-usgs/adm/releases/download/v0.0.1/clip_20181006_Oregon_Inlet_to_Hatteras_Inlet_1m_UTM18N_NAVD88_cog.tif",
"https://github.com/dbuscombe-usgs/adm/releases/download/v0.0.1/clip_20181007_New_River_Inlet_to_Oak_Island_1m_UTM18N_NAVD88_cog.tif"

]

files = [u.split('/')[-1] for u in urls]

for file, url in zip(files,urls):
    filename = os.path.join(os.getcwd(), file)
    tf.keras.utils.get_file(filename, url)
