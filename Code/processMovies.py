

import os
from glob import glob

for ff in glob("../Movies/*"):
    print(ff)
    # ffmpeg -i Movies/ -r 30 Images/img-%04d.png
