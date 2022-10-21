

import os
from glob import glob
import subprocess
import shlex

for ff in glob("../Movies/*"):
    dd,ext=os.path.split(ff)
    mv_nm=glob(f"{ff}/*mp4")[0]
    proc=f"ffmpeg -i {mv_nm} -r 30 ../Images/{ext}/img-%04d.png"
    print(shlex.split(proc))
    subprocess.run(shlex.split(proc))
