import subprocess, os

def executeJupyter():
    #env_dir = "../main_env_dir/"
    #os.chdir(env_dir)

    # source jupyter_env/bin/activate
    env_activate = "jupyter_env/bin/activate_this.py"

    activate_env = exec(open(env_activate).read(), {'__file__': env_activate})

    # Open jupyter notebook as a subprocess
    openJupyter = "jupyter notebook"
    subprocess.Popen(openJupyter, shell=True)

#executeJupyter()


os.system("jupyter notebook ../notebooks/VisualWidget.ipynb")