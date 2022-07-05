import os


def ensure_dir(targetdir):
    if not os.path.isdir(targetdir):
        try:
            os.makedirs(targetdir)
        except:
            pass