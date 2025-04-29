# To check project requirments

import importlib.util

def check_installed(pkg):
    return importlib.util.find_spec(pkg) is not None

packages = ["numpy", "pandas", "matplotlib", "scipy", "seaborn", "torch", "tqdm", "mne", "sklearn"]
for pkg in packages:
    print(f"{pkg}: {'✓ Installed' if check_installed(pkg) else '✗ Not Installed'}")