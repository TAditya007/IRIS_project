============================================================
setup/ -- iRIS Replication Bootstrap
============================================================

This folder is self-contained. Anyone who clones the repo can run a
single .bat file and end up with the full project skeleton + a
verified Python environment.

------------------------------------------------------------
CONTENTS
------------------------------------------------------------

File                      Purpose
------------------------  ---------------------------------------------
create_project.bat        Builds iris_replication/ tree (idempotent)
verify_setup.py           Python verifier (imports, CUDA, math, files)
verify_setup.ipynb        Notebook verifier (same checks, nicer UI)
environment.yml           Conda env spec (portable one-shot install)
requirements.txt          pip pins (fallback install path)
SETUP_GUIDE.txt           Full human-readable setup guide
README.txt                You are here

------------------------------------------------------------
QUICK START (3 STEPS)
------------------------------------------------------------

1. Build the project skeleton
       setup\create_project.bat

2. Create and activate the conda env
       conda env remove -n iris -y
       conda create -n iris python=3.10 -y
       conda activate iris
       pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 ^
           --index-url https://download.pytorch.org/whl/cu121
       pip install numpy==1.26.4 scipy==1.13.1 matplotlib==3.8.4 ^
           pandas==2.2.2 pyyaml==6.0.1 ruamel.yaml==0.18.6 ^
           tqdm==4.66.4 ipykernel==6.29.4 jupyter_client==8.6.2
       python -m ipykernel install --user --name=iris --display-name "Python (iris)"

3. Verify
       python setup\verify_setup.py

   Expected: "VERDICT: ALL GREEN -- READY FOR PHASE 0"

See SETUP_GUIDE.txt for the full version, troubleshooting table,
version-pin rationale, and the complete green-light checklist.

------------------------------------------------------------
REQUIREMENTS
------------------------------------------------------------

- Windows 10/11 (Linux/macOS work with minor tweaks)
- NVIDIA driver >= 525.60.13 (verify with: nvidia-smi)
- Miniconda or Anaconda
- ~8 GB free disk space
- VS Code + Python extension + Jupyter extension

------------------------------------------------------------
TROUBLESHOOTING (SHORT VERSION)
------------------------------------------------------------

CUDA not available             -> Update NVIDIA driver to >= 525
Kernel "iris" not listed       -> python -m ipykernel install --user
                                   --name=iris
Env > 8 GB                     -> conda env remove -n iris -y
                                   and reinstall (no CUDA toolkit)
Notebook "Kernel died"         -> Select "Python (iris)" as kernel
create_project.bat blocked     -> Move repo outside OneDrive
ModuleNotFoundError: yaml      -> pip install pyyaml==6.0.1
numpy 2.x conflict             -> pip install "numpy==1.26.4"
                                   --force-reinstall

Full troubleshooting table in SETUP_GUIDE.txt Section 7.

============================================================
END OF README
============================================================