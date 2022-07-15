conda create -p .venv python=3.7 -y
conda activate .venv/
pip install -r requirements.txt
conda env export > conda.yaml