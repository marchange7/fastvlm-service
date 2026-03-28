#!/bin/bash
# Download FastVLM models from HuggingFace
pip install huggingface-hub
python -c "
import os
from huggingface_hub import snapshot_download
print('Downloading FastVLM-0.5B...')
snapshot_download('apple/FastVLM-0.5B-fp16', local_dir=os.path.expanduser('~/models/fastvlm-0.5b'))
print('Downloading FastVLM-1.5B...')
snapshot_download('apple/FastVLM-1.5B-int8', local_dir=os.path.expanduser('~/models/fastvlm-1.5b'))
print('Done.')
"
