#!/bin/bash
source activate torch-cuda
PYTHONPATH="." python tools/start_prompt_server.py