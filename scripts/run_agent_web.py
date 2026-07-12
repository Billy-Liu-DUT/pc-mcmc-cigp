from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pc_mcmc_cigp.agent_backend.http_api import serve


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--host",default="127.0.0.1"); parser.add_argument("--port",type=int,default=8765); parser.add_argument("--projects",default=str(ROOT/"projects"))
    args=parser.parse_args(); serve(args.host,args.port,args.projects,ROOT/"web")
