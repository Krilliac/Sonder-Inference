import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # bench/http -> import httpbench, bench_http
sys.path.insert(0, HERE)                   # fake_server
