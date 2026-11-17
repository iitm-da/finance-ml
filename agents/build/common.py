"""Shared by the agents deck: recorded runs, numbers measured in the LLM session, colours, small helpers."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'build'))
from fx import *

TRC = json.load(open(os.path.join(HERE, 'traces.json')))                 # five runs of the loop, recorded from the model
LM = json.load(open(os.path.join(HERE, '..', '..', 'llm', 'build', 'm_llm.json')))
ON, INJ = LM['rag']['onto'], LM['prompt']['inj']
JSB = '<span class="jsbadge">browser JS</span>'
