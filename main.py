import csv
import os
import time

from tool_kit.colors import bcolors
from Test_Module.Test import Test_Module
from DB.schema_definition import DB
from configuration.configuration import getConfig

modules_dict = {}
modules_dict["Test_Module"] = Test_Module
modules_dict["DB"] = DB

window_start = getConfig().eval("DEFAULT", "start_date")
newbmrk = os.path.isfile("benchmark.csv")
bmrk_file = open("benchmark.csv", 'a', newline='')
bmrk_results = csv.DictWriter(bmrk_file,
                                ["time", "jobnumber", "config", "window_size", "window_start", "dones", "posts",
                                "authors"] + list(modules_dict.keys()),
                                dialect="excel", lineterminator="\n")
if not newbmrk:
    bmrk_results.writeheader()

db = DB()
modules_dict["DB"] = lambda x: x
pipeline = []
for module in getConfig().sections():
    parameters = {}
    if modules_dict.get(module):
        pipeline.append(modules_dict.get(module)(db))

bmrk = {"config": getConfig().getfilename(), "window_start": "setup"}
for module in pipeline:
    T = time.perf_counter()
    module.setUp()
    T = time.perf_counter() - T
    bmrk[module.__class__.__name__] = T

bmrk_results.writerow(bmrk)
bmrk_file.flush()

bmrk = {"config": getConfig().getfilename(), "window_start": "execute"}
for module in pipeline:
    T = time.time()
    print(bcolors.YELLOW+'Started executing ' + module.__class__.__name__+bcolors.ENDC)

    module.execute(window_start)

    print(bcolors.YELLOW+'Finished executing ' + module.__class__.__name__+bcolors.ENDC)
    T = time.time() - T
    bmrk[module.__class__.__name__] = T

bmrk_results.writerow(bmrk)
bmrk_file.flush()
