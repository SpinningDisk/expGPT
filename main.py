from tkinter import *
from tkinter.ttk import *
import customtkinter as ctk
import re
import sys

import keras

# global master_window
global model
global rlAgentEnviron
global rendering

#class arena():
#    def __init__(self, environment: class):
#        self.env = environment


def load_model(function_args):
    model = None
    print(function_args)
    if len(function_args) < 1:
        print(f"{'\033[1;31m'}Error: Missing arguments at function {'\033[1;32m'}\"load_model()\"{'\033[1;31m'}: need at least {'\033[1;30m'}1{'\033[1;31m'} argument (model_path: str){'\033[0m'}")
    else:
        model_name = function_args[0]
        load_fast = True
        if len(function_args) > 1:
            try:
                assert function_args[1] in ["True", "False"]
                load_fast = [True if function_args[1] == "True" else False][0]
            except:
                try:
                    load_fast = int(function_args[1])
                except:
                    print(f"{'\033[1;31m'}Error: Argument 2 is not of {'\033[1;30m'}[int, bool] in {'\033[1;32m'}\"load_model()\"{'\033[1;31m'}{'\033[0m'}")
        print(["loading compiled model" if load_fast else "loading model from src"])
        if load_fast:
            try:
                model = keras.models.load_model(f"/mnt/data/dev/ML/models/.compiled/{function_args[0]}.keras")
            except:
                print(f"{'\033[1;31m'}Error: could not load {'\033[1;30m'}{function_args[0]}.keras{'\033[1;32m'}")
        else:
            import importlib
            import os
            import pathlib
            import inspect
            folder_path = pathlib.Path("models/src")
            modules = {}
            for file in folder_path.glob("*.py"):
                if file.name == "__init__.py":
                    continue
                module_name = f"models.src.{file.stem}"
                modules[file.stem] = importlib.import_module(module_name)
            for name, module in modules.items():
                if name == function_args[0].split("/")[0]:
                    if len(function_args[0].split("/")) == 3:
                        classes = inspect.getmembers(module, inspect.isclass)
                        model_class = None
                        for cname, cls in classes:
                            if cls.__module__ == module.__name__:
                                if cname == function_args[0].split("/")[1]:
                                    model_class = cls()
                        if model_class == None:
                            print(f"{'\033[1;31m'}Error: Could not find class {'\033[1;32m'}{function_args[0].split("/")[1]}{'\033[1;31m'} within module {'\033[1;31m'}{name}{'\033[0m'}")
                        else:
                            func = None
                            
                            for fname, fun in inspect.getmembers(model_class, inspect.ismethod):
                                if fname == function_args[0].split("/")[2]:
                                    model = fun()
                            if model == None:
                                print(f"{'\033[1;31m'}Error: Could not find function {'\033[1;32m'}{function_args[0].split("/")[2]}{'\033[1;31m'} within module {'\033[1;31m'}{name}{'\033[0m'}")
                    else:
                        for fname, fun in inspect.getmembers(module, inspect.isfunction):
                            if fun.__module__ == module.__name__:
                                if fname == function_args[0].split("/")[1]:
                                    model = fun()
            if model == None:
                print(f"{'\033[1;31'}Error: Could not fetch model{'\033[0m'}")
                return None
    return model
def load_env(function_args):
    environment = None
    print(function_args)
    if len(function_args) < 1:
        print(f"{'\033[1;31m'}Error: Missing arguments at function {'\033[1;32m'}\"load_environment()\"{'\033[1;31m'}: need at least {'\033[1;30m'}1{'\033[1;31m'} argument (environment_path: str){'\033[0m'}")
    else:
        environment_name = function_args[0]
        import importlib
        import os
        import pathlib
        import inspect
        environ_folder = function_args[0].split("/")[0]
        folder_path = pathlib.Path("environments/"+environ_folder)
        modules = {}
        for file in folder_path.glob("*.py"):
            print(file.name)
            if file.name == "__init__.py":
                continue
            module_name = f"environments.{function_args[0].split("/")[0]}.{file.stem}"
            modules[file.stem] = importlib.import_module(module_name)
        for name, module in modules.items():
            if name == function_args[0].split("/")[1]:
                if len(function_args[0].split("/")) == 4:
                    classes = inspect.getmembers(module, inspect.isclass)
                    model_class = None
                    for cname, cls in classes:
                        if cls.__module__ == module.__name__:
                            if cname == function_args[0].split("/")[2]:
                                environment_parent_class = cls()
                    if environment_parent_class == None:
                        print(f"{'\033[1;31m'}Error: Could not find class {'\033[1;32m'}{function_args[0].split("/")[2]}{'\033[1;31m'} within module {'\033[1;31m'}{name}{'\033[0m'}")
                    else:
                        environment_class = None
                        
                        for cname, cls in inspect.getmembers(environment_class, inspect.isclass):
                            if cname == function_args[0].split("/")[3]:
                                environment = cls()
                        if model == None:
                            print(f"{'\033[1;31m'}Error: Could not find sub-class {'\033[1;32m'}{function_args[0].split("/")[3]}{'\033[1;31m'} within module {'\033[1;31m'}{name}{'\033[0m'}")
                else:
                    for cname, cls in inspect.getmembers(module, inspect.isclass):
                        if cls.__module__ == module.__name__:
                            if cname == function_args[0].split("/")[2]:
                                environment = cls()
        if environment == None:
            print(f"{'\033[1;31'}Error: Could not fetch environment{'\033[0m'}")
            return None
    environment.reset()
    return environment
def adjust_model(function_args, model, rlAgentEnviron):
    # script to add/modify first layer of agent to make it compatible with output of _make_frame() of environment
    # addin' layer: function_args[0] needs to be 1/True
    #           -> fucntion_args[1] in ["conv", "max_pool", "avg_pool", "Dense", "embedding", "selective_transform"]
    #              "embedding"+"selective_transform" only works if target dim is greater than environemt output
    #              "max_pool"+"avg_pool" only works if target dim is smaller than environment output
    # changin' layer: function_args[0] either not given or 0/False
    #           -> always modifies model layers.Input() to environment output (or if I get to doing it training data shape)
    #              and then every subsequent layer following it to check if layer is compatible with the given shape
    #              In case it is not compatible we throw a warning
    match(len(function_args)):
        case 0:
            addedLayerName = None
        case default:
            addedLayerName = function_args[1] if ((function_args[0] in [str(x) for x in range(2)] and int(function_args[0])==1) or (function_args[0] in ["True", "False"] and function_args[0]=="True")) else None
    assert addedLayerName in ["conv", "max_pool", "avg_pool", "Dense", "embedding", "selective_transformation", None]
    if addedLayerName == None:
        firstLayer = model.layers[0]
        print(firstLayer.name)
def enable_render():
    global rendering
    global rlAgentEnviron
    rendering = True
    rlAgentEnviron.render()
def disable_render():
    global rendering
    global rlAgentEnviron
    rendering = True
    rlAgentEnviron.close()
def env_step(function_args):
    global model
    global rlAgentEnviron
    global rendering
    import numpy as np
    steps = int(function_args[0]) if (function_args[0].isnumeric()) else 1
    for i in range(steps):
        action = model.predict(rlAgentEnviron._make_frame())
        if rlAgentEnviron.step(action=action)[2] == True:
            rlAgentEnviron.reset()
        if rendering:
            rlAgentEnviron.render()
    return None



##  AIlang
def cmd(inputs):
    headless = True
    state = 0 # state machine
    pattern = re.compile(r'(\w+)\s*\((.*?)\)\s*')
#    while headless:
    stream =     function = pattern.findall(inputs)
    if stream in ["help", "info", "man", "manual"]:
        print(f"{'\033[0m'}functions:\n\t{'\033[1;32m'}load_model({'\033[1;35m'}[modelName: str]{'\033[1;32m'}, {'\033[1;35m'}[is_compiled: bool/int]{'\033[1;32m'})\n")
        return True
    function_name = function[0][0]
    function_args = function[0][1].split(", ")
    global model
    global rlAgentEnviron
    global learningSchedule
    if state == 0:
        if function_name == "exit":
            exit(function_args[0] or 0)
        if function_name == "help":
            command = function_args[0]
            match command:
                case "\"exit\"":
                    print(f"ussage: {"\033[1;32m"}exit(<int>[exit code]){"\033[0m"}\n\texits with code provided\n\tdefaults to exit code 0")
                case "\"help\"":
                    print(f"ussage: {"\033[1;32m"}help(<str>[command]){"\033[0m"}\n\tprovides help for the given command\n\tif no command is provided, a short description of each command is provided")
                case "":
                    print(f"commands: {"\033[1;30m"}exit; help; open_gui; load_model; load_env/load_environment; load_sched/load_schedule; list_models; list_envs; adjust_model; adjust_env; reload; env_step; infere; render; unrender{"\033[0m"}")
                case default:
                    print(f"NJI")
        elif function_name == "open_gui":
            return False
        elif function_name == "load_model":
            model = load_model(function_args)
        elif function_name in ["load_env", "load_environment"]:
            rlAgentEnviron = load_env(function_args)
        elif function_name in ["load_sched", "load_schedule"]:
            learningSchedule = load_sched(function_args)
        elif function_name == "list_models":            # TODO
            list_models(function_args)
        elif function_name == "list_envs":              # TODO
            list_envs(function_args)
        elif function_name == "adjust_model":           # TODO
            model = adjust_model(function_args, model, rlAgentEnviron)
        elif function_name == "adjust_env":
            rlAgentEnviron = adjust_env(function_args, model, rlAgentEnviron)
        elif function_name == "reload":                 # TODO
            reload_modules()
        elif function_name == "env_step":
            env_step(function_args)
        elif function_name == "infere":
            # global model
            # global rlAgentEnviron
            output = model(rlAgentEnviron.state)
        elif function_name == "render":
            enable_render()
        elif function_name == "unrender":
            disable_render()
    return True
def gui():
    print("running GUI!")
    headless = False
    master_window = Tk()
    master_window.geometry("300x200")
    master_window.title("Lab")
    Label(master_window, text="This is the main window").pack(pady=10)
    def decapitate(headless=headless):
        master_window.destroy()
    Button(master_window, text="turn headless", command=decapitate).pack(pady=10)
    master_window.mainloop()
    return 1


if len(sys.argv)>1:
    file = open(sys.argv[1], "r").read()

    for line in file.split("\n"):
        if len(line)!=0:
            print(f"{'\033[0;33m'}>> {'\033[0m'}{line}")
            cmd(line)
        else:
            continue
    cmd("exit()")
headless = True

while True:
    if headless:
        inputs = input(f"{'\033[0;33m'}>> {'\033[0m'}")
        headless = cmd(inputs)
    else:
        headless = gui()
