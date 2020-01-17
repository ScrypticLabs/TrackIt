from env import *
import time

if __name__ == "__main__":
    configs = []
    with open("configs.txt", "r") as f:
        for line in f.readlines():
            line = line.strip()
            config = "config/"+line[0]+"/"+line
            configs.append(config)
            
    for config in configs:
        env = Environment()
        env.load(file=config)
        env.start()
        time.sleep(1)
        print("collected %s ..." % (config))