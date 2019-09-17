import os

if __name__ == '__main__':
    base_path = r"E:\Users\root\Projects\VideoSynchronizationWithPytorch\trainings\hyperparameter_search"
    best_score = 0
    best_path = None
    for path, subfolders, files in os.walk(base_path):
        if "result.txt" in files:
            f = open(path + r"\result.txt", 'r')
            res = f.readline()
            f.close()
            res = float(res)
            if res > best_score:
                best_score = res
                best_path = path

    if best_path is not None:
        f = open(best_path + r"\config.json", 'r')
        config = f.readline()
        f.close()
        print(f"Best result of {best_score} for training {best_path}")
        print("Config:", config)
