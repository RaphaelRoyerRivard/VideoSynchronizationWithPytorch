import os

if __name__ == '__main__':
    base_path = r"E:\Users\root\Projects\VideoSynchronizationWithPytorch\trainings\hyperparameter_search"
    sorted_xp = []
    best_score = 0
    best_path = None
    for path, subfolders, files in os.walk(base_path):
        if "result.txt" in files:
            f = open(path + r"\result.txt", 'r')
            res = f.readline()
            f.close()
            f = open(path + r"\config.json", 'r')
            config = f.readline()
            f.close()
            res = float(res)
            sorted_xp.append((res, path.split('\\')[-1], config))
            if res > best_score:
                best_score = res
                best_path = path

    sorted_xp.sort(key=lambda tup: tup[0], reverse=True)
    for xp in sorted_xp:
        print(xp)
