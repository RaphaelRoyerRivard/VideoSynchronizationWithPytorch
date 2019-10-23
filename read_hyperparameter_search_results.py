import os
import numpy as np

if __name__ == '__main__':
    base_path = r"E:\Users\root\Projects\VideoSynchronizationWithPytorch\trainings\hyperparameter_search"
    failed_xp = []
    sorted_xp = [[], [], []]
    best_scores = [0, 0, 0]
    best_paths = [None, None, None]
    for path, subfolders, files in os.walk(base_path):
        if "config.json" in files:
            f = open(path + r"\config.json", 'r')
            config = f.readline()
            f.close()
            xp_id = path.split('\\')[-1]
            if "result.txt" in files:
                f = open(path + r"\result.txt", 'r')
                lines = f.readlines()
                f.close()
                for matrix_type, res in enumerate(lines):
                    res = float(res)
                    sorted_xp[matrix_type].append((res, xp_id, config))
                    if res > best_scores[matrix_type]:
                        best_scores[matrix_type] = res
                        best_paths[matrix_type] = path
            else:
                failed_xp.append((xp_id, config))

    print("Failed XP")
    for xp in failed_xp:
        print(xp)

    for i, matrix_type in enumerate(["Distance", "Similarity", "Combination"]):
        if len(sorted_xp[i]) > 0:
            sorted_xp[i].sort(key=lambda tup: tup[0], reverse=True)
            print(f"Sorted XP for {matrix_type} matrices. AVG: {np.array([x[0] for x in sorted_xp[i]]).mean()}")
            for xp in sorted_xp[i]:
                print(xp)
