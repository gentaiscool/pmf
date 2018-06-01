import numpy as np

data = np.load("data.npz")

items = data["item_id"]
users = data["user_id"]
ratings = data["rating"]

with open("train.txt", "w+") as train_file:
    with open("test.txt", "w+") as test_file:
        
        num_train = int(0.8 * len(items))
        num_test = int(0.2 * len(items))

        for i in range(len(items)):
            if i < num_train:
                train_file.write(str(users[i][0]) + " " + str(items[i][0]) + " " + str(ratings[i][0]) + "\n")
            else:
                test_file.write(str(users[i - num_train][0]) + " " + str(items[i - num_train][0]) + " " + str(ratings[i - num_train][0]) + "\n")