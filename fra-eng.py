import numpy as np

f = open("./assets/fra.txt", "r")


lines = f.readlines()

line = f.readline()
lineContent = line.split("\t")

encoder_input_data = np.ndarray(shape=(len(lines), ))

print(lineContent)