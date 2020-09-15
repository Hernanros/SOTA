import csv
import numpy as np

DATA_FILE = '../MurgedHumanLabeledSets/GYAFC_rewrites_random_human_murged.csv'

avg_labels = []
with open(DATA_FILE, 'r', newline='') as input_csv:
    reader = csv.reader(input_csv)
    isFirst = True
    for row in reader:
        if isFirst:
            isFirst = False
            continue
        first_label = int(row[2])
        second_label = int(row[3])
        third_label = int(row[4])
        avg_labels.append((first_label + second_label + third_label) / 3)


print(np.mean(avg_labels))
print(np.std(avg_labels))