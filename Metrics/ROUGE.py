import rouge
import csv
import numpy as np

DATA_FILE = '../Datasets/GIAFC_rewrites_Random.csv'

pairs = []
with open(DATA_FILE, 'r', newline='') as input_csv:
    reader = csv.DictReader(input_csv)
    for row in reader:
        first_sentence = row['text1'].strip()
        second_sentence = row['text2'].strip()
        pairs.append((first_sentence, second_sentence))


evaluator = rouge.Rouge(metrics=['rouge-1', 'rouge-2', 'rouge-l'])

scores_rouge1 = []
scores_rouge2 = []
scores_rougel = []
for first_sentence, second_sentence in pairs:
    scores = evaluator.get_scores([first_sentence], [second_sentence])
    scores_rouge1.append(scores[0]['rouge-1']['f'])
    scores_rouge2.append(scores[0]['rouge-2']['f'])
    scores_rougel.append(scores[0]['rouge-l']['f'])

print(np.mean(scores_rouge1))
print(np.std(scores_rouge1))

print(np.mean(scores_rouge2))
print(np.std(scores_rouge2))

print(np.mean(scores_rougel))
print(np.std(scores_rougel))

# summary = 'the cat was found under the bed'
# reference = 'the cat was under the bed'
#
# score = evaluator.get_scores([summary], [reference])
#
# print(score)

