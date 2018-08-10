"""
Process top-level SNOMED code
"""

"""
Takes in raw data and preprocess this into good format
for TorchText to train on
"""

import re
import numpy as np

"""
We train without diagnosis, and with multilabel
"""

np.random.seed(1234)

# ======== Split =========
train_size = 0.9

assert (train_size < 1 and train_size > 0)
split_proportions = {
    "train": train_size,
    "valid": (1 - train_size) / 2,
    "test": (1 - train_size) / 2
}
assert (sum([split_proportions[split] for split in split_proportions]) == 1)

print("the data split is: {}".format(split_proportions))

inflating_test_set = True

# maybe not predicting 17 (it's a catch-all disease)

def write_to_tsv(data, file_name, label_list):
    # we are translating labels here
    with open(file_name, 'wb') as f:
        for line in data:
            mapped_labels = [str(label_list.index(l)) for l in line[1].split()]
            f.write(line[0] + '\t' + " ".join(mapped_labels) + '\n')


def count_freq(list_labels):
    dic = {}
    for l in list_labels:
        if l not in dic:
            dic[l] = 1
        else:
            dic[l] += 1
    return dic


def get_most_freq_label(dic):
    most_f_l = None
    most_f_f = 0.
    for l, f in dic.iteritems():
        if f > most_f_f:
            most_f_l = l
    return most_f_l

def collapse_label(labels):
    # Note: for SNOMED we no longer take out category 17 (no longer exist)
    labels = labels.strip()
    # labels = labels.replace('17', '')
    list_labels = filter(lambda l: len(l) > 0, labels.split('-'))
    # if len(list_labels) == 0:
    #     list_labels = ['17']  # meaning it only has 17
    set_labels = set(list_labels)  # remove redundancies
    return list(set_labels)

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    cleantext = re.sub(r'^https?:\/\/.*[\r\n]*', '', cleantext, flags=re.MULTILINE)
    return cleantext

# TODO: 2. Preserve things like "Texas A&M", the ampersand in the middle
def preprocess_text(text, no_description):
    no_html = cleanhtml(text)
    one_white_space = ' '.join(no_html.split())
    no_html_entities = re.sub('&[a-z]+;', '', one_white_space)

    if no_description:
        # delete both diagnosis and discharge status
        no_html_entities = no_html_entities.split('Diagnosis:')[0]

    return no_html_entities


if __name__ == '__main__':
    header = True

    examples = []
    labels_dist = []
    with open("../../data/csu/final_csu_file_snomed", 'r') as f:
        for line in f:
            if header:
                header = False
                continue
            columns = line.split('\t')
            labels = columns[-1]

            text = preprocess_text(columns[4], no_description=True)

            seq_labels = collapse_label(labels)
            labels_dist.extend(seq_labels)
            # start from 0, and also join back to " " separation
            examples.append([text, " ".join(seq_labels)])

    # import matplotlib.pyplot as plt
    #
    # n, bins, patches = plt.hist(labels_dist, 50, normed=1, facecolor='green', alpha=0.75)
    # plt.show()

    import csv
    with open("../../data/csu/Files_for_parsing/snomed_ICD_mapped.csv", 'r') as f:
        csv_reader = csv.reader(f, delimiter=';')
        snomed_code_to_name = {}
        for row in csv_reader:
            snomed_code_to_name[row[0]] = row[1]

    labels_dist = count_freq(labels_dist)

    print("number of labels is {}".format(len(labels_dist)))

    with open("../../data/csu/snomed_dist.csv", 'wb') as f:
        for k, v in labels_dist.items():
            f.write(snomed_code_to_name[k] + "," + str(v) + "\n")

    labels_prob = map(lambda t: (t[0], float(t[1]) / sum(labels_dist.values())), labels_dist.items())

    labels_prob = sorted(labels_prob, key=lambda t: t[1])

    print "code, n, p"
    for k, prob in labels_prob:
        print "{}, {}, {}".format(k, labels_dist[k], prob)

    label_list = [t[0] for t in labels_prob]

    # process them into tsv format, but also collect frequency distribution
    serial_numbers = range(len(examples))
    np.random.shuffle(serial_numbers)

    train_numbers = serial_numbers[:int(np.rint(len(examples) * split_proportions['train']))]
    valid_numbers = serial_numbers[
                    int(np.rint(len(examples) * split_proportions['train'])): \
                        int(np.rint(len(examples) * (split_proportions['train'] + split_proportions['valid'])))]
    test_numbers = serial_numbers[
                   int(np.rint(len(examples) * (split_proportions['train'] + split_proportions['valid']))):]

    print(
        "train/valid/test number of examples: {}/{}/{}".format(len(train_numbers), len(valid_numbers),
                                                               len(test_numbers)))
    train, valid, test = [], [], []

    for tn in train_numbers:
        train.append(examples[tn])
    for tn in valid_numbers:
        valid.append(examples[tn])
    for tn in test_numbers:
        test.append(examples[tn])

    write_to_tsv(train, "../../data/csu/snomed_multi_label_no_des_train.tsv", label_list)
    write_to_tsv(valid, "../../data/csu/snomed_multi_label_no_des_valid.tsv", label_list)
    write_to_tsv(test, "../../data/csu/snomed_multi_label_no_des_test.tsv", label_list)

    import json
    with open('../../data/csu/snomed_labels.json', 'wb') as f:
        json.dump(label_list, f)

    names = [snomed_code_to_name[l] for l in label_list]
    # index matches 0 to 41
    with open('../../data/csu/snomed_labels_to_name.json', 'wb') as f:
        json.dump(names, f)
