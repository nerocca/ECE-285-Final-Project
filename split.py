import random

INPUT_FILE = "data/dna_train.txt"
TRAIN_FILE = "data/train.txt"
VAL_FILE = "data/val.txt"
TEST_FILE = "data/test.txt"

VAL_RATIO = 0.2
Test_RATIO = 0.1
with open(INPUT_FILE) as f:
    lines = f.readlines()

random.shuffle(lines)

split1 = int(len(lines) * (1 - VAL_RATIO - Test_RATIO))
split2 = int(len(lines) * (1 - Test_RATIO))

train_lines = lines[:split1]
val_lines = lines[split1:split2]
test_lines = lines[split2:]

with open(TRAIN_FILE, "w") as f:
    f.writelines(train_lines)

with open(VAL_FILE, "w") as f:
    f.writelines(val_lines)

with open(TEST_FILE, "w") as f:
    f.writelines(test_lines)

print("train:", len(train_lines))
print("val:", len(val_lines))
print("test:", len(test_lines))