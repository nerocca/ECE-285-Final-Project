import re

FASTA_FILE = "data/chr22.fa"
OUTPUT_FILE = "data/dna_train.txt"

SEQ_LEN = 1024
STRIDE = 256


def read_fasta(file):

    seq = []

    with open(file) as f:
        for line in f:
            if not line.startswith(">"):
                seq.append(line.strip().upper())

    genome = "".join(seq)

    genome = re.sub("[^ATCGN]", "", genome)

    return genome


def generate_sequences(genome):

    samples = []

    for i in range(0, len(genome) - SEQ_LEN, STRIDE):

        seq = genome[i:i+SEQ_LEN]

        if "N" not in seq:
            samples.append(seq)

    return samples


def save_samples(samples):

    with open(OUTPUT_FILE, "w") as f:
        for s in samples:
            f.write(s + "\n")


if __name__ == "__main__":

    genome = read_fasta(FASTA_FILE)

    samples = generate_sequences(genome)

    save_samples(samples)

    print("Genome length:", len(genome))
    print("Training samples:", len(samples))
    print("Saved to:", OUTPUT_FILE)