import torch
from torch.utils.data import DataLoader
from dna_gpt.model.dna_gpt import DNAGPT
from dataset import DNADataset
from collections import Counter
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import io

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


writer = SummaryWriter(log_dir="runs/dnargpt_eval")

def base_frequency(sequences):
    bases = ["A","C","G","T"]
    counts = {b:0 for b in bases}

    for seq in sequences:
        for c in seq:
            if c in counts:
                counts[c] += 1

    total = sum(counts.values())
    freq = {b: counts[b]/total for b in bases}

    return freq

def sequence_accuracy(input_seq, pred_seq):
    correct = sum(a == b for a, b in zip(input_seq, pred_seq))
    return correct / len(input_seq)

def highlight_diff(input_seq, pred_seq):
    highlighted = ""
    for a, b in zip(input_seq, pred_seq):
        if a == b:
            highlighted += a
        else:
            highlighted += f"<span style='color:red'><b>{b}</b></span>"
    return highlighted

vocab_size = 7
model_name = "dna_gpt0.1b_h"
model = DNAGPT.from_name(model_name, vocab_size=vocab_size)

checkpoint_path = "checkpoints/dnargpt_finetune.pth"
state_dict = torch.load(checkpoint_path, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()


test_dataset = DNADataset("data/test.txt")
test_loader = DataLoader(test_dataset, batch_size=8)

criterion = torch.nn.CrossEntropyLoss()


total_loss = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()

avg_test_loss = total_loss / len(test_loader)
test_ppl = torch.exp(torch.tensor(avg_test_loss))

print(f"Test Loss: {avg_test_loss:.4f}, Test Perplexity: {test_ppl:.2f}")
writer.add_scalar("Test/Loss", avg_test_loss)
writer.add_scalar("Test/Perplexity", test_ppl)


token_to_base = {0:"A", 1:"C", 2:"G", 3:"T", 4:"N", 5:"R", 6:"S"}
num_samples = 5  

with torch.no_grad():
    for i, (x, y) in enumerate(test_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        pred_ids = torch.argmax(logits, dim=-1)

        real_sequences = []
        pred_sequences = []

        for j in range(min(num_samples, x.size(0))):
            input_seq = "".join([token_to_base[t.item()] for t in x[j]])
            pred_seq = "".join([token_to_base[t.item()] for t in pred_ids[j]])

            real_sequences.append(input_seq)
            pred_sequences.append(pred_seq)

            sample_id = i * test_loader.batch_size + j + 1

            highlighted_pred = highlight_diff(input_seq, pred_seq)

            text_log = f"""
<h3>Sample {sample_id}</h3>

<b>Input:</b><br>
<pre>{input_seq}</pre>

<b>Prediction (red = error):</b><br>
<pre>{highlighted_pred}</pre>
"""
            acc = sequence_accuracy(input_seq, pred_seq)
            writer.add_scalar("Prediction/SequenceAccuracy", acc, sample_id)

            print(f"Sample {sample_id}")
            print("Input :", input_seq[:60])
            print("Pred  :", pred_seq[:60], "\n")

            writer.add_text(
                tag="Sequence_Predictions",
                text_string=text_log,
                global_step=sample_id
            )


        num_samples -= x.size(0)
        if num_samples <=0:
            break


def gc_content(seq):
    gc_count = seq.count("G") + seq.count("C")
    total = seq.count("A") + seq.count("T") + gc_count
    return gc_count / total if total>0 else 0

gc_values = []
with torch.no_grad():
    for x, _ in test_loader:
        x = x.to(DEVICE)
        for seq in x:
            seq_str = "".join([token_to_base[t.item()] for t in seq])
            gc_values.append(gc_content(seq_str))

avg_gc = sum(gc_values)/len(gc_values)
print(f"Average GC content in test set: {avg_gc:.4f}")


plt.figure()
plt.hist(gc_values, bins=20, color='skyblue')
plt.title("GC Content Distribution")
plt.xlabel("GC Ratio")
plt.ylabel("Frequency")
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
image = torch.tensor(plt.imread(buf) * 255, dtype=torch.uint8).permute(2,0,1)
writer.add_image("GC_content_distribution", image, 0)
plt.close()

#K-mer
k = 3

real_kmer = Counter()
gen_kmer = Counter()

with torch.no_grad():
    for x, _ in test_loader:
        x = x.to(DEVICE)
        for seq in x:
            seq_str = "".join([token_to_base[t.item()] for t in seq])
            for i in range(len(seq_str) - k+1):
                real_kmer[seq_str[i:i+k]] += 1

        logits = model(x)
        pred_ids = torch.argmax(logits, dim=-1)

        for seq in pred_ids:
            seq_str = "".join([token_to_base[t.item()] for t in seq])
            for i in range(len(seq_str) - k+1):
                gen_kmer[seq_str[i:i+k]] += 1
top_real = real_kmer.most_common(20)
top_real_kmers = [k for k, _ in top_real]

top_gen = gen_kmer.most_common(20)
top_gen_kmers = [k for k, _ in top_gen]

real_counts = [real_kmer[k] for k in top_real_kmers]
gen_counts = [gen_kmer[k] for k in top_real_kmers]

print("Top 20 3-mers in real test set:")
for kmer, count in top_real:
    print(f"{kmer}: {count}")
print("Top 20 3-mers in generated set:")
for kmer, count in top_gen:
    print(f"{kmer}: {count}")



x = range(len(top_real_kmers))

plt.figure(figsize=(12,5))
plt.bar(x, real_counts, width=0.4, label="Real", align='center')
plt.bar([i+0.4 for i in x], gen_counts, width=0.4, label="Generated")

plt.xticks([i+0.2 for i in x], top_real_kmers, rotation=90)
plt.ylabel("Count")
plt.title("Real vs Generated 3-mer Distribution")
plt.legend()

plt.tight_layout()
plt.show()

#Bases frequency comparison
real_freq = base_frequency(real_sequences)
pred_freq = base_frequency(pred_sequences)

bases = ["A","C","G","T"]

real_vals = [real_freq[b] for b in bases]
pred_vals = [pred_freq[b] for b in bases]

plt.figure(figsize=(6,4))
x = range(len(bases))

plt.bar(x, real_vals, width=0.4, label="Real", align="edge")
plt.bar(x, pred_vals, width=-0.4, label="Generated", align="edge")

plt.xticks(x, bases)
plt.ylabel("Frequency")
plt.title("Base Frequency Comparison")
plt.legend()

buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)

image = torch.tensor(plt.imread(buf)*255, dtype=torch.uint8).permute(2,0,1)
writer.add_image("DNA/Base_Frequency", image, 0)

plt.close()

#GC conetent comparison
real_gc = [gc_content(s) for s in real_sequences]
pred_gc = [gc_content(s) for s in pred_sequences]

plt.figure(figsize=(6,4))

plt.hist(real_gc, bins=20, alpha=0.6, label="Real")
plt.hist(pred_gc, bins=20, alpha=0.6, label="Generated")

plt.xlabel("GC Content")
plt.ylabel("Frequency")
plt.title("GC Content Distribution")
plt.legend()

buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)

image = torch.tensor(plt.imread(buf)*255, dtype=torch.uint8).permute(2,0,1)
writer.add_image("DNA/GC_Content_Comparison", image, 0)

plt.close()



writer.close()
print("Evaluation complete! Launch TensorBoard to visualize.")