import torch
from torch.utils.data import DataLoader
from dna_gpt.model.dna_gpt import DNAGPT
from dataset import DNADataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import os

BATCH_SIZE = 8
EPOCHS = 4
LR = 3e-5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

writer = SummaryWriter(log_dir="runs/dnargpt_finetune")

print("loading checkpoint...")
vocab_size = 7
model_name = "dna_gpt0.1b_h"
model = DNAGPT.from_name(model_name, vocab_size=vocab_size)


checkpoint_path = "checkpoints/dna_gpt0.1b_h.pth"
state_dict = torch.load(checkpoint_path, map_location=DEVICE)

if "model" in state_dict:
    state_dict = state_dict["model"]


for k in list(state_dict.keys()):
    if "wte" in k or "mlm_head" in k:
        del state_dict[k]

model.load_state_dict(state_dict, strict=False) 
model = model.to(DEVICE)
model.train() 

train_dataset = DNADataset("data/train.txt")
val_dataset = DNADataset("data/val.txt")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE
)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

criterion = torch.nn.CrossEntropyLoss()

global_step = 0

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)
    
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        global_step += 1

        # TensorBoard: batch loss
        writer.add_scalar("Loss/train_batch", loss.item(), global_step)
        loop.set_postfix(batch_loss=loss.item())

    avg_train_loss = epoch_loss / len(train_loader)
    writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)

    # -----------------------------
    # 验证集评估
    # -----------------------------
    model.eval()
    val_loss_total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            val_loss_total += loss.item()

    avg_val_loss = val_loss_total / len(val_loader)
    writer.add_scalar("Loss/val_epoch", avg_val_loss, epoch)
    val_ppl = torch.exp(torch.tensor(avg_val_loss))
    writer.add_scalar("Perplexity/val", val_ppl, epoch)

    print(f"Epoch {epoch+1} train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} val_ppl={val_ppl:.2f}")

    # -----------------------------
    # 自动保存微调权重（每个 epoch）
    # -----------------------------
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"dnargpt_finetuned_epoch{epoch+1}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Checkpoint saved to {save_path}")

torch.save(model.state_dict(), "checkpoints/dnargpt_finetune.pth")
print("Finetuned model saved to checkpoints/dnargpt_finetune.pth")
# ----------------------
# 结束训练
# ----------------------
writer.close()
print("Training complete! Launch TensorBoard to visualize.")
