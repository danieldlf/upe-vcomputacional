import torch
import os
from torch.utils.data import DataLoader, random_split
from src.data import DrivingDataset
from src.vlm import MultimodalPolicy
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

CSV_PATH = "dataset_big_highway/dataset_highway_200k.csv"
ROOT_DIR = "dataset_big_highway"
MODEL_NAME = "Qwen/Qwen3-0.6B"
CLIP_DIM = 512

EPOCHS = 30
BATCH_SIZE = 32
LR = 2e-4
CHECKPOINT_DIR = "checkpoints_v3"
BEST_MODEL_PATH = "vlm_v3.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def main():
    print("Carregando dataset...")
    full_dataset = DrivingDataset(CSV_PATH, root_dir=ROOT_DIR)
    
    val_size = int(0.20 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Treino: {len(train_dataset)} | Validação: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = MultimodalPolicy(model_name=MODEL_NAME, clip_dim=CLIP_DIM, action_size=5).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    
    loss_fn = torch.nn.CrossEntropyLoss() 

    scheduler = OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS)
    scaler = torch.amp.GradScaler('cuda') # type: ignore

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in loop:
            clip_feat = batch["clip_feat"].to(device, non_blocking=True)
            action = batch["action"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(clip_feat=clip_feat)
                loss = loss_fn(logits, action)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                clip_feat = batch["clip_feat"].to(device, non_blocking=True)
                action = batch["action"].to(device, non_blocking=True)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(clip_feat=clip_feat)
                    loss = loss_fn(logits, action)
                
                val_loss += loss.item()
                
                _, predicted = torch.max(logits, 1)
                total += action.size(0)
                correct += (predicted == action).sum().item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        print(f"--- Fim Epoca {epoch+1} | Loss Train: {avg_train:.4f} | Loss Val: {avg_val:.4f} | Acc: {accuracy:.2f}% ---")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"🏆 Melhor modelo salvo! (Acc: {accuracy:.2f}%)")

    print("Fim do treino.")

if __name__ == "__main__":
    main()