from dataset import ReviewDataset
from model import ReviewGenerator
from process_data import load_and_clean_data
import torch
from torch.utils.data import DataLoader
from model import GPTConfig
from tokenizer import Tokenizer
from sklearn.model_selection import train_test_split
import os, glob, time

def train_loop(data, model, optimizer):
    model.train()
    train_loss = 0
    for batch in data:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        preds, loss = model(input_ids, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()
    print(f"  Train Loss: {train_loss / len(data):.4f}")

def test_loop(data, model):
    model.eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch in data:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            preds, loss = model(input_ids, labels)
            test_loss += loss.item()
            mask = labels != -100
            pred_ids = preds.argmax(-1)
            correct += (pred_ids[mask] == labels[mask]).sum().item()
            total += mask.sum().item()
    avg_loss = test_loss / len(data)
    print(f"  Test Loss:   {avg_loss:.4f}")
    print(f"  Test Accuracy: {correct / total:.4f}")
    return avg_loss

if __name__ == "__main__":
    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    csv_paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    titles, ratings, reviews = load_and_clean_data(csv_paths)
    train_titles, val_titles, train_ratings, val_ratings, train_reviews, val_reviews = train_test_split(
        titles, ratings, reviews, test_size=0.05, random_state=67
    )
    tokenizer = Tokenizer()
    tokenizer.load(os.path.join(data_dir, "tokenizer.model"))
    train_dataset = ReviewDataset(train_titles, train_ratings, train_reviews, tokenizer)
    val_dataset = ReviewDataset(val_titles, val_ratings, val_reviews, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    config = GPTConfig(vocab_size=8013, n_embeddings=128, n_heads=4, n_layers=4, block_size=512, dropout=0.2)

    model = ReviewGenerator(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    best_test_loss = float('inf')
    patience, wait = 3, 0
    start_epoch = 0

    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'ckpt_best.pth')

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_test_loss = checkpoint['best_test_loss']
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from checkpoint at epoch {start_epoch}, best test loss {best_test_loss:.4f}")

    epochs = 8
    run_start = time.time()
    for t in range(start_epoch, start_epoch + epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        epoch_start = time.time()
        train_loop(train_loader, model, optimizer)
        test_loss = test_loop(test_loader, model)
        epoch_time = time.time() - epoch_start
        total_time = time.time() - run_start
        print(f"  Epoch time: {epoch_time/60:.1f} min | Total elapsed: {total_time/60:.1f} min")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save({
                'epoch': t,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_test_loss': best_test_loss
            }, os.path.join(checkpoint_dir, 'ckpt_best.pth'))

            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {t+1}")
                break
