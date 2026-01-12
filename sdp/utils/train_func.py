import time
import torch


def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, 
                num_epochs, device, checkpoint_path):
    """
    param:
        model (nn.Module): model to training process.
        criterion: loss function
        optimizer: 
        scheduler: to refine learning rate
        train_loader: Pytorch dataLoader which return data and label
        val_loader: Pytorch dataLoader which return data and label
        num_epochs (int): total epoches of training process
        device (str): cuda or cpu
        checkpoint_path (str): path to save best model

    return:
        history: dict contains training and evaluation record
    """
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'epoch': [], 'lr': []
    }

    best_val_acc = 0.0
    start_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        
        # --- training ---
        model.train()

        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for i, (csi_data, labels) in enumerate(train_loader):
            csi_data = csi_data.to(device)
            labels = labels.to(device)

            predictions = model(csi_data)
            loss = criterion(predictions, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * csi_data.size(0)

            _, predicted = torch.max(predictions.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = 100 * train_correct / train_total

        # --- eval ---
        model.eval()
        running_vloss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for csi_data, labels in val_loader:
                csi_data, labels = csi_data.to(device), labels.to(device)
                
                predictions = model(csi_data)
                
                loss = criterion(predictions, labels)
                running_vloss += loss.item() * csi_data.size(0)
                _, predicted = torch.max(predictions.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = running_vloss / len(val_loader.dataset)
        epoch_val_acc = 100 * val_correct / val_total
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        if scheduler:
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(epoch_val_loss) if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else scheduler.step()
        else:
            current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}% | "
              f"Duration: {epoch_duration:.2f}s | "
              f"LR: {current_lr:.6f}")
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['epoch'].append(epoch_duration)
        history['lr'].append(current_lr)
        
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            print(f"  -> new best acc: {best_val_acc:.2f}%. saved to {checkpoint_path}")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_val_acc': best_val_acc,
                'history': history,
            }, checkpoint_path)
            
    return history