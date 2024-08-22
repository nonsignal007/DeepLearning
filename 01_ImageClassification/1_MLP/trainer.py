
import torch
import time

from data_loader import DataLoad

class Train:
    def __init__(self, model, optimizer, criterion, EPOCHS, config , train=True):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = torch.backends.mps.is_available() if 'mps' else 'cpu'
        self.epoch = EPOCHS
        data_load = DataLoad(config['ROOT'], config['SEED'], config['BATCH_SIZE'], config['VAL_RATIO'])
        self.train_iterator, self.valid_iterator, self.test_iterator = data_load
        if train:
            best_valid_loss = float('inf')

            for epoch in range(self.epoch):
                start_time = time.monotonic()
                
                train_loss, train_acc = self.train(model, self.train_iterator, optimizer, criterion, self.device)
                valid_loss, valid_acc = self.evaluate(model, self.valid_iterator, criterion, self.device)

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), 'mlp-model.pt')
                
                end_time = time.monotonic()

                epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

                print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
                print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
                print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
        else:
            self.evaluate(self.model, self.iterator, self.criterion)

    def calculate_accuracy(self, y_pred, y):
        top_pred = y_pred.argmax(1, keepdim=True)
        correct = top_pred.eq(y.view_as(top_pred)).sum()
        acc = correct.float() / y.shape[0]
        return acc

    def train(self, model, iterator, optimizer, criterion):
        epoch_loss = 0
        epoch_acc = 0

        model.train()

        for (x, y) in tqdm(iterator, desc='Training', leave=False):
            x = x.to(self.device)
            y = y.to(self.device)

            optimizer.zero_grad()

            y_pred, _ = model(x)

            loss = criterion(y_pred, y)
            
            acc = calculate_accuracy(y_pred, y)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)
    
    def evaluate(self, model, iterator, criterion):
        epoch_loss = 0
        epoch_acc = 0

        model.eval()

        with torch.no_grad():
            for (x,y) in tqdm(iterator, desc = 'Evaluating', leave=False):
                x = x.to(self.device)
                y = y.to(self.device)

                y_pred, _ = model(x)

                loss = criterion(y_pred, y)

                acc = calculate_accuracy(y_pred, y)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    
