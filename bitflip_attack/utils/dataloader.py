
class CustomDataLoader:
    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.batch_size = batch_size
             
    def __iter__(self):
        for i in range(len(self.dataset)):
            batch = self.dataset[i]
            yield (
                {
                    'input_ids': batch['input_ids'].unsqueeze(0),
                    'attention_mask': batch['attention_mask'].unsqueeze(0)
                }, 
                batch['label'].unsqueeze(0)
            )
         
    def __len__(self):
        return len(self.dataset)