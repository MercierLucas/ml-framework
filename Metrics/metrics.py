import numpy as np

from ml_framework.Utils.preprocessing import from_categorical

def eq(arr, e):
    return np.array([i == e for i in arr])

class Metrics:

    def __init__(self, y, y_pred):
        assert y.shape[0] == y_pred.shape[0], f"y:{y.shape[0]} and y_pred:{y_pred.shape[0]} must have same number of elements"
        
        y = y.squeeze()
        y_pred = y_pred.squeeze()

        if len(y.shape) > 1:
            if y.shape[1] != 1:
                y = from_categorical(y)

            if y_pred.shape[1] != 1:
                y_pred = from_categorical(y_pred)
        
        self.uniques = np.unique(y)
        self.conf = self.confusion_matrix(y, y_pred)
        self.y_true = y
        self.y_pred = y_pred
        
        self.metrics = {
            "recall" : self.recall,
            "precision" : self.precision
        }
        

    def available_metrics(self):
        return [m for m in self.metrics]
    
    def compute_many(self, metrics=["recall"]):
        for m in metrics:
            assert m in self.available_metrics(), f"{m} isn't an available metric"
            
        res = dict()
        for m in metrics:
            res[m] = self.metrics[m]()
            
        return res
        
    def confusion_matrix(self, y, y_pred):
        res = []
        for i in self.uniques:
            res.append([(sum(eq(y_pred, i) & eq(y, j))) for j in self.uniques])
        return np.array(res)
    
    def recall(self, mean_only=True):
        rec = []
        for i in range(self.conf.shape[1]):
            col = self.conf[:,i]
            if sum(col) == 0:
                res = 0
            else:
                res = col[i]/sum(col)
            rec.append(res)
            
        if mean_only:
            return np.mean(rec)
        rec.append(np.mean(rec))
        
        return rec
    
    def precision(self, mean_only=True):
        rec = []
        for i in range(self.conf.shape[1]):
            row = self.conf[i,:]
            if sum(row) == 0:
                res = 0
            else:
                res = row[i]/sum(row)
            rec.append(res)
        if mean_only:
            return np.mean(rec)
        rec.append(np.mean(rec))
        
        return rec
    
    def report(self, digits = 3):
        metrics = [self.metrics[k](False) for k in self.metrics]
        
        max_length = max([len(str(i)) for i in self.available_metrics()]) + 5
        
        header = f'{" ":^{max_length}}'
        for i in self.available_metrics():
            header += f'{i:^{max_length}}'
        print(header)
        
        length = len(self.available_metrics())
        labels = self.uniques.tolist()
        labels.append("Average")
        
        for i in range(len(labels)):
            row_txt = f'{labels[i]:^{max_length}}'
            for j in range(length):
                row_txt += f'{metrics[j][i]:^{max_length}.{digits}f}'
            print(row_txt)
            
    def display_conf(self, labels = None):
        if labels is None:
            labels = self.uniques
        
        max_length = max([len(str(i)) for i in labels]) + 5
        
        header = f'{" ":^{max_length}}'
        for i in labels:
            header += f'{i:^{max_length}}'
        print(header) 
        
        for i in range(self.conf.shape[0]):
            row = self.conf[i]
            row_txt = f'{labels[i]:^{max_length}}'
            for col in row:
                row_txt += f'{col:^{max_length}}'
            print(row_txt)