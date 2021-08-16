import os
import os.path
import pickle

root_dir = os.getcwd()
dataset_dir = os.path.join(root_dir, 'dataset')
models_dir = os.path.join(root_dir, 'models')
save_model_path = os.path.join(models_dir, 'knn.pkl')
data_model_path = os.path.join(models_dir, 'data.pkl')

def loadData(file):
    db = ()
    if os.path.exists(file):
        # dbfile = open(file, 'rb') 
        with open(file,'rb') as rfp:     
            db = pickle.load(rfp)       
        # for keys in db:
        #     print(keys, '=>', db[keys])
            rfp.close()  
    print(db)        
    return db

loadData(data_model_path)