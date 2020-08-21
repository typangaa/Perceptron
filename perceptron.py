import numpy as np
from csv import reader
import matplotlib.pyplot as plt
import time

def load_csv(filename,seed = 1):
    dataset = []
    labels = []
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        next(csv_reader) #skip the heading
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    
    np_dataset = np.asarray(dataset,dtype='f')

    #shuffle the dataset
    np.random.seed(seed)
    np.random.shuffle(np_dataset)

    #normalize the dataset
    dataset = normalization(np_dataset[:,:-1])
    labels = np.reshape(np_dataset[:,-1],(np_dataset[:,-1].shape[0],1))

    #print(dataset.shape)
    #print(labels.shape)
    return dataset,labels

def normalization(dataset):

        mean = dataset.mean()
        std = dataset.std()
        
        #print(mean)
        #print(std)
        #print(dataset)
        return (dataset-mean)/std

def train_test_split(dataset,test_set_ratio):

    test_set_len = int(len(dataset)*test_set_ratio)
    
    test_set_with_labels = dataset[0:test_set_len]

    train_set_with_labels = dataset[test_set_len:-1]

    #train_set_with_labels = np.append(train_set,train_labels,axis = 1)
    #test_set_with_labels = np.append(test_set,test_labels,axis = 1)

    return train_set_with_labels,test_set_with_labels

def plot_history(train_history,val_history):
    # Plot the results (shifting validation curves appropriately)
    plt.figure(figsize=(8,5))
    n = len(train_history)
    plt.plot(np.arange(0,n),train_history, color='orange')
    plt.plot(np.arange(0,n)+0.5,val_history,'r')  # offset both validation curves
    plt.legend(['Train Acc','Val Acc'])
    plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    plt.grid(True)
    plt.gca().set_ylim(0, 1) # set the vertical range to [0-1] 
    plt.show() 


class Perceptron():
    """Perceptron algorithm:
    
    Parameter
    ---------
    lr : float
        Learning rate (default = 0.001)
    epochs : int
        Number of iteration (default = 200)
    number_of_features : int
        Number of features (default = 8)
    seed : int
        Random seed for initializting weight (default = 1)

    Attributes
    ----------
    W : numpy array
        Weights with small random initialization 
    """
    def __init__(self, lr=0.001, epochs=200,random_initial_w = False, number_of_features = 8, seed = 1):
        self.bias = np.ones(1)
        self.epochs = epochs
        self.lr = lr
        self.number_of_features = number_of_features
        self.seed = seed

        #print(random_initial_w)
        if random_initial_w == True:
            np.random.seed(seed)
            self.W = np.random.uniform(-1,1,self.number_of_features+1)
        else:
            self.W = np.zeros(self.number_of_features+1) #include the bias term
        #self.show_initialization()

    def show_initialization(self):

        print('weight:')
        print(self.W)
        print('bias:')
        print(self.bias)
        print('number_of_features:')
        print(self.number_of_features)
        print(f'learning rate:{self.lr}')
        print(f'epochs:{self.epochs}')
        
    def activation(self, x):
        pass
        return 1 if x >= 0 else 0
 
    def predict(self, X):
        if len(X) == self.number_of_features :
            x = np.insert(X, 0, self.bias) # add the bias term if not added
        else:
            x = X
        z = self.W.T.dot(x)
        a = self.activation(z)
        return a
    
    def predict_list(self, X_list):
        prediction = []
        for x in X_list:
            prediction.append(self.predict(x))
        np_prediction = np.asarray(prediction)
        return np.reshape(np_prediction,(np_prediction.shape[0],1))
    
    def fit(self,train_dataset,val_dataset,stop_patient = 20,shuffle = False):
        train_accuracy_history = []
        val_accuracy_history = []
        temp_train_accuracy = 0
        temp_val_accuracy = 0
        max_accuracy = 0
        number_of_train_epoch = 0

        val_data = val_dataset[:,:-1]
        val_label = val_dataset[:,-1].reshape(val_dataset.shape[0],1)

        for epoch in range(self.epochs):
            temp_accuracy = 0
            #shuffle the dataset
            if shuffle == True:
                np.random.seed(epoch)
                np.random.shuffle(train_dataset)
            train_data = train_dataset[:,:-1]
            train_label = train_dataset[:,-1].reshape(train_dataset.shape[0],1)
            for i in range(train_data.shape[0]):
                x = np.insert(train_data[i], 0, self.bias) # add the bias term
                y = self.predict(x)
                error = train_label[i][0] - y
                self.W = self.W + self.lr * error * x
            temp_train_accuracy = self.accuracy(train_label,self.predict_list(train_data))
            temp_val_accuracy = self.accuracy(val_label,self.predict_list(val_data))
            
            train_accuracy_history.append(temp_train_accuracy)
            val_accuracy_history.append(temp_val_accuracy)

            if temp_val_accuracy > max_accuracy:
                max_accuracy = temp_val_accuracy

            count = 0
            if len(val_accuracy_history)>= stop_patient:
                for k in range(stop_patient):
                    if val_accuracy_history[-k-1]== max_accuracy:
                        break
                    elif val_accuracy_history[-k-1]<= max_accuracy:
                        count += 1
            if count == stop_patient:
                number_of_train_epoch = epoch
                return train_accuracy_history, val_accuracy_history, number_of_train_epoch
            number_of_train_epoch = epoch
        return train_accuracy_history, val_accuracy_history, number_of_train_epoch

    def fit_average(self,train_dataset,val_dataset,stop_patient = 20,shuffle =False):
        train_accuracy_history = []
        val_accuracy_history = []
        temp_train_accuracy = 0
        temp_val_accuracy = 0
        max_accuracy = 0

        val_data = val_dataset[:,:-1]
        val_label = val_dataset[:,-1].reshape(val_dataset.shape[0],1)
        
        W_temp = np.zeros(self.number_of_features+1)
        

        for epoch in range(self.epochs):
            temp_accuracy = 0
            #shuffle the dataset
            if shuffle == True:
                np.random.seed(epoch)
                np.random.shuffle(train_dataset)

            train_data = train_dataset[:,:-1]
            train_label = train_dataset[:,-1].reshape(train_dataset.shape[0],1)
        
            for i in range(train_data.shape[0]):
                x = np.insert(train_data[i], 0, self.bias) # add the bias term
                y = self.predict(x)
                error = train_label[i][0] - y
                W_temp += self.lr * error * x
                self.W += W_temp # accumulated sum of the weight vectors
            
            temp_train_accuracy = self.accuracy(train_label,self.predict_list(train_data))
            temp_val_accuracy = self.accuracy(val_label,self.predict_list(val_data))
            
            train_accuracy_history.append(temp_train_accuracy)
            val_accuracy_history.append(temp_val_accuracy)

            if temp_val_accuracy > max_accuracy:
                max_accuracy = temp_val_accuracy

            count = 0
            if len(val_accuracy_history)>= stop_patient:
                for k in range(stop_patient):
                    if val_accuracy_history[-k-1]== max_accuracy:
                        break
                    elif val_accuracy_history[-k-1]<= max_accuracy:
                        count += 1
            if count == stop_patient:
                
                return train_accuracy_history, val_accuracy_history
        
        return train_accuracy_history, val_accuracy_history

    def predict_vote(self,X):
        s = 0
        b = 0
        if len(X) == self.number_of_features :
            x = np.insert(X, 0, self.bias) # add the bias term if not added
        else:
            x = X

        for i in range(self.k):
            s += self.c[i]*np.sign(self.v[i].T.dot(x))
        
        a = self.activation(s)

        return a

    def predict_vote_list(self, X_list):
        prediction = []
        for x in X_list:
            prediction.append(self.predict_vote(x))

        np_prediction = np.asarray(prediction)
        return np.reshape(np_prediction,(np_prediction.shape[0],1))
    

    def fit_vote(self,train_dataset,val_dataset,stop_patient = 20):
        train_accuracy_history = []
        val_accuracy_history = []
        temp_train_accuracy = 0
        temp_val_accuracy = 0
        max_accuracy = 0

        train_data = train_dataset[:,:-1]
        train_label = train_dataset[:,-1].reshape(train_dataset.shape[0],1)
        
        val_data = val_dataset[:,:-1]
        val_label = val_dataset[:,-1].reshape(val_dataset.shape[0],1)
        
        W_temp = np.zeros(self.number_of_features+1)
        #Wa = np.zeros(self.number_of_features+1)
        self.c = [0]
        self.k = 0
        self.v = [self.W]

        for _ in range(self.epochs):
            temp_accuracy = 0
            for i in range(train_data.shape[0]):
                x = np.insert(train_data[i], 0, self.bias) # add the bias term
                y = self.predict_vote(x)
                error = train_label[i][0] - y
                if error == 0 :
                    self.c[self.k] += 1
                else:
                    self.v.append(self.v[self.k]+error*x)
                    self.c.append(1)
                    self.k += 1
                               
            
            temp_train_accuracy = self.accuracy(train_label,self.predict_vote_list(train_data))
            temp_val_accuracy = self.accuracy(val_label,self.predict_vote_list(val_data))
            
            train_accuracy_history.append(temp_train_accuracy)
            val_accuracy_history.append(temp_val_accuracy)

            if temp_val_accuracy > max_accuracy:
                max_accuracy = temp_val_accuracy

            count = 0
            if len(val_accuracy_history)>= stop_patient:
                for k in range(stop_patient):
                    if val_accuracy_history[-k-1]== max_accuracy:
                        break
                    elif val_accuracy_history[-k-1]<= max_accuracy:
                        count += 1
            if count == stop_patient:
                
                return train_accuracy_history, val_accuracy_history
        
        return train_accuracy_history, val_accuracy_history

    def accuracy(self, true_y,pred_y):
        count_error = np.sum(abs(true_y-pred_y))
        result = (len(true_y)-count_error)/len(true_y)
        return result

    def k_fold(self,data_with_labels,k = 5):
        fold_list = []
        dataset_copy = data_with_labels.copy()
        fold_size = int(data_with_labels.shape[0]/k)
        for i in range(k): 
            fold_list.append(dataset_copy[fold_size*i:fold_size*(i+1)])
        return fold_list
            
    def k_fold_cv(self,folds,k = 5,random_initial_w = True, lr = 0.001,stop_patient = 20):
        avg_accuracy = 0
        average_epoch = 0
        total_acc = 0
        epoch_run = []
        for i in range(k):
            np_test_set = np.array([], dtype=float).reshape(0,9)
            np_train_set = np.array([], dtype=float).reshape(0,9)
            np_fold_set = np.asarray(folds)
        

            for j in range(k):
                
                if j == i:
                    np_test_set = np.concatenate((np_test_set,np_fold_set[j]),axis= 0)
                else:
                    np_train_set = np.concatenate((np_train_set,np_fold_set[j]),axis= 0)
            #print(lr)
            perceptron = Perceptron(lr=lr, epochs=self.epochs,random_initial_w = random_initial_w, number_of_features = 8, seed = self.seed)
            train_acc, val_acc , number_of_epoch= perceptron.fit(np_train_set,np_test_set,stop_patient)
            total_acc += val_acc[-1]
            epoch_run.append(number_of_epoch)
        #print(epoch_run)
        average_epoch = np.sum(epoch_run)/k
        avg_accuracy = total_acc/k
        
        return avg_accuracy, int(average_epoch)


if __name__ == '__main__':
    
    best_epoch_train_acc_list = []
    best_epoch_val_acc_list = []
    best_epoch_test_acc_list = []

    best_epoch_with_shuffle_train_acc_list = []
    best_epoch_with_shuffle_val_acc_list = []
    best_epoch_with_shuffle_test_acc_list = []



    best_patient_train_acc_list = []
    best_patient_val_acc_list = []
    best_patient_test_acc_list = []

    best_patient_with_shuffle_train_acc_list = []
    best_patient_with_shuffle_val_acc_list = []
    best_patient_with_shuffle_test_acc_list = []



    best_epoch_avg_train_acc_list = []
    best_epoch_avg_val_acc_list = []
    best_epoch_avg_test_acc_list = []



    best_epoch_with_shuffle_avg_train_acc_list = []
    best_epoch_with_shuffle_avg_val_acc_list = []
    best_epoch_with_shuffle_avg_test_acc_list = []


    best_patient_avg_train_acc_list = []
    best_patient_avg_val_acc_list = []
    best_patient_avg_test_acc_list = []


    best_patient_with_shuffle_avg_train_acc_list = []
    best_patient_with_shuffle_avg_val_acc_list = []
    best_patient_with_shuffle_avg_test_acc_list = []


    
    for seed in range(50):
    
        random_Seed = seed

        dataset, labels = load_csv('diabetes.csv',seed = random_Seed)

        dataset_with_labels = np.append(dataset,labels,axis = 1)
        
        train_set_with_labels, test_set_with_labels = train_test_split(dataset_with_labels,test_set_ratio = 0.1)


        perceptron = Perceptron(lr=0.001, epochs=200,random_initial_w = True, number_of_features = 8, seed = random_Seed)

        k_folds = perceptron.k_fold(train_set_with_labels,k = 9)
        #,0.0003,0.0001
        lr_list = [1,0.1,0.01,0.001]
        stop_patient_list = [5,10,25,50,100]

        best_lr = 0
        best_stop_patient = 0
        best_acc = 0
        for lr in lr_list:
            for stop_patient in stop_patient_list:
                avg_accuracy,avg_epoch = perceptron.k_fold_cv(k_folds,k=9,random_initial_w = True,lr = lr,stop_patient=stop_patient)
                #print(f'{lr}:{stop_patient}:{avg_accuracy}')
                if best_acc < avg_accuracy:
                    best_acc = avg_accuracy
                    best_lr = lr
                    best_stop_patient = stop_patient
                    best_epoch = avg_epoch
                
        print(f'Random seed:{seed}')
        print(f'best_lr:{best_lr}\n best_acc:{best_acc}\n best_stop_patient:{best_stop_patient}\n best_epoch:{best_epoch}')

        #best_lr = 0.01
        #best_stop_patient = 40

        final_train_set_with_labels, final_val_set_with_labels = train_test_split(train_set_with_labels,test_set_ratio = 0.11111)
        
        test_set_without_labels = test_set_with_labels[:,:-1]
        test_set_labels = test_set_with_labels[:,-1].reshape(test_set_with_labels.shape[0],1)

        epoch_final_perceptron = Perceptron(lr = best_lr,random_initial_w = True,epochs = best_epoch,seed = random_Seed)
        patient_final_perceptron = Perceptron(lr = best_lr,random_initial_w = True,epochs = 200,seed = random_Seed)

        epoch_with_shuffle_final_perceptron = Perceptron(lr = best_lr,random_initial_w = True,epochs = best_epoch,seed = random_Seed)
        patient_with_shuffle_final_perceptron = Perceptron(lr = best_lr,random_initial_w = True,epochs = 200,seed = random_Seed)



        epoch_avg_final_perceptron = Perceptron(lr = best_lr,random_initial_w = True,epochs = best_epoch,seed = random_Seed)
        patient_avg_final_perceptron = Perceptron(lr = best_lr,random_initial_w = True,epochs = 200,seed = random_Seed)
        
        epoch_with_shuffle_avg_final_perceptron = Perceptron(lr = best_lr,random_initial_w = True,epochs = best_epoch,seed = random_Seed)
        patient_with_shuffle_avg_final_perceptron = Perceptron(lr = best_lr,random_initial_w = True,epochs = 200,seed = random_Seed)
        

        vote_final_perceptron = Perceptron(lr = best_lr,random_initial_w = True,epochs = best_epoch,seed = random_Seed)
        
        t1 = time.time()

        #best epoch
        best_epoch_train_acc,best_epoch_val_acc,_ = epoch_final_perceptron.fit(train_set_with_labels,final_val_set_with_labels,shuffle = False,stop_patient = best_epoch)
        best_epoch_y_predict = epoch_final_perceptron.predict_list(test_set_without_labels)
        best_epoch_test_acc = epoch_final_perceptron.accuracy(test_set_labels,best_epoch_y_predict)
        
        best_epoch_train_acc_list.append(best_epoch_train_acc[-1])
        best_epoch_val_acc_list.append(best_epoch_val_acc[-1])
        best_epoch_test_acc_list.append(best_epoch_test_acc)

        #best epoch with shuffle
        best_epoch_with_shuffle_train_acc,best_epoch_with_shuffle_val_acc,_ = epoch_with_shuffle_final_perceptron.fit(train_set_with_labels,final_val_set_with_labels,shuffle = True,stop_patient = best_epoch)
        best_epoch_with_shuffle_y_predict = epoch_with_shuffle_final_perceptron.predict_list(test_set_without_labels)
        best_epoch_with_shuffle_test_acc = epoch_with_shuffle_final_perceptron.accuracy(test_set_labels,best_epoch_with_shuffle_y_predict)
        
        best_epoch_with_shuffle_train_acc_list.append(best_epoch_with_shuffle_train_acc[-1])
        best_epoch_with_shuffle_val_acc_list.append(best_epoch_with_shuffle_val_acc[-1])
        best_epoch_with_shuffle_test_acc_list.append(best_epoch_with_shuffle_test_acc)

        #best patient
        best_patient_train_acc,best_patient_val_acc,_ = patient_final_perceptron.fit(train_set_with_labels,final_val_set_with_labels,shuffle = False,stop_patient = best_stop_patient)
        best_patient_y_predict = patient_final_perceptron.predict_list(test_set_without_labels)
        best_patient_test_acc = patient_final_perceptron.accuracy(test_set_labels,best_patient_y_predict)
        
        best_patient_train_acc_list.append(best_patient_train_acc[-1])
        best_patient_val_acc_list.append(best_patient_val_acc[-1])
        best_patient_test_acc_list.append(best_patient_test_acc)


        #best patient with shuffle
        best_patient_with_shuffle_train_acc,best_patient_with_shuffle_val_acc,_ = patient_with_shuffle_final_perceptron.fit(train_set_with_labels,final_val_set_with_labels,shuffle = True,stop_patient = best_stop_patient)
        best_patient_with_shuffle_y_predict = patient_with_shuffle_final_perceptron.predict_list(test_set_without_labels)
        best_patient_with_shuffle_test_acc = patient_with_shuffle_final_perceptron.accuracy(test_set_labels,best_patient_with_shuffle_y_predict)
        
        best_patient_with_shuffle_train_acc_list.append(best_patient_with_shuffle_train_acc[-1])
        best_patient_with_shuffle_val_acc_list.append(best_patient_with_shuffle_val_acc[-1])
        best_patient_with_shuffle_test_acc_list.append(best_patient_with_shuffle_test_acc)


        t2 = time.time()
        #print(t2 - t1)

        #average perceptron
        #best epoch
        best_epoch_avg_train_acc,best_epoch_avg_val_acc = epoch_avg_final_perceptron.fit_average(train_set_with_labels,test_set_with_labels,shuffle = False,stop_patient = best_epoch)
        best_epoch_avg_y_predict = epoch_avg_final_perceptron.predict_list(test_set_without_labels)
        best_epoch_avg_test_acc = epoch_avg_final_perceptron.accuracy(test_set_labels,best_epoch_avg_y_predict)
        
        best_epoch_avg_train_acc_list.append(best_epoch_avg_train_acc[-1])
        best_epoch_avg_val_acc_list.append(best_epoch_avg_val_acc[-1])
        best_epoch_avg_test_acc_list.append(best_epoch_avg_test_acc)

        #best epoch with shuffle
        best_epoch_with_shuffle_avg_train_acc,best_epoch_with_shuffle_avg_val_acc = epoch_with_shuffle_avg_final_perceptron.fit_average(train_set_with_labels,test_set_with_labels,shuffle = True,stop_patient = best_epoch)
        best_epoch_with_shuffle_avg_y_predict = epoch_with_shuffle_avg_final_perceptron.predict_list(test_set_without_labels)
        best_epoch_with_shuffle_avg_test_acc = epoch_with_shuffle_avg_final_perceptron.accuracy(test_set_labels,best_epoch_with_shuffle_avg_y_predict)
        
        best_epoch_with_shuffle_avg_train_acc_list.append(best_epoch_with_shuffle_avg_train_acc[-1])
        best_epoch_with_shuffle_avg_val_acc_list.append(best_epoch_with_shuffle_avg_val_acc[-1])
        best_epoch_with_shuffle_avg_test_acc_list.append(best_epoch_with_shuffle_avg_test_acc)

        #best patient
        best_patient_avg_train_acc,best_patient_avg_val_acc = patient_avg_final_perceptron.fit_average(train_set_with_labels,final_val_set_with_labels,shuffle = False,stop_patient = best_stop_patient)
        best_patient_avg_y_predict = patient_final_perceptron.predict_list(test_set_without_labels)
        best_patient_avg_test_acc = patient_final_perceptron.accuracy(test_set_labels,best_patient_avg_y_predict)
        best_patient_avg_train_acc_list.append(best_patient_avg_train_acc[-1])
        best_patient_avg_val_acc_list.append(best_patient_avg_val_acc[-1])
        best_patient_avg_test_acc_list.append(best_patient_avg_test_acc)

        #best patient with shuffle
        best_patient_with_shuffle_avg_train_acc,best_patient_with_shuffle_avg_val_acc = patient_with_shuffle_avg_final_perceptron.fit_average(train_set_with_labels,final_val_set_with_labels,shuffle = True,stop_patient = best_stop_patient)
        best_patient_with_shuffle_avg_y_predict = patient_with_shuffle_avg_final_perceptron.predict_list(test_set_without_labels)
        best_patient_with_shuffle_avg_test_acc = patient_with_shuffle_avg_final_perceptron.accuracy(test_set_labels,best_patient_with_shuffle_avg_y_predict)
        best_patient_with_shuffle_avg_train_acc_list.append(best_patient_with_shuffle_avg_train_acc[-1])
        best_patient_with_shuffle_avg_val_acc_list.append(best_patient_with_shuffle_avg_val_acc[-1])
        best_patient_with_shuffle_avg_test_acc_list.append(best_patient_with_shuffle_avg_test_acc)

        #best_stop_patient_avg_train_acc,best_stop_patient_avg_val_acc = patient_avg_final_perceptron.fit_average(train_set_with_labels,test_set_with_labels,stop_patient = best_stop_patient)
        t3 = time.time()
        #print(t3 - t2)

        #vote_train_acc,vote_val_acc = vote_final_perceptron.fit_vote(train_set_with_labels,test_set_with_labels,stop_patient = best_stop_patient)
        
        #t4 = time.time()
        #print(t4 - t3)
        """
        print("best_epoch")
        print(f'Training acc:{best_epoch_train_acc[-1]}')
        print(f'Val acc:{best_epoch_val_acc[-1]}')
        print(f'Test acc:{best_epoch_test_acc}')
        print("best_patient")
        print(f'Training acc:{best_patient_train_acc[-1]}')
        print(f'Val acc:{best_patient_val_acc[-1]}')
        print(f'Test acc:{best_patient_test_acc}')

        print("best_epoch with shuffle")
        print(f'Training acc:{best_epoch_with_shuffle_train_acc[-1]}')
        print(f'Val acc:{best_epoch_with_shuffle_val_acc[-1]}')
        print(f'Test acc:{best_epoch_with_shuffle_test_acc}')
        
        print("best_patient with shuffle")
        print(f'Training acc:{best_patient_with_shuffle_train_acc[-1]}')
        print(f'Val acc:{best_patient_with_shuffle_val_acc[-1]}')
        print(f'Test acc:{best_patient_with_shuffle_test_acc}')
        

        print("best_epoch_avg ")
        print(f'Training acc:{best_epoch_avg_train_acc[-1]}')
        print(f'Val acc:{best_epoch_avg_val_acc[-1]}')
        print(f'Test acc:{best_epoch_avg_test_acc}')

        print("best_patient_avg")
        print(f'Training acc:{best_patient_avg_train_acc[-1]}')
        print(f'Val acc:{best_patient_avg_val_acc[-1]}')
        print(f'Test acc:{best_patient_avg_test_acc}')

        print("best_epoch_avg with shuffle")
        print(f'Training acc:{best_epoch_with_shuffle_avg_train_acc[-1]}')
        print(f'Val acc:{best_epoch_with_shuffle_avg_val_acc[-1]}')
        print(f'Test acc:{best_epoch_with_shuffle_avg_test_acc}')
        
        print("best_patient_avg with shuffle")
        print(f'Training acc:{best_patient_with_shuffle_avg_train_acc[-1]}')
        print(f'Val acc:{best_patient_with_shuffle_avg_val_acc[-1]}')
        print(f'Test acc:{best_patient_with_shuffle_avg_test_acc}')
        

        


        #print(vote_train_acc[-1])
        #print(vote_val_acc[-1])

        #predict_y = final_perceptron.predict_list(test_set)

        #accuracy = final_perceptron.accuracy(test_labels,predict_y)
        #print(accuracy)

        plot_history(best_epoch_train_acc,best_epoch_val_acc)
        plot_history(best_patient_train_acc,best_patient_val_acc)
        plot_history(best_epoch_with_shuffle_train_acc,best_epoch_with_shuffle_val_acc)
        plot_history(best_patient_with_shuffle_train_acc,best_patient_with_shuffle_val_acc)



        plot_history(best_epoch_avg_train_acc,best_epoch_avg_val_acc)
        plot_history(best_patient_avg_train_acc,best_patient_avg_val_acc)
        plot_history(best_epoch_with_shuffle_train_acc,best_epoch_with_shuffle_val_acc)
        plot_history(best_patient_with_shuffle_train_acc,best_patient_with_shuffle_val_acc)
        """
        #plot_history(vote_train_acc,vote_val_acc)
    print("best_epoch")
    #print(f'Training acc:{best_epoch_train_acc_list}')
    print(f'Training acc:{np.average(best_epoch_train_acc_list)}')
    #print(f'Val acc:{best_epoch_val_acc_list}')
    print(f'Val acc:{np.average(best_epoch_val_acc_list)}')
    #print(f'Test acc:{best_epoch_test_acc_list}')
    print(f'Test acc:{np.average(best_epoch_test_acc_list)}')

    print("best_epoch with shuffle")
    #print(f'Training acc:{best_epoch_with_shuffle_train_acc_list}')
    ##print(f'Val acc:{best_epoch_with_shuffle_val_acc_list}')
    #print(f'Test acc:{best_epoch_with_shuffle_test_acc_list}')
    print(f'Training acc:{np.average(best_epoch_with_shuffle_train_acc_list)}')
    print(f'Val acc:{np.average(best_epoch_with_shuffle_val_acc_list)}')
    print(f'Test acc:{np.average(best_epoch_with_shuffle_test_acc_list)}')

    best_epoch_diff_test_acc = np.average(best_epoch_with_shuffle_test_acc_list) - np.average(best_epoch_test_acc_list)
    print(f'test acc dif of best_epoch with and without shuffle:{best_epoch_diff_test_acc}')

    print("best_patient")
    #print(f'Training acc:{best_patient_train_acc_list}')
    print(f'Training acc:{np.average(best_patient_train_acc_list)}')
    #print(f'Val acc:{best_patient_val_acc_list}')
    print(f'Val acc:{np.average(best_patient_val_acc_list)}')
    #print(f'Test acc:{best_patient_test_acc_list}')
    print(f'Test acc:{np.average(best_patient_test_acc_list)}')

    print("best_patient with shuffle")
    #print(f'Training acc:{best_patient_with_shuffle_train_acc_list}')
    #print(f'Val acc:{best_patient_with_shuffle_val_acc_list}')
    #print(f'Test acc:{best_patient_with_shuffle_test_acc_list}')
    print(f'Training acc:{np.average(best_patient_with_shuffle_train_acc_list)}')
    print(f'Val acc:{np.average(best_patient_with_shuffle_val_acc_list)}')
    print(f'Test acc:{np.average(best_patient_with_shuffle_test_acc_list)}')

    best_patient_diff_test_acc = np.average(best_patient_with_shuffle_test_acc_list) - np.average(best_patient_test_acc_list)
    print(f'test acc dif of best_patient with and without shuffle:{best_patient_diff_test_acc}')

    print("best_epoch_avg ")
    #print(f'Training acc:{best_epoch_avg_train_acc_list}')
    #print(f'Val acc:{best_epoch_avg_val_acc_list}')
    #print(f'Test acc:{best_epoch_avg_test_acc_list}')
    print(f'Training acc:{np.average(best_epoch_avg_train_acc_list)}')
    print(f'Val acc:{np.average(best_epoch_avg_val_acc_list)}')
    print(f'Test acc:{np.average(best_epoch_avg_test_acc_list)}')

    print("best_epoch_avg with shuffle")
    #print(f'Training acc:{best_epoch_with_shuffle_avg_train_acc_list}')
    #print(f'Val acc:{best_epoch_with_shuffle_avg_val_acc_list}')
    #print(f'Test acc:{best_epoch_with_shuffle_avg_test_acc_list}')
    print(f'Training acc:{np.average(best_epoch_with_shuffle_avg_train_acc_list)}')
    print(f'Val acc:{np.average(best_epoch_with_shuffle_avg_val_acc_list)}')
    print(f'Test acc:{np.average(best_epoch_with_shuffle_avg_test_acc_list)}')

    best_epoch_avg_diff_test_acc = np.average(best_epoch_with_shuffle_avg_test_acc_list) - np.average(best_epoch_avg_test_acc_list)
    print(f'test acc dif of best_epoch_avg with and without shuffle:{best_epoch_avg_diff_test_acc}')

    print("best_patient_avg")
    #print(f'Training acc:{best_patient_avg_train_acc_list}')
    #print(f'Val acc:{best_patient_avg_val_acc_list}')
    #print(f'Test acc:{best_patient_avg_test_acc_list}')
    print(f'Training acc:{np.average(best_patient_avg_train_acc_list)}')
    print(f'Val acc:{np.average(best_patient_avg_val_acc_list)}')
    print(f'Test acc:{np.average(best_patient_avg_test_acc_list)}')

    print("best_patient_avg with shuffle")
    #print(f'Training acc:{best_patient_with_shuffle_avg_train_acc_list}')
    #print(f'Val acc:{best_patient_with_shuffle_avg_val_acc_list}')
    #print(f'Test acc:{best_patient_with_shuffle_avg_test_acc_list}')
    print(f'Training acc:{np.average(best_patient_with_shuffle_avg_train_acc_list)}')
    print(f'Val acc:{np.average(best_patient_with_shuffle_avg_val_acc_list)}')
    print(f'Test acc:{np.average(best_patient_with_shuffle_avg_test_acc_list)}')

    best_patient_avg_diff_test_acc = np.average(best_patient_with_shuffle_avg_test_acc_list) - np.average(best_patient_avg_test_acc_list)
    print(f'test acc dif of best_patient_avg with and without shuffle:{best_patient_avg_diff_test_acc}')