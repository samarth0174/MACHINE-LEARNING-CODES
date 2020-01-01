import pandas as pd
import numpy as np
from pprint import pprint



#Import the dataset 
dataset = pd.read_csv('zoo.data',
                      names=['animal_name','hair','feathers','eggs','milk',
                                                   'airbone','aquatic','predator','toothed','backbone',
                                                  'breathes','venomous','fins','legs','tail','domestic','catsize','class',])#Import all columns omitting the fist which consists the names of the animals



#drop the animal names since this is not a good feature to split the data on
dataset=dataset.drop('animal_name',axis=1)


###################



def entropy(target_col):
    elements,counts = np.unique(target_col,return_counts = True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy


################### 
    
###################


def InfoGain(data,split_attribute_name,target_name="class"):
    total_entropy = entropy(data[target_name])
    vals,counts= np.unique(data[split_attribute_name],return_counts=True)
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain
       
###################

###################


def ID3(data,originaldata,features,target_attribute_name="class",parent_node_class = None):

    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]

    elif len(data)==0:
        return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name],return_counts=True)[1])]
    
    
    elif len(features) ==0:
        return parent_node_class
    
    else:
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name],return_counts=True)[1])]
        
        item_values = [InfoGain(data,feature,target_attribute_name) for feature in features] #Return the information gain values for the features in the dataset
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        tree = {best_feature:{}}
        
        
        features = [i for i in features if i != best_feature]
        
        for value in np.unique(data[best_feature]):
            value = value
            sub_data = data.where(data[best_feature] == value).dropna()
            
            subtree = ID3(sub_data,dataset,features,target_attribute_name,parent_node_class)
            
            tree[best_feature][value] = subtree
            
        return(tree)    
                
###################

###################


    
    
def predict(query,tree,default = 1):
       
    #1.
    for key in list(query.keys()):
        if key in list(tree.keys()):
            #2.
            try:
                result = tree[key][query[key]] 
            except:
                return default
  
            #3.
            result = tree[key][query[key]]
            #4.
            if isinstance(result,dict):
                return predict(query,result)

            else:
                return result


###################

###################

def train_test_split(dataset):
    training_data = dataset.iloc[:80].reset_index(drop=True)
    testing_data = dataset.iloc[80:].reset_index(drop=True)
    return training_data,testing_data

training_data = train_test_split(dataset)[0]
testing_data = train_test_split(dataset)[1] 



def test(data,tree):
    
    queries = data.iloc[:,:-1].to_dict(orient = "records")
    
    predicted = pd.DataFrame(columns=["predicted"]) 
    
    #Calculate the prediction accuracy
    for i in range(len(data)):
        predicted.loc[i,"predicted"] = predict(queries[i],tree,1.0) 
    print('The prediction accuracy is: ',(np.sum(predicted["predicted"] == data["class"])/len(data))*100,'%')
    return predicted   

def precision(label, confusion_matrix):
    col = confusion_matrix[:, label]
    return confusion_matrix[label, label] / col.sum()
    
def recall(label, confusion_matrix):
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()

def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return (diagonal_sum / sum_of_all_elements) * 100  

def error_rate(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return (1 - (diagonal_sum / sum_of_all_elements))* 100  

def precision_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_precisions = 0
    for label in range(rows):
        sum_of_precisions += precision(label, confusion_matrix)
    return sum_of_precisions / rows

def recall_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_recalls = 0
    for label in range(columns):
        
        sum_of_recalls += recall(label, confusion_matrix)
    return sum_of_recalls / columns

"""
Train the tree, Print the tree and predict the accuracy
"""
print("IMPLEMENTED ID3 DECISION TREES ON  UCI Zoo Data Set")
tree = ID3(training_data,training_data,training_data.columns[:-1])
print("************************DECISION TREE**************************\n")
pprint(tree)
predicted = test(testing_data,tree)


"""confusion matrix"""

pp  = testing_data.iloc[:,16]
pp1 = pp.values
pred = predicted.values.reshape(21,)
pred = np.int64(pred)
conf_data  = pd.concat([pp,predicted],sort=False,axis=1)

confmat  = np.zeros([len(pp.unique()),len(pp.unique())])
confmat  =np.int64(confmat)
for i in range(len(predicted)):
    confmat[pp[i]-1][pred[i]-1]+=1
    

print("************************CONFUSION MATRIX**************************\n")
print(confmat)
print("The prediction accuracy is:",accuracy(confmat))
print("Error Rate:",error_rate(confmat),"\n")
print("label precision recall")
prec_avg=0
recall_avg=0
for label in range(7):
    print(f"{label+1:5d} {precision(label, confmat):9.3f} {recall(label, confmat):6.3f}")
    prec_avg += precision(label, confmat)
    recall_avg += recall(label, confmat)

print("precision total:", prec_avg/7)

print("recall total:", recall_avg/7)    

