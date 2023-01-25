import csv

## 
# Dot product:
# activation = sum(weight_i * x_i) + bias
# This will attempt to make a prediction on the class given the data points
# and weights
##  
def predict (row, weights):
    ## adds the bias constant 
    activation = weights[0]
    for i in range(len(row)-1):
        #ignores bias adds the dot product of the rest
        activation += weights[i + 1] * row[i]
    ## returns the predicted class
    return 1.0 if activation >= 0.0 else 0.0



## new_weight = old_weight + learning_rate * (expected - predicted) * x
## bias = bias + learning_rate * (expected - predicted)
def optimize_weights (points, step, epochs):
    dim = len(points[0])
    weights = [0.0] * dim
    for iter in range(epochs):
        sum_error = 0.0
        for row in points:
            prediction = predict(row, weights)
            # expected - predicted
            error = row[-1] - prediction
            sum_error += error**2
            # bias = bias + learning_rate * (expected - predicted)
            weights[0] = weights[0] + step * error
            for i in range(dim-1):
            # new_weight = old_weight + learning_rate * (expected - predicted) * x
                weights[i+1] = weights[i+1] + step * error * row[i]
    return weights



# TP = predcicted = expected = 1
# FP = predcicted != expected = 1
# TN = predcicted = expected = 0
# FN = predcicted != expected = 0


## TP/TP+FN
def recall (data, weights):
    TP = 0.0
    FN = 0.0
    for row in data:
        prediction = predict(row, weights)
        # TP count 
        if row[-1] == 1 and  prediction == 1:
            TP += 1
        # FN count 
        if row[-1] == 1 and  prediction == 0:
            FN += 1        
    if FN == 0 and TP == 0: 
        return .0
    else: 
        return TP/(TP+FN)
        

## TP/TP+FP
def precision (data, weights):
    TP = 0.0
    FP = 0.0
    for row in data:
        prediction = predict(row, weights)
        # TP count 
        if row[-1] == 1 and  prediction == 1:
            TP += 1
        # FP count 
        if row[-1] == 0 and  prediction == 1:
            FP += 1         
    if FP == 0 and TP == 0: 
        return .0
    else:
        return TP/(TP+FP)

## TP+TN/TP+FP+FN+TN
def accuracy (data, weights):
    TP = 0.0
    TN = 0.0
    for row in data:
        prediction = predict(row, weights)
        # TP count 
        if row[-1] == 1 and  prediction == 1:
            TP += 1
        # TN count 
        if row[-1] == 0 and  prediction == 0:
            TN += 1     
    return float((TP+TN)/float(len(data)))

def fit(data, learn_rate, epochs):
    return optimize_weights(data, learn_rate, epochs)

def score(data, weights):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    print("Accuracy=%s, Precision=%s, Recall=%s" % (accuracy(data, weights),precision(data, weights),recall(data, weights)))
    for row in data:
        prediction = predict(row, weights)
        # TP count 
        if row[-1] == 1 and  prediction == 1:
            TP += 1
        # FP count 
        if row[-1] == 0 and  prediction == 1:
            FP += 1
        # TN count 
        if row[-1] == 0 and  prediction == 0:
            TN += 1
        # FN count 
        if row[-1] == 1 and  prediction == 0:
            FN += 1    
    print("     T    F  ")
    print("T | %s | %s |" % (TP,FP))   
    print("  |--------|")
    print("F | %s | %s |"% (FN,TN)) 

def data_split (data, test_size):
    size = len(data)
    mid_pnt = int(test_size * size)
    train = data[mid_pnt:]
    test = data[:mid_pnt]
    return[train,test]

## getting the dataset and one hot encoding 
file = open('sonar.all-data', "r")
data = list(csv.reader(file, delimiter=","))
for row in data:
    if row[-1] == 'M':
        row[-1] = 1
    else:
        row[-1] = 0
    for i in range(len(row)):
        row[i] = float(row[i])   
split = data_split(data, .4)
test_split = split[1]
train_split = split[0]


# Testing
# Sort of rough, it really struggles with the test split
# which makes me think it was overfit to the training split. 
# But with more inspection since the dataset was not randomised 
# the majority of the test_spit were class 0, somthing my model 
# struggled with.
weights = fit(train_split, .1, 150)
print("train acc:")
score(train_split,weights)
print("test acc:")
score(test_split,weights)



