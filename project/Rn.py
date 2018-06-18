import numpy as np
import pandas as pd
import time

  
class NN :
    def LoadData(self, filePath):
        self.df = pd.read_csv(filePath, header = None)
        
        self.df = self.df.dropna()
        self.df = (self.df - self.df.min()) / (self.df.max() - self.df.min())
        self.setClassList = list(set(self.df.iloc[:,-1]))
        
    def Initialization (self):
        self.averagePerClass = pd.DataFrame()
        self.matrix = np.identity(len(self.setClassList))

        self.percentsArray = []
        self.f = np.vectorize(self.ActivationFunction)
        self.Run()
        
    def CrossValidation(self,_df):
        df2 = _df.reindex(np.random.permutation(_df.index))#.reset_index(drop = True)
        t1, t2, t3, testing=np.array_split(df2,4)
        training=np.concatenate((np.concatenate((t1,t2),axis=0),t3),axis=0)
        self.training = pd.DataFrame(training)
        self.testing = pd.DataFrame(testing)
    
    def GetCodificateClass(self,clasa):
        index = self.setClassList.index(clasa)
        return self.matrix[index]
    
    def ConcordantPair(self,_atr1_clasa, _atr2_clasa):
        return np.sign(_atr1_clasa[0] - _atr2_clasa[0]) == np.sign(_atr1_clasa[1] - _atr2_clasa[1])
    
    def DiscordantPair(self,_atr1_clasa, _atr2_clasa):
        return np.sign(_atr1_clasa[0] - _atr2_clasa[0]) == -np.sign(_atr1_clasa[1] - _atr2_clasa[1])
    
    def Pair(self,_classAttribut, _listAttributeClass):
        concordantPairs = 0
        discordantPairs = 0
        for element in _listAttributeClass:
            if self.ConcordantPair(_classAttribut, element) :
                concordantPairs += 1
            elif self.DiscordantPair(_classAttribut, element) :
                discordantPairs += 1
        return [concordantPairs, discordantPairs]
    
    def Coefficient(self,_columnData, _classData, average):
        coef = [0 , 0]
        for indexRow in np.arange(len(_classData) -1 ):
            element = [_columnData[indexRow] - average,_classData[indexRow]]
            lista = np.column_stack((_columnData[indexRow : ], _classData[indexRow :]))
            newCoefficient = self.Pair(element, lista )
            coef = [x + y for x, y in zip(coef, newCoefficient)]
        return abs((coef[0] - coef[1]) /(coef[0] + coef[1]))
        
    def WeightInitialization(self, _df):  
        weights = np.empty(shape = self.averagePerClass.shape)
        weights.fill(0)
        for classIndex in np.arange(len(self.averagePerClass)):
            for columnIndex in np.arange(len(_df.columns) - 1 ):
                weights[classIndex,columnIndex] = self.Coefficient(_df.iloc[:,columnIndex],_df.iloc[:,-1], self.averagePerClass.iloc[classIndex,columnIndex])
        return weights
    
    def Discriminant(self, _df, _weights):
        u = np.zeros((len(_df),len(_weights)))
        for lineIndex in np.arange(len(_df)):
            for weightClassIndex in np.arange(len(_weights)):
                for columnIndex in np.arange(len(_df.columns) -1):
                    val = np.abs(_df.iloc[lineIndex, columnIndex] - self.averagePerClass.iloc[weightClassIndex,columnIndex])
                    u[lineIndex,weightClassIndex] += _df.iloc[lineIndex,columnIndex] * _weights[weightClassIndex, columnIndex]  /val #( val if val != 0 else 1)
        return u
                    
    def ActivationFunction(self, x):
        return 1.7159 * np.tanh(2.0/3 * x)
                
    def SoftMax(self, activationResults, _forecasted):
        g = np.zeros(activationResults.shape)
        for lineIndex in np.arange(len(activationResults)):
            suma = 0
            maxim = np.max(activationResults[lineIndex])
            _forecasted.append( self.setClassList[ np.argmax( activationResults[lineIndex] ) ] )
            for columnIndex in np.arange(len(activationResults[0])):
                suma += np.exp( activationResults[lineIndex,columnIndex] - maxim)
            for columnIndex in np.arange(len(activationResults[0])):
                g[lineIndex,columnIndex] = np.exp( activationResults[lineIndex,columnIndex]) / suma
        return g
    
     
    def Error(self, _g, _classes):
        error =np.zeros(len(_classes))
        for lineIndex in np.arange(len(_classes)):
            codificatedClass = self.GetCodificateClass(_classes[lineIndex])
            for y1,y2 in zip(_g[lineIndex],codificatedClass):
                error[lineIndex] += np.sqrt((y2 - y1)**2)
        return error
    
    def WeightsUpdate(self, _df, _weights, _error):
        newPonderi = np.empty(shape = self.averagePerClass.shape)
        newPonderi.fill(0)
        for classIndex in np.arange(len(self.averagePerClass)):
            for columnIndex in np.arange(len(_df.columns) - 1 ):
                newPonderi[classIndex,columnIndex] = self.Coefficient(_df.iloc[:,columnIndex],_error, self.averagePerClass.iloc[classIndex,columnIndex]) * _weights[classIndex,columnIndex]
            suma = np.sum(newPonderi[classIndex])
            for columnIndex in np.arange(len(_df.columns) - 1 ):
                newPonderi[classIndex,columnIndex] =  newPonderi[classIndex,columnIndex] / suma
            
        return newPonderi
    
    def Accuracy(self, _initials, _forecasted):
        count = 0 
        for item1,item2 in zip(_initials,_forecasted):
            if item1 == item2:
                count +=1
        return count / len(_initials) *100 
    
    def TrainingStep(self, _trainingPercent, _forecasted):
        for iter in np.arange(100):
            discriminant = self.Discriminant(self.training,self.weights)
            resultSet = self.f(discriminant)
            g = self.SoftMax(resultSet, _forecasted)
            error = self.Error(g, self.training.iloc[:,-1])
            newPonderi = self.WeightsUpdate(self.training,self.weights,error)
            self.weights = newPonderi
            _trainingPercent.append ( self.Accuracy(self.training.iloc[:,-1],_forecasted))
            _forecasted.clear()
    
    def TestingStep(self, _forecasted, step):
        
        discriminant = self.Discriminant(self.testing,self.weights)
        resultSet = self.f(discriminant)
        self.SoftMax(resultSet, _forecasted)
        return self.Accuracy(self.testing.iloc[:,-1], _forecasted)
            
    def Run(self):
        for step in np.arange(1):
            self.CrossValidation(self.df)
            grouped = (self.training.groupby(self.df.columns[-1]))
            self.averagePerClass = grouped.aggregate(np.mean)
            forecasted = list()
            self.weights = self.WeightInitialization(self.training)
            trainingPercent = list()
            percentTesting = 0
            
            timer = time.time()
            
            self.TrainingStep(trainingPercent, forecasted)           
            percentTesting = self. TestingStep(forecasted, step)
            
            forecasted.clear()
            print ("step: " , step , trainingPercent[-1], " / ", percentTesting)
            self.percentsArray.append([trainingPercent[-1], percentTesting])
            print( timer - time.time())