# %%writefile framework.py
import tensorflow as tf
from collections import OrderedDict
from IPython.display import clear_output
import time
import datetime
import numpy as np

class TensorFlowComputation():
    """
    The base class for carrying out computations in tensor flow.
    On its own this doesn't really do anything, but contains methods common to any computation you might want to run in tensor flow.
    Expects:
        - graphStandIn instance
        - sessionComputation method
    Run the computation by calling `runSession()`
    """
    graphStandIn = None
    def createSession(self):
        """
        Creates a new tensor flow session and assigns it to self.session.
        """
        if not hasattr(self,'session') or self.session._closed:
            self.session = tf.Session(graph=self.graphStandIn.graph)
    def closeSession(self):
        if hasattr(self,'session'):
            self.session.close()
    def runSession(self):
        """this won't close the session after the computation is done. Do so by calling `closeSession()`"""
        self.createSession()
        with self.graphStandIn.graph.as_default():
            initVariables = tf.initialize_all_variables()
        self.session.run(initVariables)
        self.sessionComputation()
    def sessionComputation(self):
        pass
            
class LearningModel(TensorFlowComputation):
    """
    The base class for any machine learning algorithm that involves training using SGD
    It expects values for trainingDataset, trainingLabels, numberOfTrainingSteps.
    Expects:
        - graphStandIn instance
        - trainingDataset variable
        - trainingLabels variable
        - numberOfTrainingSteps variable
    Optional
        - doStuffInTrainingLoop(step,trainTimes,stuffTimes,trainingEventData) method
        - batch_size int
    Run the training by calling `runSession()`
    """
    trainingDataset = None
    trainingLabels = None
    numberOfTrainingSteps = 0
    batch_size = 128
    
    def sessionComputation(self):
        for step in range(self.numberOfTrainingSteps):
            trainingEventData = self.runTrainingStep(step)
            self.doStuffInTrainingLoop(step,trainingEventData)
            
    def runTrainingStep(self,step):
        indexes=np.arange(step*self.batch_size,(step*self.batch_size)+self.batch_size)
        batch_data = self.trainingDataset.take(indexes,mode='wrap',axis=0)
        batch_labels = self.trainingLabels.take(indexes,mode='wrap',axis=0)
        feed_dict = {self.graphStandIn.inputPlaceholder: batch_data,
                     self.graphStandIn.inputLabelPlaceholder: batch_labels}
        _, lossValue, accuracy = self.session.run(
                                    [self.graphStandIn.optimizer, self.graphStandIn.loss,self.graphStandIn.trainAccuracy],
                                    feed_dict=feed_dict)
        return {'loss':lossValue,'offset':indexes[0]%self.trainingDataset.shape[0], 'trainAccuracy':accuracy}
        
    def doStuffInTrainingLoop(self,step,trainTimes,stuffTimes,trainingEventData):
        pass
    
class LearningModelWithMonitoring(LearningModel):
    """
    Same as `LearningModel` but with some fancy features:
     - estimation of remaining run time
     - fixed data logging  and display on certain steps (dictated by self.stepsToLog).
     - additional data logging and display automatically restricted to be below a certain percentage (dictated by self.maxLogFraction, which defaults to 10%) of run time (since last log event).
     - logging and display of default data: step, total number of steps, time remaining, loss, training offset.
    data is stored in the self.trainingRunLog dictionary. Handy hint: pandas.DataFrame(self.trainingRunLog).T return the data in a useful format.
    
    Expects:
        - graphStandIn instance
        - trainingDataset variable
        - trainingLabels variable
        - numberOfTrainingSteps variable
    Optional:
        - logExtraTrainingStats(trainingEventData) method
        - displayExtraTrainingStats() method
        - stepsToLog list/tuple/set
        - maxLogFraction float
        - batch_size int
    Exposes:
        - trainingRunLog - a dictionary 'step':trainingEventData pairs for every step recorded
        
    Run the training by calling `runSession()`
    Run self.trainingRunLog={} before runSession
    All the data logged is in trainingRunLog. Do `pandas.DataFrame(trainingRunLog).T` to get in a nicer form
    Has `calculateStepsToLog` method to get a list of logarithmically distributed numbers
    """
    def calculateStepsToLog(self,maxNumberOfSteps=20):
        maxValue = self.numberOfTrainingSteps
        a=(0.49/(maxValue-1))**(1.0/(maxNumberOfSteps-1))
        l=[maxValue-1]
        while len(l)<=(maxNumberOfSteps-1):
            l[:0]=[l[0]*a] #"prepend"
        steps=map(int,map(round,l))
        return list(OrderedDict.fromkeys(steps))
    
    stepsToLog = []
    maxLogFraction = 0.1
    _lastDuration = 0
    _lastExecutionTime = 0
    _lastExecutionStep = -1
        
    def doStuffInTrainingLoop(self,step,trainingEventData):
        # keep skipping until elapsed time is greater than something times the predicted duration of logAndDisplay function and it has been more than 0.25 seconds since last execution.
        timeSinceLastExecution = time.time()-self._lastExecutionTime
        okToExecute = (self._lastDuration < (self.maxLogFraction/(1.0-self.maxLogFraction))*(timeSinceLastExecution)) and timeSinceLastExecution > 0.25
        if step in self.stepsToLog or okToExecute:
            t = time.time()
            secondsRemaining=((t-self._lastExecutionTime+self._lastDuration)/(step-self._lastExecutionStep)) * (self.numberOfTrainingSteps-1-step)
            self.logAndDisplayTrainingStats(step,trainingEventData,secondsRemaining)
            now = time.time()
            self._lastDuration = now-t
            self._lastExecutionTime = now
            self._lastExecutionStep = step
            
    trainingRunLog = {}
    def logAndDisplayTrainingStats(self,step,trainingEventData,secondsRemaining):
        trainingEventData = self.logExtraTrainingStats(trainingEventData)
        self.trainingRunLog[step] = trainingEventData
        clear_output(wait=True)
        print('   completed step: {} of {}'.format(step,self.numberOfTrainingSteps-1))
        print('   time remaining: {}'.format(datetime.timedelta(seconds=round(secondsRemaining))))
        print('             loss: {}'.format(trainingEventData['loss']))
        #pad to at least 5 characters, round to two decimal points
        print('training accuracy: {:5.2%}'.format(trainingEventData['trainAccuracy']))
        print('  training offset: {}'.format(trainingEventData['offset']))
        self.displayExtraTrainingStats()
    def logExtraTrainingStats(self,trainingEventData):
        #trainingEventData['somethingUseful'] = someComputation()
        return trainingEventData
    def displayExtraTrainingStats(self):
        # draw a graph or print omething
        pass



class GraphStandInBase():
    """
    An object that describes, creates and contains a tensor flow graph.
    
    This is the base class that contains many possible graph elements written as methods.
    Subclass this and change the init method to actually build a graph.
    The following attributes are expected by the SGD method: inputPlaceholder, inputLabelPlaceholder, loss, optimizer.
    Assign tensors/ops to self to be able to access them after training.
    Alternatively assign them tensor flow name atributes and use tensor flow to access them afterwards.
    """
    def __init__(self,inputDataShape,numLabels):
        """create a graph"""
        self.graph = tf.Graph()
        with self.graph.as_default():
            pass
    def returnInputPlaceholder(self,inputDataShape):
        return tf.placeholder(tf.float32,shape=(None,)+tuple(inputDataShape))
    def returnInputLabelPlaceholder(self,numLabels):
        return tf.placeholder(tf.float32,shape=(None,numLabels))
    def returnModelLogitFunc(self,inputDataShape,numLabels):
        pass
    def returnLoss(self,logits,labels):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))
    def returnOptimizer(self,loss,learnRate=0.01,globalStep = None):
        if globalStep == None:
            return tf.train.GradientDescentOptimizer(learnRate).minimize(loss)
        else:
            return tf.train.GradientDescentOptimizer(learnRate).minimize(loss,global_step=globalStep)
    def returnAccuracy(self,predictions,labels):
        totalCorrect = tf.reduce_sum(tf.to_int32(tf.equal(tf.argmax(predictions,1),tf.argmax(labels,1))))
        return tf.div(tf.to_float(totalCorrect),tf.to_float(tf.shape(predictions)[0]))
    
class LogisticClassifierGraph(GraphStandInBase):
    def __init__(self,inputDataShape,numLabels):
        """create a graph"""
        self.graph = tf.Graph()
        with self.graph.as_default():
            # add in self.'s onto these variables if I want to be able to access them outside outside of this graph building step.
            # self.optimizer is required others might be
            # alternatively name the ops here and use the name to find them in tensor flow afterwards.
            self.inputPlaceholder = self.returnInputPlaceholder(inputDataShape)
            self.inputLabelPlaceholder = self.returnInputLabelPlaceholder(numLabels)
            
            modelLogitFunc = self.returnModelLogitFunc(inputDataShape,numLabels)
            logits = modelLogitFunc(self.inputPlaceholder)
            self.prediction = tf.nn.softmax(logits)
            
            self.loss = self.returnLoss(logits,self.inputLabelPlaceholder)
            self.optimizer = self.returnOptimizer(self.loss)
            
            self.trainAccuracy = self.returnAccuracy(self.prediction,self.inputLabelPlaceholder)
            
    def returnModelLogitFunc(self,inputDataShape,numLabels):
        weights = tf.Variable(tf.truncated_normal(tuple(inputDataShape)+(numLabels,), stddev=0.1)) #this seems to work
        biases = tf.Variable(tf.zeros([numLabels]))
        def modelLogitFunc(data):
            return tf.matmul(data,weights)+biases
        return modelLogitFunc
    
class DeepClassifierGraph(GraphStandInBase):
    def __init__(self,inputDataShape,numLabels,hiddenUnits=[],dropoutKeepProbability=0.5,initalLearnRate=0.1,learnRateHalfLife=np.inf,l2NormParameter=0.0):
        """
        inputDataShape - tuple/list indicating the shape of each data sample in the training/test data
        numLabels - an integer indicating the number of classes
        hiddenUnits - a list of integers, each integer indicating the number of hidden units in that layer
        """
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputPlaceholder = self.returnInputPlaceholder(inputDataShape)
            self.inputLabelPlaceholder = self.returnInputLabelPlaceholder(numLabels)
            step = tf.Variable(0)
            
            modelLogitFunc = self.returnModelLogitFunc(inputDataShape,numLabels,hiddenUnits)
            logits = modelLogitFunc(self.inputPlaceholder,dropoutKeepProbability)
            
            self.prediction = tf.nn.softmax(modelLogitFunc(self.inputPlaceholder,dropoutKeepProbability=1.0))
            
            self.loss = tf.add(self.returnLoss(logits,self.inputLabelPlaceholder),self.returnl2Norm(self.weightsMatricesList,l2NormParameter))
            self.learnRate = self.returnAdjustedLearnRate(initalLearnRate,step,learnRateHalfLife)
            self.optimizer = self.returnOptimizer(self.loss,self.learnRate,globalStep=step)
            
            self.trainAccuracy = self.returnAccuracy(self.prediction,self.inputLabelPlaceholder)
    
    def returnModelLogitFunc(self,inputDataShape,numLabels,hiddenUnits):
        ### Variables.
        self.weightsMatricesList = []
        self.biasesMatricesList = []
        for n,m in self._pairwise([reduce(lambda x, y: x*y, inputDataShape)]+hiddenUnits+[numLabels]):
            self.weightsMatricesList.append(tf.Variable(tf.truncated_normal([n,m])))
            self.biasesMatricesList.append(tf.Variable(tf.zeros([m])))
    
        def modelLogitFunc(data,dropoutKeepProbability):
            logitsList=[]
            logitsList.append(tf.matmul(data, self.weightsMatricesList[0]) + self.biasesMatricesList[0])
            for i in range(1,len(self.weightsMatricesList)):
                logitsList.append(tf.matmul(tf.nn.dropout(tf.nn.relu(logitsList[i-1]),dropoutKeepProbability), self.weightsMatricesList[i]) + self.biasesMatricesList[i])
            return logitsList[-1]
        return modelLogitFunc
    def returnl2Norm(self,weightMatrices,hyperParameter):
        return tf.mul(hyperParameter,tf.reduce_sum(tf.pack(map(tf.nn.l2_loss,weightMatrices))))
    def returnAdjustedLearnRate(self,initial_learn_rate,globalStep,learn_rate_half_life):
        return tf.train.exponential_decay(initial_learn_rate,globalStep,learn_rate_half_life,0.5)
    def _pairwise(self,iterable):
        """
        iterable: [s0,s1,s2,s3,s4,...]
        returns: (s0, s1), (s1, s2), (s2, s3), ...
        """
        from itertools import izip
        a = iter(iterable)
        b = iter(iterable)
        b.next()
        return izip(a, b)
