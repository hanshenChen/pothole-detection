from keras.layers import Input
from keras.optimizers import SGD, Adam,RMSprop,Nadam
from keras.models import Model
from .myresnet50 import share_layers,share_layers2,densitymap_layers,classifier_layers

def create_resnet50(input_shape=(192, 256,1)):
    input_layer = Input(shape=input_shape)
    shared_feature= share_layers(input_layer)
    classifier=classifier_layers(shared_feature)
    opt = Adam(0.0001)#,decay=0.0001)
    model_classifier =Model(input_layer,classifier)
    model_classifier.compile(optimizer=opt,loss='binary_crossentropy', metrics=['acc'])#metrics.binary_accuracy])
    return model_classifier

def create_lcnn_model(input_shape=(192, 256,1)):
    input_layer = Input(shape=input_shape)
    shared_feature=share_layers2(input_layer)
    densitymap=densitymap_layers(shared_feature)
    model_lcnn=Model(input_layer,densitymap)
    model_lcnn.summary()
    opt = Adam(0.0001)#5)
    model_lcnn.compile(optimizer=opt, loss='mse',metrics=['mse'])
    return model_lcnn
