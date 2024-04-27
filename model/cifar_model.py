# Implement a specific model for the CIFAR dataset

from .base_model import BaseModel
from tensorflow.keras import models, layers
from configs.cifar_config import CIFARConfig

class CIFARModel(BaseModel):
    def __init__(self):
        self.model = self.build_model()
    
    def build_model(self):
        model = models.Sequential()
        # Add layers to the model
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=CIFARConfig.image_size))
        model.add(layers.Flatten()) 
        # Continue adding layers...
        model.add(layers.Dense(CIFARConfig.num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train, batch_size=CIFARConfig.batch_size, epochs=10)
    
    def predict(self, x):
        return self.model.predict(x)
