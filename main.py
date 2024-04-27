# The entry point for training and evaluating the model

from dataloader.cifar_dataloader import CIFARDataloader
from model.cifar_model import CIFARModel

def main():
    # Load the data
    dataloader = CIFARDataloader()
    (x_train, y_train), (x_test, y_test) = dataloader.load_data()
    
    # Initialize and train the model
    model = CIFARModel()
    model.train(x_train, y_train)
    
    # Evaluate the model
    loss, accuracy = model.model.evaluate(x_test, y_test)
    print(f"Test accuracy: {accuracy*100:.2f}%")

if __name__ == '__main__':
    main()