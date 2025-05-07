"""This is full life cycle for ml model"""
import argparse

TRAIN_SIZE = 0.2
MODEL_NAME = []

def parse_cian():
    """Parse data to data/raw"""
    pass

def preprocess_data():
    """Filter and remove """
    pass

def train_model():
    pass

def test_model():
    pass


if __name__ == '__main__':
    """Parse arguments and run lifecycle steps"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split', type=float, 
                        help='Split data, test relative size, from 0 to 1',
                        default=TRAIN_SIZE)
    parser.add_argument('-m', '--model', 
                        help='Model name', 
                        default=MODEL_NAME)
    args = parser.parse_args()
    
    parse_cian()
    preprocess_data()
    train_model()
    test_model()