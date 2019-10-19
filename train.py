import model as M
PATH_TO_IMAGES = "/home/hossam/Projects/UnCertainity_Estimation/data/images/"
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 0.01

preds, aucs = M.train_cnn(PATH_TO_IMAGES, LEARNING_RATE, WEIGHT_DECAY)