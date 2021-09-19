import database
import algorithm
import analysis

if __name__ == "__main__":

    dataset = database.load()
    X, y, labels_inv_map = database.extract_data_array(dataset)
    X_train, X_test, y_train, y_test = database.split_data(X,y)

    model = algorithm.train(X_train, y_train)
    
    