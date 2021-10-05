from . import database
from . import algorithm
from . import analysis

def main():
    dataset = database.load()
    X, y, labels_inv_map = database.extract_data_array(dataset)
    X_train, X_test, y_train, y_test = database.split_data(X, y)

    model = algorithm.train(X_train, y_train)

    y_pred_test = algorithm.predict(model, X_test)
    print("Test classification report:")
    analysis.visualize_report(y_test, y_pred_test, labels_inv_map)

    print("\nMisclassification inspection:")
    analysis.inspect_misclassified(y_test, y_pred_test, labels_inv_map)

    print("\nTraining accuracy:", algorithm.score(model, X_train, y_train))
    print("Test accuracy:", algorithm.score(model, X_test, y_test))



if __name__ == "__main__":
    main()

