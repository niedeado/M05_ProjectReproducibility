import database
import algorithm
import analysis

if __name__ == "__main__":

    dataset = database.load()
    X, y, labels_inv_map, _ = database.extract_data_array(dataset)
    X_train, X_test, y_train, y_test = database.split_data(X,y)

    model = algorithm.train(X_train, y_train)
    
    y_pred_test = model.predict(X_test)
    print("Test classification report:")
    report = analysis.visualize_report(y_test, y_pred_test, labels_inv_map)
    print(report)

    print("\nMisclassification inspection:")
    misclassified_msg = analysis.inspect_misclassified(y_test, y_pred_test, labels_inv_map)
    for msg in misclassified_msg:
        print(msg)

    print("\nTraining accuracy:", model.score(X_train, y_train))
    print("Test accuracy:", model.score(X_test, y_test))