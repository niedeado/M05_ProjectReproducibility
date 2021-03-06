"""Main script to be launched from the command line"""
from . import database
from . import algorithm
from . import analysis
import pytest
import os
import pkg_resources


def main():
    """Function called from the command line"""

    import argparse

    example_doc = """\
examples:
    1. Outputs classification report, misclassifications and score (no hyperparam tuning):
       $ python pyfiles\main_script.py 
    2. Performs hyperparam tuning and outputs score only:
       $ python pyfiles\main_script.py -t=1 -o=001
    3. Performs hyperparam tuning (and pickle dumps best ones)
       and outputs classification report and score:
       $ python paper.py --tuning=2 --output=101
    """

    parser = argparse.ArgumentParser(
        usage="python %(prog)s [options]",
        description="Performs Plant Species Classification based on Shape features",
        epilog=example_doc,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-t",
        "--tuning",
        choices=[0, 1, 2],
        type=int,
        default=0,
        help="Determines whether to conduct hyperparameter tuning or not "
        "If you choose '0', then it does not perform hyperparameter "
        "tuning.  If you choose '1', then it performs hyperparameter "
        "tuning.  If you choose '2', then it performs hyperparameter "
        "tuning and pickle dumps best hyperparameters.",
    )

    parser.add_argument(
        "-o",
        "--output",
        choices=["000", "001", "010", "011", "100", "101", "110", "111"],
        type=str,
        default="111",
        help="Determines the script output based on a string of "
        "three binary digits, e.g. '110'. If the first character "
        "is '1', then it outputs a classification report, otherwise "
        "it does not. If the second character is '1', then it "
        "outputs misclassified species, otherwise it does not. "
        "If third character is '1', then it outputs the score, "
        " otherwise it does not.",
    )

    parser.add_argument(
        "-n",
        "--notebook",
        choices=[False, True],
        type=bool,
        default=False,
        help="Determines whether to open jupyter notebook with the file "
        "for visualize the data "
        "If you choose True, the regarding jupyter notebook file will "
        "be opened, otherwise not.",
    )

    args = parser.parse_args()

    dataset = database.load()
    X, y, labels_inv_map, _ = database.extract_data_array(dataset)
    X_train, X_test, y_train, y_test = database.split_data(X, y)

    if args.tuning == 0:
        model = algorithm.train(X_train, y_train)
    elif args.tuning == 1:
        model = algorithm.train(X_train, y_train, hyperparameters=None)
    elif args.tuning == 2:
        model = algorithm.train(
            X_train, y_train, hyperparameters=None, pickle_dump=True
        )

    y_pred_test = model.predict(X_test)

    if args.output[0] == "1":
        print("Test classification report:")
        report = analysis.visualize_report(y_test, y_pred_test, labels_inv_map)
        print(report)

    if args.output[1] == "1":
        print("\nMisclassification inspection:")
        misclassified_msg = analysis.inspect_misclassified(
            y_test, y_pred_test, labels_inv_map
        )
        for msg in misclassified_msg:
            print(msg)

    if args.output[2] == "1":
        print("\nTraining accuracy:", model.score(X_train, y_train))
        print("Test accuracy:", model.score(X_test, y_test))

    if args.notebook:
        IPYNB_FILE = pkg_resources.resource_filename(
            __name__, "./notebooks/VisualWidget.ipynb"
        )
        os.system(f"jupyter notebook {IPYNB_FILE}")

    return True


def main_test():
    """Function called from the command line to run all existing tests in this package"""

    FILE_1 = pkg_resources.resource_filename(__name__, "test_visualisation.py")
    FILE_2 = pkg_resources.resource_filename(__name__, "test_modeling.py")
    pytest.main(["-x", FILE_1, "-vv"])
    pytest.main(["-x", FILE_2, "-vv"])


if __name__ == "__main__":
    main()
