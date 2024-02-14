import pandas
import numpy
import random
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from matplotlib.pyplot import figure

# ---------------------------------------------------------------------------------------------------------------------
# EXPERIMENT PARAMETERS
# ---------------------------------------------------------------------------------------------------------------------
# 1. Directory and file-related information.
WORKING_DIRECTORY = '/Users/amrit/Downloads/'
DATA_FILE_NAME = 'final_data_matrix_with_labels.csv'
DATA_FILE_PATH = os.path.join(WORKING_DIRECTORY, DATA_FILE_NAME)
ROC_FILE_CURVE = os.path.join(WORKING_DIRECTORY, 'roc_curve.png')

# 2. Hyperparameters for the random forest.
PARAM_GRID = {
    'n_estimators': [2 ** i for i in range(12)],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [2 ** i for i in range(7)] + [None],
    'max_leaf_nodes': [1, 3, 5, 9, 11, None],
}
BEST_CLASSIFIER = {'n_estimators': 512, 'max_features': 'sqrt', 'max_depth': None, 'max_leaf_nodes': None}

# 3. We report the average result over 100 distinct initializations of the random forest for each setting. Set initial
# seed to be 0 for replicability. Then, randomly select 100 values to seed (randomness with which to initialize) the
# random forest.
SEED = 0
random.seed(SEED)
NO_OF_SEEDS = 100
SEED_LIST = random.sample(range(100_000), NO_OF_SEEDS)

# 4. Default train and test labels. TODO: Add description of what [1, 2, 3, 4] correspond to.
DEFAULT_TRAIN_LABELS = [3, 4]
DEFAULT_TEST_LABELS = [1, 2]


# ---------------------------------------------------------------------------------------------------------------------


def create_data_matrix():
    """
    We use this function to create the final data matrix with labels file. We save the file after these steps were
    implemented and directly use the saved file.

    TODO: Add description of:
     1. Columns in input_x.
     2. corrected_y has 4 labels = [1, 2, 3, 4] - Write out what each one is in words.
    """
    input_x = pandas.read_csv(WORKING_DIRECTORY + 'Gene_expression_matrix_hum_fish.csv')

    # Input matrix has cell has columns and features as rows. Take transpose to put in form that models is expecting.
    column_names = ['cell_bar_code'] + input_x['Unnamed: 0'].tolist()

    # 1st row in the input matrix has cell bar codes. Extract them, start from 2nd row and use 1st row as column name.
    input_x = input_x.T.reset_index().iloc[1:].reset_index(drop=True)
    input_x.columns = column_names

    # Correcting column dtype.
    column_list = [i for i in input_x.columns if i not in ['cell_bar_code']]
    input_x[column_list] = input_x[column_list].astype(float)

    # Add y label to input_x
    input_y = pandas.read_csv(WORKING_DIRECTORY + 'ids_merge_zfhs_cart_only.csv')
    input_x['corrected_y'] = input_y['x']

    # Set y = 0 (Hyaline), 1 (Elastic).
    input_x.loc[input_x.index, 'y'] = 0
    input_x.loc[input_x[input_x['corrected_y'].isin([2, 4])].index, 'y'] = 1

    input_x.to_csv(DATA_FILE_PATH, index=False)


class CartilageClassifier:
    def __init__(self, train_labels=None, test_labels=None):
        # Read saved input data with features and labels.
        self.data_matrix = pandas.read_csv(DATA_FILE_PATH)

        # Remove y-related columns and bar code to prevent label leakage.
        exclude = ['cell_bar_code', 'y_true', 'y_pred', 'Correct', 'Incorrect', 'corrected_y', 'y']
        self.column_list = [i for i in self.data_matrix.columns if i not in exclude]

        # Split into train and test sets - Note that train and test sets have NO overlapping labels!
        self.train_labels, self.test_labels = self.get_test_and_train_labels(train_labels, test_labels)
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_into_train_and_test()

        # Initialize empty lists where we store results.
        self.train_list = []
        self.test_list = []

    @staticmethod
    def get_test_and_train_labels(train_labels, test_labels):
        # Specify default test and training labels. 
        if train_labels is None:
            train_labels = DEFAULT_TRAIN_LABELS

        if test_labels is None:
            test_labels = DEFAULT_TEST_LABELS

        test_train_intersection = set.intersection(set(train_labels), set(test_labels))
        assert len(test_train_intersection) == 0, 'Train and test labels should be disjoint. Please check.'

        return train_labels, test_labels

    def split_into_train_and_test(self):
        # Split into test and train using species.
        train_mask = (self.data_matrix['corrected_y'].isin(self.train_labels))
        test_mask = (self.data_matrix['corrected_y'].isin(self.test_labels))
        x_train = self.data_matrix.loc[self.data_matrix[train_mask].index, self.column_list]
        x_test = self.data_matrix.loc[self.data_matrix[test_mask].index, self.column_list]
        y_train = self.data_matrix.loc[self.data_matrix[train_mask].index, 'y']
        y_test = self.data_matrix.loc[self.data_matrix[test_mask].index, 'y']
        return x_train, x_test, y_train, y_test

    def run_single_instance_of_rf(self, seed, n_estimators, max_features, max_depth, max_leaf_nodes):
        model = RandomForestClassifier(random_state=seed, n_estimators=n_estimators, max_features=max_features,
                                       max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)
        model.fit(self.x_train, self.y_train)
        train_score = model.score(self.x_train, self.y_train)
        test_score = model.score(self.x_test, self.y_test)
        return train_score, test_score, model

    def grid_search(self, splits=5, repeats=5, verbose=1):
        cv = RepeatedStratifiedKFold(n_splits=splits, n_repeats=repeats, random_state=SEED)

        estimator = RandomForestClassifier(random_state=SEED)

        model = GridSearchCV(estimator=estimator, param_grid=PARAM_GRID, verbose=verbose, cv=cv,
                             scoring=['accuracy'], refit='accuracy', n_jobs=-1)

        # Fit model
        model.fit(self.x_train, self.y_train)

        # Save GridSearchCV results.
        results_df = pandas.DataFrame(model.cv_results_)
        results_path = os.path.join(WORKING_DIRECTORY, 'random_forest_cross_validation.csv')
        results_df.to_csv(results_path, index=False, encoding='utf-8')

    def find_test_accuracy(self):
        for seed in SEED_LIST:
            _train, _test, _ = self.run_single_instance_of_rf(seed, **BEST_CLASSIFIER)
            self.train_list.append(_train)
            self.test_list.append(_test)

        train_mean = round(numpy.mean(self.train_list) * 100.0, 2)
        train_std = round(numpy.std(self.train_list) * 100.0, 2)
        test_mean = round(numpy.mean(self.test_list) * 100.0, 2)
        test_std = round(numpy.std(self.test_list) * 100.0, 2)
        print(f'Train = {train_mean}% ({train_std}%), Test = {test_mean}% ({test_std}%)')

    def save_results(self):
        with open(os.path.join(WORKING_DIRECTORY, 'accuracy_of_individual_runs.json'), 'w') as fp:
            json.dump({'train': self.train_list, 'test': self.test_list, 'seeds': SEED_LIST}, fp)

    def plot_roc_curve(self):
        figure(figsize=(8, 6), dpi=400)
        ax = plt.gca()
        _, _, model = self.run_single_instance_of_rf(seed=SEED, **BEST_CLASSIFIER)
        _ = RocCurveDisplay.from_estimator(model, self.x_test, self.y_test, ax=ax)
        plt.plot([0, 1], [0, 1], linestyle='--', label='Random guessing')
        plt.legend()

        plt.savefig(ROC_FILE_CURVE, dpi=400)
        plt.show()


if __name__ == '__main__':
    c = CartilageClassifier()
    c.find_test_accuracy()
    c.plot_roc_curve()
