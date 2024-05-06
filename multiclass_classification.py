import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize

class LCRDataset:
    def __init__(self, train_csv_path, test_csv_path):
        self.data_train = pd.read_csv(train_csv_path, sep=';')
        self.data_test = pd.read_csv(test_csv_path, sep=';')

    def preprocess_data(self):
        X_train = self.data_train.iloc[:,:-6]
        y_train = self.data_train['labels']
        X_test = self.data_test.iloc[:, :-6]
        y_test = self.data_test['labels']
        return X_train, y_train, X_test, y_test

class ClassifierEvaluation:
    def __init__(self, clf, X_train, y_train, X_test, y_test):
        self.clf = clf
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = None
        self.ytest = None
        self.ypred = None

    def evaluate_classifier(self):
        self.clf.fit(self.X_train, self.y_train)
        self.y_pred = self.clf.predict(self.X_test)
        accuracy = precision_score(self.y_test, self.y_pred, average='macro', zero_division=0)
        recall = recall_score(self.y_test, self.y_pred, average='macro', zero_division=0)
        f1 = f1_score(self.y_test, self.y_pred, average='macro', zero_division=0)
        # You need the labels to binarize
        labels = [0, 1, 2, 3, 4, 5, 6, 7]
        # Binarize ytest with shape (n_samples, n_classes)
        self.ytest = label_binarize(self.y_test, classes=labels)
        # Binarize ypreds with shape (n_samples, n_classes)
        self.ypreds = label_binarize(self.y_pred, classes=labels)
        auc = roc_auc_score(self.ytest, self.ypreds, average='macro', multi_class='ovo')


        print('Overall Accuracy with Standard Deviation:', accuracy)
        print('Overall Recall with Standard Deviation:', recall)
        print('Overall F1-score with Standard Deviation:', f1)
        print('Overall AUC with Standard Deviation', auc)

if __name__ == "__main__":
    clf = RandomForestClassifier(n_estimators=1500, max_depth=100)

    lcr_dataset = LCRDataset('sample_data/X_train_embeddings_multiclass.csv', 'sample_data/X_test_embeddings_multiclass.csv')
    X_train, y_train, X_test, y_test = lcr_dataset.preprocess_data()

    classifier_evaluation = ClassifierEvaluation(clf, X_train, y_train, X_test, y_test)
    classifier_evaluation.evaluate_classifier()