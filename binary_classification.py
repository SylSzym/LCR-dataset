import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

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

    def evaluate_classifier(self):
        self.clf.fit(self.X_train, self.y_train)
        self.y_pred = self.clf.predict(self.X_test)
        accuracy = precision_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred)
        auc = roc_auc_score(self.y_test, self.y_pred)

        print('Accuracy:', accuracy)
        print('Recall:', recall)
        print('F1-score:', f1)
        print('AUC:', auc)

if __name__ == "__main__":
    clf = RandomForestClassifier(n_estimators=1500, max_depth=100)

    lcr_dataset = LCRDataset('sample_data/X_train_embeddings_binary.csv', 'sample_data/X_test_embeddings_binary.csv')
    X_train, y_train, X_test, y_test = lcr_dataset.preprocess_data()

    classifier_evaluation = ClassifierEvaluation(clf, X_train, y_train, X_test, y_test)
    classifier_evaluation.evaluate_classifier()