from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score)
import matplotlib.pyplot as plt
import seaborn as sns

def train_model(X_train, y_train):
    model = MultinomialNB(alpha=0.5)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, digits=4)
    
    # Ambil accuracy dari classification report untuk konsistensi
    accuracy = report['accuracy']
    
    metrics = {
     'confusion_matrix': cm,
        'classification_report': report,
        'classes': model.classes_,
        'accuracy': report['accuracy'],
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1': report['weighted avg']['f1-score'],
        'macro_precision': report['macro avg']['precision'],
        'macro_recall': report['macro avg']['recall'],
        'macro_f1': report['macro avg']['f1-score']
    }
    return metrics
