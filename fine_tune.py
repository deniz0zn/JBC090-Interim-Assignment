from sklearn.model_selection import GridSearchCV
from joblib import dump, load
from config import param_grid, __DEBUG__, CV
import os

def fine_tune(model,model_path, vectorizer , vectorizer_path,
              X_train, y_train,param_grid = param_grid,
              cv=CV , DEBUG = __DEBUG__):
    """
    Fine-tune a model with grid search and save the fine-tuned model and vectorizer.
    """
    if DEBUG and os.path.exists(model_path) and os.path.exists(vectorizer_path):
        print(f"Loading fine-tuned model from {model_path} and vectorizer from {vectorizer_path}...")
        best_model = load(model_path)
        vectorizer = load(vectorizer_path)
        return best_model, vectorizer, None

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=cv,
        n_jobs=-1,
        verbose=10
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)

    dump(best_model, model_path)
    dump(vectorizer, vectorizer_path)

    print(f"Fine-tuned model saved to {model_path}.")
    print(f"Vectorizer saved to {vectorizer_path}.")

    return best_model, grid_search.best_params_


# def fine_tune_transformer(trainer, train_dataset, eval_dataset, output_dir="./results", save_path="transformer_model.pt", DEBUG = __DEBUG__):
#     """
#     Fine-tunes a transformer-based model using the HuggingFace Trainer and saves the fine-tuned model.
#     """
#     if DEBUG:
#         print(f"Loading fine-tuned transformer model from {save_path}...")
#         trainer.model = torch.load(save_path)
#         return trainer
#
#     print("Fine-tuning transformer model...")
#     training_args = TrainingArguments(
#         output_dir=output_dir,
#         evaluation_strategy="epoch",
#         save_strategy="epoch",
#         learning_rate=5e-5,
#         per_device_train_batch_size=16,
#         num_train_epochs=3,
#         weight_decay=0.01,
#     )
#
#     trainer.args = training_args
#     trainer.train()
#
#     # Save the model
#     torch.save(trainer.model, save_path)
#     print(f"Fine-tuned transformer model saved to {save_path}.")
#     return trainer
