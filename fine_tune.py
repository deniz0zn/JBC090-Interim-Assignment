from sklearn.model_selection import GridSearchCV
from transformers import Trainer, TrainingArguments
from joblib import dump, load
from config import param_grid
import os
import torch


def fine_tune_logistic(model, X_train, y_train, param_grid = param_grid, cv=5, save_path="logistic_model.pkl"):
    """
    Fine-tunes a logistic regression model using GridSearchCV and saves the best model.
    """
    if os.path.exists(save_path):
        print(f"Loading fine-tuned logistic model from {save_path}...")
        best_model = load(save_path)
        return best_model, None

    print("Fine-tuning logistic regression model...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=cv,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)

    dump(best_model, save_path)
    print(f"Fine-tuned logistic model saved to {save_path}.")
    return best_model, grid_search.best_params_


def fine_tune_transformer(trainer, train_dataset, eval_dataset, output_dir="./results", save_path="transformer_model.pt"):
    """
    Fine-tunes a transformer-based model using the HuggingFace Trainer and saves the fine-tuned model.
    """
    if os.path.exists(save_path):
        print(f"Loading fine-tuned transformer model from {save_path}...")
        trainer.model = torch.load(save_path)
        return trainer

    print("Fine-tuning transformer model...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer.args = training_args
    trainer.train()

    # Save the model
    torch.save(trainer.model, save_path)
    print(f"Fine-tuned transformer model saved to {save_path}.")
    return trainer
