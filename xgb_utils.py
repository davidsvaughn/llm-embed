import numpy as np
import torch
import xgboost as xgb
import time
from sklearn.model_selection import KFold, cross_val_predict
from kappa import ikappa
from ddp_utils import is_main
from utils import clear_cuda_tensors


def run_xgb_on_item(embedder, item, records, preds_file=None, K=5, step=None, random_state=42, **kwargs):
    """
    Runs XGBoost regression on embeddings for a specific item with cross-validation or train/valid split.
    Args:
        embedder: An embedder object that computes embeddings for the input records.
        item: The specific item identifier to process.
        records: Collection of records containing the data to process.
        preds_file (str, optional): File path to save predictions. Only used with K-fold CV. Defaults to None.
        K (int, optional): Number of folds for cross-validation. If K=1, uses train/valid split. Defaults to 5.
        step (int, optional): Training step number for logging purposes. Defaults to None.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.
        **kwargs: Additional keyword arguments passed to embedder.compute_embeddings().
    Returns:
        float: Quadratic weighted kappa score (QWK) of the predictions.
    Notes:
        - Uses pre-defined best parameters for XGBoost model.
        - If K>1, performs K-fold cross-validation and can save predictions to file.
        - If K=1, uses original train/valid split from the records.
        - Handles distributed training environment with torch.distributed.
        - Prints timing information and QWK scores for monitoring.
        - Automatically manages memory by clearing large data structures.
    """
    
    best_params = {'learning_rate': 0.05, 'n_estimators': 200, 'max_depth': 3, 
                  'min_child_weight': 1, 'subsample': 0.3, 'colsample_bytree': 0.9}
    
    item_start = time.time()
    
    # Compute embeddings just for this item
    emb_data = embedder.compute_embeddings({item: records}, **kwargs)
    
    # Clear cuda memory
    clear_cuda_tensors()
    
    if is_main():
        # Run XGBoost for this item
        print(f"Time to compute embeddings for item {item}: {time.time()-item_start:.2f} seconds")
        xgb_start = time.time()
        
        # get embeddings
        X = emb_data[item]['x']
        y = emb_data[item]['y']
        idx = emb_data[item]['idx']
        
        # Clear memory
        del emb_data[item]['x']
        del emb_data[item]['y']
        del emb_data[item]['idx']
        del emb_data
        
        # Run XGBoost
        xgb_mod = xgb.XGBRegressor(objective='reg:squarederror', 
                                    eval_metric='rmse', 
                                    verbosity=0, 
                                    **best_params)
        
        # use K-fold cross-validation
        if K>1:
            kfold = KFold(n_splits=K, shuffle=True, random_state=random_state)
            y_pred = cross_val_predict(xgb_mod, X, y, cv=kfold, n_jobs=-1)
            qwk = ikappa(y, y_pred)
            
        # use originl train/valid split
        else:
            train_idx, valid_idx = [], []
            # get train/valid indices
            for rec in records:
                if rec.split.startswith('train'): train_idx.append(rec.index)
                elif rec.split.startswith('valid'): valid_idx.append(rec.index)
            train_idx, valid_idx = np.array(train_idx), np.array(valid_idx)
            X_train = X[np.in1d(idx, train_idx)]
            y_train = y[np.in1d(idx, train_idx)]
            X_valid = X[np.in1d(idx, valid_idx)]
            y_valid = y[np.in1d(idx, valid_idx)]
            
            xgb_mod.fit(X_train, y_train)
            y_pred = xgb_mod.predict(X_valid)
            qwk = ikappa(y_valid, y_pred)
        
        print(f"Time to run XGBoost for item {item}: {time.time()-xgb_start:.2f} seconds")
        print(f"Total time for item {item}: {time.time()-item_start:.2f} seconds")
        
        # save K-fold predictions for entire dataset
        if preds_file and K>1:
            # print(f"QWK = {qwk:.4f}")
            preds_table = np.column_stack((idx, y, y_pred.round().astype(int)))
            # sort table by first column
            preds_table = preds_table[preds_table[:,0].argsort()]
            np.savetxt(preds_file, preds_table, delimiter=',', fmt='%d,%d,%d')
            print(f"Predictions saved to {preds_file}")

        # Clear memory
        del X, y, y_pred, xgb_mod, idx

    else:
        qwk = None

    if torch.distributed.is_initialized():
        torch.distributed.barrier(device_ids=[torch.cuda.current_device()])  # Wait for rank 0 to finish
    
    if is_main():
        if step is not None:
            print(f"QWK_{item}\t{step}\t{qwk:.4f}")
        else:
            print(f"QWK_{item}\t{qwk:.4f}")

    return qwk


def run_xgb_on_items(embedder, data_by_item, **kwargs):
    """Runs XGBoost model training and evaluation for multiple items in the dataset.
    Args:
        embedder: Text embedding model used to convert text to vectors.
        data_by_item (dict): Dictionary containing data records grouped by item.
        **kwargs: Additional keyword arguments to pass to run_xgb_on_item function.
    Returns:
        numpy.ndarray or None: Array of quadratic weighted kappa scores rounded to 4 decimal places 
        for each successfully processed item if running in main process, None otherwise.
        Returns None if processing is interrupted by KeyboardInterrupt (Ctrl+C).
    Notes:
        - Processes items sequentially and collects QWK scores for each item
        - Prints progress updates when running in main process
        - only main process (rank0) returns the array of QWK scores
        - Gracefully handles Ctrl+C interruptions
    """

    # use random_state to generate a sequence of random states, one for each item
    random_state = kwargs.pop('random_state', 42)
    random_states = np.random.RandomState(random_state).randint(0, 2**32-1, len(data_by_item))
    
    try:
        qwks = []
        for i, (item, records) in enumerate(data_by_item.items()):
            if is_main():
                print(f"\nProcessing item: {item} ({i+1}/{len(data_by_item)})")
            
            qwk = run_xgb_on_item(embedder, 
                                  item, 
                                  records,
                                  random_state=random_states[i], 
                                  **kwargs)
            
            if qwk is not None:
                # only rank0 has qwk != None
                qwks.append(qwk)

        if is_main():
            return np.array(qwks).round(4)
        else:
            return None
    except KeyboardInterrupt:
        if is_main():
            print("\nProcessing interrupted by user (Ctrl+C). Returning None.")
        return None