import time
from pathlib import Path
import torch
import numpy as np
from alg.opt import *
from alg import alg, modelopera
from utils.util import set_random_seed, get_args, print_row, print_args, train_valid_target_eval_names, alg_loss_dict, print_environ
from datautil.getdataloader_single import get_act_dataloader

def main(args):
    s = print_args(args, [])
    set_random_seed(args.seed)
    if args.dataset == 'emg':
        args.num_classes = 6
    else:
        args.num_classes = 36
    print_environ()
    print(s)
    
    # Data verification before training - FIXED
    print("Verifying data shapes and ranges:")
    train_loader, _, _, _, _, _, _ = get_act_dataloader(args)
    for i, batch in enumerate(train_loader):
        inputs = batch[0]
        labels = batch[1]
        
        print(f"Batch {i}: Data shape={inputs.shape}, Min={inputs.min():.4f}, Max={inputs.max():.4f}")
        print(f"Labels: {torch.unique(labels, return_counts=True)}")
        if i == 2:  # Check first 3 batches
            break

    if args.latent_domain_num < 6:
        args.batch_size = 32*args.latent_domain_num
    else:
        args.batch_size = 16*args.latent_domain_num

    train_loader, train_loader_noshuffle, valid_loader, target_loader, _, _, _ = get_act_dataloader(args)

    # Calculate class weights - FIXED TYPE CONVERSION
    all_labels = []
    for batch in train_loader:
        # Convert labels to integers
        int_labels = batch[1].to(torch.long)
        all_labels.append(int_labels)
        
    all_labels = torch.cat(all_labels, dim=0)
    class_counts = np.bincount(all_labels.numpy())
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.cuda()
    print(f"Class weights: {class_weights}")

    best_valid_acc, target_acc = 0, 0
    best_model_state = None
    patience = 10
    patience_counter = 0

    algorithm_class = alg.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(args, class_weights=class_weights).cuda()
    algorithm.train()
    optd = get_optimizer(algorithm, args, nettype='Diversify-adv')
    opt = get_optimizer(algorithm, args, nettype='Diversify-cls')
    opta = get_optimizer(algorithm, args, nettype='Diversify-all')
    
    # Add learning rate schedulers
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=0.5, patience=5, verbose=True
    )

    for round in range(args.max_epoch):
        print(f'\n========ROUND {round}========')
        print('====Feature update====')
        loss_list = ['class']
        print_row(['epoch']+[item+'_loss' for item in loss_list], colwidth=15)

        for step in range(args.local_epoch):
            for data in train_loader:
                loss_result_dict = algorithm.update_a(data, opta)
            print_row([step]+[loss_result_dict[item]
                              for item in loss_list], colwidth=15)

        print('====Latent domain characterization====')
        loss_list = ['total', 'dis', 'ent']
        print_row(['epoch']+[item+'_loss' for item in loss_list], colwidth=15)

        for step in range(args.local_epoch):
            for data in train_loader:
                loss_result_dict = algorithm.update_d(data, optd)
            print_row([step]+[loss_result_dict[item]
                              for item in loss_list], colwidth=15)

        algorithm.set_dlabel(train_loader)

        print('====Domain-invariant feature learning====')
        loss_list = alg_loss_dict(args)
        eval_dict = train_valid_target_eval_names(args)
        print_key = ['epoch']
        print_key.extend([item+'_loss' for item in loss_list])
        print_key.extend([item+'_acc' for item in eval_dict.keys()])
        print_key.append('total_cost_time')
        print_row(print_key, colwidth=15)

        sss = time.time()
        for step in range(args.local_epoch):
            for data in train_loader:
                step_vals = algorithm.update(data, opt)

            results = {
                'epoch': step,
            }

            results['train_acc'] = modelopera.accuracy(
                algorithm, train_loader_noshuffle, None)

            acc = modelopera.accuracy(algorithm, valid_loader, None)
            results['valid_acc'] = acc

            acc = modelopera.accuracy(algorithm, target_loader, None)
            results['target_acc'] = acc

            for key in loss_list:
                results[key+'_loss'] = step_vals[key]
                
            # Update learning rate scheduler
            scheduler.step(results['valid_acc'])
            
            # Early stopping logic
            if results['valid_acc'] > best_valid_acc:
                best_valid_acc = results['valid_acc']
                target_acc = results['target_acc']
                patience_counter = 0
                best_model_state = algorithm.state_dict().copy()
                print(f"🔥 New best validation accuracy: {best_valid_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"⏹️ Early stopping at epoch {round} step {step}")
                    break
                    
            results['total_cost_time'] = time.time()-sss
            print_row([results[key] for key in print_key], colwidth=15)
            
        if patience_counter >= patience:
            break

    # Save best model
    if best_model_state:
        algorithm.load_state_dict(best_model_state)
        model_path = Path(args.output) / "best_model.pth"
        torch.save(algorithm.state_dict(), model_path)
        print(f"🔥 Saved BEST model to {model_path}")
    else:
        model_path = Path(args.output) / "model.pth"
        torch.save(algorithm.state_dict(), model_path)
        print(f"✅ Saved trained model to {model_path}")
        
    print(f'Best Validation Accuracy: {best_valid_acc:.4f}')
    print(f'Target Accuracy: {target_acc:.4f}')


if __name__ == '__main__':
    args = get_args()
    main(args)
