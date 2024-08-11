import argparse
import os
import torch
from downstream_tasks.loader import make_dataset
from downstream_tasks.setup import setup_and_build_model
from downstream_tasks.utils import extract_features


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--train-dataset", dest="train_dataset_str", type=str)
    parser.add_argument("--val-dataset", dest="val_dataset_str", type=str)
    parser.add_argument("--test-dataset", dest="test_dataset_str", type=str, default='')
    parser.add_argument("--batch_size", type=int)

    return parser


def main(args, save_dir):
    model, transform, _ = setup_and_build_model(args.model_name)
    print(model)
    print(transform)
    train_dataset = make_dataset(
        dataset_str=args.train_dataset_str,
        transform=transform,
    )
    
    val_dataset = make_dataset(
        dataset_str=args.val_dataset_str,
        transform=transform,
    )
    path = os.path.join(save_dir, 'train.pt')
    if not os.path.exists(path):
        print('Extracting train features ...')
        train_features, train_labels, paths = extract_features(model, train_dataset, args.batch_size, 8, gather_on_cpu=True)
        train_data = {'feature': train_features, 'label': train_labels, 'paths': paths}
        print('save:', path)
        torch.save(train_data, path)
    
    path = os.path.join(save_dir, 'val.pt')
    if not os.path.exists(path):
        print('Extracting val features ...')
        val_features, val_labels, paths = extract_features(model, val_dataset, args.batch_size, 8, gather_on_cpu=True)
        val_data = {'feature': val_features, 'label': val_labels, 'paths': paths}
        print('save:', path)
        torch.save(val_data, path)
        
    # test dataset
    path = os.path.join(save_dir, 'test.pt')
    if args.test_dataset_str:
        if not os.path.exists(path):
            print('Extracting test features...')
            test_dataset = make_dataset(
                dataset_str=args.test_dataset_str,
                transform=transform,
            )        
            path = os.path.join(save_dir, 'test.pt')
            if not os.path.exists(path):
                test_features, test_labels, paths = extract_features(model, test_dataset, args.batch_size, 4, gather_on_cpu=True)
                test_data = {'feature': test_features, 'label': test_labels, 'paths': paths}
                print('save:', path)
                torch.save(test_data, path)


if __name__ == "__main__":
    args_parser = get_args_parser()
    args = args_parser.parse_args()
    save_dir = os.path.join(args.output_dir, args.model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    main(args, save_dir)
