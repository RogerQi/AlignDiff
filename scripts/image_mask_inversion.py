import os

from aligndiff.nm_inversion import normalized_masked_inverse_one
from aligndiff.utils import parse_args, read_json_file

def main(args):
    dataset_dir = args.dataset_dir
    dataset_json_file = os.path.join(dataset_dir, "dataset.json")
    dataset = read_json_file(dataset_json_file)
    for item in dataset:
        if item['type'] == 'NM_embedding':
            obj_folder_path = os.path.join(dataset_dir, item['path'])
            embedding_dir_path = os.path.join(obj_folder_path, "learned_embeddings")
            print("Optimizing embedding for {} in {}".format(item["category"], obj_folder_path))

            if os.path.exists(embedding_dir_path):
                print("***** Embedding already exists for {} in {}. *****".format(item["category"], obj_folder_path))
                print("**********")

            normalized_masked_inverse_one(args.diffusion_model,
                obj_folder_path,
                args.inversion_train_bs,
                args.inversion_lr,
                embedding_dir_path,
                item["category"],
                args.inversion_steps,
                args.bg_lambda)

if __name__ == "__main__":
    args = parse_args()
    main(args)
