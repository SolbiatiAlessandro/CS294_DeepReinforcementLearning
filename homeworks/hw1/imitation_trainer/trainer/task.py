import pickle as pkl
import argparse
from model import Model as ImitationTrainerModel

def load_data(expert_data_filename, local=True):
    """
    returns numpy arrays X, Y
    X.shape = (, 44)
    Y.shape = (, 17)
    """
    if local:
        expert_rollouts = pkl.load(open(expert_data_filename,"rb"))
        X = expert_rollouts['observations']
        Y = expert_rollouts['actions']
    else:
        # download from GCS
        raise NotImplemented
        
    assert len(X) == len(Y)
    return X, Y

def train_and_save(args):
    """
    routine for training and saving model

    note: to see how to predict look inside task.ipynb
    """
    local = args.mode == "local"
    X, Y = load_data(args.input_files, local)
    model = ImitationTrainerModel()
    train_mse = model.train(
            X,
            Y, 
            steps=args.train_steps,
            batch_size=args.train_batch_size,
            save_folder=args.job_dir)

    # visualize mse
    # model.visualize_train_mse(train_mse)

    if not local: 
        # upload on GCS
        raise NotImplemented

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--mode',
            type=str,
            help="local or global (triggers GCS)",
            default="local")
    parser.add_argument(
        '--input-files',
        nargs='+',
        help='Training files local or GCS',
        default="../../expert_data/RoboschoolHumanoid-v1.pkl")
    parser.add_argument(
        '--job-dir',
        type=str,
        help="""GCS or local dir for checkpoints, exports, and summaries.
          Use an existing directory to load a trained model, or a new directory
          to retrain""",
        default='/tmp/imitation-learning/')
    parser.add_argument(
        '--train-steps',
        type=int,
        help='Maximum number of training steps to perform.')
    parser.add_argument(
        '--eval-steps',
        help="""Number of steps to run evalution for at each checkpoint.train_batch_size,
        If unspecified, will run for 1 full epoch over training data""",
        default=None,
        type=int)
    parser.add_argument(
        '--train-batch-size',
        type=int,
        default=40,
        help='Batch size for training steps')
    parser.add_argument(
        '--eval-batch-size',
        type=int,
        default=40,
        help='Batch size for evaluation steps')
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.003,
        help='Learning rate for SGD')


    args, _ = parser.parse_known_args()
    train_and_save(args)
