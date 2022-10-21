from subprocess import Popen, PIPE, STDOUT, CalledProcessError
from sys import argv
from argparse import ArgumentParser
from pathlib import Path
import re
import os
from multiprocessing import cpu_count

def relpath(path):
    return (Path(__file__).parent / path).resolve()

default_model_path = relpath("../models")

epoch_regex = re.compile("epoch_(\d+)")

parser = ArgumentParser(description="Train an epoch and automatically export the resulting ckpt file")
parser.add_argument("--source_path", type=Path, required=True, help="directory where source images will be taken from")
parser.add_argument("--train_path", type=Path, help="directory where the model is stored, and where training will take place")
parser.add_argument("--export_path", type=Path, help="Directory to export the resulting ckpt file to from the last trained epoch")
parser.add_argument("--model_name", type=str, help="Name of the model, used to create the training folder, and to create the export file name")
parser.add_argument("--instance_prompt", type=str, required=True, help="prompt to trigger the model with the newly trained data")
parser.add_argument("--class_prompt", type=str, help="prompt to use when identifying class samples")
parser.add_argument("--class_path", type=Path, help="directory where class images will be taken from")
parser.add_argument("--base_checkpoint", type=Path, default=None, help="path to a checkpoint file to start training from if this is the first epoch")
parser.add_argument("--learn_rate", type=float, default=1e-6, help="learning rate of the training")
parser.add_argument("--steps", type=int, default=1000, help="number of training steps to perform")
parser.add_argument("--seed", type=int, default=3434554, help="training seed")
parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs to run and generate checkpoints for")

def execute(args):
    with Popen(args, stdout=PIPE, stderr=STDOUT, bufsize=1, universal_newlines=True) as p:
        for line in p.stdout:
            print(line, end='')
        
        returncode = p.poll()
        if returncode != 0:
            raise CalledProcessError(returncode, p.args)

def print_arglist(args): print("\n".join(f"    {a}" for a in args))

def main(args):
    if args.train_path == None:
        if args.model_name != None:
            args.train_path = default_model_path / args.model_name
        else:
            raise ValueError("Please provide model name via --model_name if no train path is specified")

    os.makedirs(str(args.train_path.resolve()), exist_ok=True)
    if not any(path for path in args.train_path.iterdir() if path.is_dir() and epoch_regex.match(path.name) != None):
        if args.base_checkpoint != None:
            # no existing epoch exists, create epoch_0 from the provided checkpoint
            base_epoch_path = args.train_path / 'epoch_0'
            create_base_epoch_args = [
                "python", relpath("convert_original_stable_diffusion_to_diffusers.py"),
                f"--checkpoint_path={args.base_checkpoint}",
                f"--dump_path={base_epoch_path}"
            ]
            print("Creating base model with these args:")
            print_arglist(create_base_epoch_args)
            try:
                execute(create_base_epoch_args)
            except CalledProcessError:
                pass
            if not base_epoch_path.is_dir():
                print("failed to create base epoch, exiting")
                return
        else:
            raise Exception("No existing epochs found, please provide a base model file to train from.")

    for epoch in range(args.n_epochs):
        # train the epoch
        existing_epochs = sorted(((p, e) for p, e in ((path, epoch_regex.match(path.name)) for path in args.train_path.iterdir() if path.is_dir()) if e != None), key=(lambda p: int(p[1].group(1))), reverse=True)
        _, latest_epoch_regex = existing_epochs[0]
        old_epoch_num = int(latest_epoch_regex.group(1))
        new_epoch_num = old_epoch_num + 1
        old_epoch_path = args.train_path / f"epoch_{old_epoch_num}"
        new_epoch_path = args.train_path / f"epoch_{new_epoch_num}"
        print(f"Beginning training on epoch {new_epoch_num}" + ('' if args.n_epochs == 1 else f", {epoch+1}/{args.n_epochs}"))

        train_args = [
            "accelerate", "launch",
            f"--num_cpu_threads_per_process={cpu_count()}",
            relpath("../examples/dreambooth/train_dreambooth.py"),
            f"--pretrained_model_name_or_path={old_epoch_path}",
            f"--instance_data_dir={args.source_path}",
            f"--output_dir={new_epoch_path}",
            f"--instance_prompt={args.instance_prompt}",
            f"--seed={args.seed}",
            "--resolution=512",
            "--train_batch_size=1",
            "--train_text_encoder",
            "--mixed_precision=fp16",
            "--use_8bit_adam",
            "--gradient_accumulation_steps=1",
            f"--learning_rate={args.learn_rate}",
            "--lr_scheduler=constant",
            "--lr_warmup_steps=0",
            "--sample_batch_size=4",
            f"--max_train_steps={args.steps}"
        ]
        if None not in [args.class_prompt, args.class_path]:
            train_args.extend([
                f"--class_prompt={args.class_prompt}",
                "--with_prior_preservation", 
                "--prior_loss_weight=1.0",
                f"--class_data_dir={args.class_path}",
                f"--num_class_images={sum(1 for p in args.class_path.iterdir() if p.is_file())}"
            ])
        
        print("training with these args:")
        print_arglist(train_args)

        try:
            os.environ["LD_LIBRARY_PATH"] = f"/usr/lib/wsl/lib"
            execute(train_args)
        except CalledProcessError:
            print("Error during training, skipping model export")
            continue

        # export the epoch
        if None not in [args.export_path, args.model_name]:
            full_export_path = args.export_path / f"{args.model_name}_e{new_epoch_num}.ckpt"
            export_args = [
                "python",
                relpath("convert_diffusers_to_original_stable_diffusion.py"),
                f"--model_path={new_epoch_path}",
                f"--checkpoint_path={full_export_path}"
            ]
            print(f"exporting model with these args:")
            print_arglist(export_args)

            try:
                execute(export_args)
            except CalledProcessError:
                print("Error during model export")
                continue
    
if __name__ == "__main__":
    main(parser.parse_args())