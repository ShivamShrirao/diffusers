from subprocess import Popen, PIPE, STDOUT, CalledProcessError
from argparse import ArgumentParser
from pathlib import Path
from shutil import rmtree
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
parser.add_argument("--vae_name", type=Path, help="Name of the pretrained vae of the chosen model")
parser.add_argument("--learn_rate", type=float, default=1e-6, help="learning rate of the training")
parser.add_argument("--steps", type=int, default=1000, help="number of training steps to perform")
parser.add_argument("--learning_rate_steps", nargs="?", default=1e-3, type=float, help="Learning-rate-steps, will override the default step value and use this to automatically determine the number of steps to perform based on the learning rate. Ex: a value of 1e-3 will result in 1000 steps with a learning rate of 1e-6, and 10000 with a learning rate of 1e-7.")
parser.add_argument("--seed", type=int, default=3434554, help="training seed")
parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs to run and generate checkpoints for")
parser.add_argument("--delete_malformed_models", action='store_true', help="Enable to allow the model to automatically delete the folders of malformed models that it detects")

def execute(args, env={}):
    class BufferPrinter:
        def __init__(self):
            self.buf = ""
        def print(self, msg, end=""):
            print(msg, end=end)
            self.buf += msg
            self.buf = self.buf[-100000:]

    environ = os.environ.copy()
    environ.update(env)
    with Popen(args, stdout=PIPE, stderr=STDOUT, env=environ) as p:
        buf = b""
        rear_buffer = BufferPrinter()
        while not p.stdout.closed:
            chunk = p.stdout.read(8)
            if chunk == b'':
                break
            buf += chunk
            flush = b""
            for n in [b"\n", b"\r"]:
                if n in buf:
                    ind = len(buf) - buf[::-1].index(n) - 1
                    new_flush, buf = buf[:ind], buf[ind:]
                    flush += new_flush
            if flush != b"":
                rear_buffer.print(flush.decode("utf-8"), end='')
                flush = b""
        rear_buffer.print(buf.decode('utf-8'))
        
        returncode = p.poll()
        if returncode != 0:
            if all(err in rear_buffer.buf for err in ["EnvironmentError", "OSError: Can't load tokenizer"]):
                raise ValueError()
            else:
                raise CalledProcessError(returncode, p.args)

def print_arglist(args): print("\n".join(f"    {a}" for a in args))

def main(args):
    if args.train_path == None:
        if args.model_name != None:
            args.train_path = default_model_path / args.model_name
        else:
            raise ValueError("Please provide model name via --model_name if no train path is specified")

    os.makedirs(str(args.train_path.resolve()), exist_ok=True)

    for epoch in range(args.n_epochs):
        while True:
            # if no existing epoch exists, create epoch_0 from the provided checkpoint
            if not any(path for path in args.train_path.iterdir() if path.is_dir() and epoch_regex.match(path.name) != None):
                if args.base_checkpoint != None:
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

            # train the epoch
            existing_epochs = sorted(((p, e) for p, e in ((path, epoch_regex.match(path.name)) for path in args.train_path.iterdir() if path.is_dir()) if e != None), key=(lambda p: int(p[1].group(1))), reverse=True)
            _, latest_epoch_regex = existing_epochs[0]
            old_epoch_num = int(latest_epoch_regex.group(1))
            new_epoch_num = old_epoch_num + 1
            old_epoch_path = args.train_path / f"epoch_{old_epoch_num}"
            new_epoch_path = args.train_path / f"epoch_{new_epoch_num}"
            """
            vae_paths = []
            if args.vae_path != None:
                vae_paths.append(args.vae_path)
            if args.base_checkpoint != None:
                vae_paths.append(args.base_checkpoint.parent / f"{args.base_checkpoint.stem}.vae.pt")
            if args.model_name != None:
                vae_paths.append(Path(f"{args.model_name}.vae.pt"))
            vae_paths.append(Path(f"sd-v1-4.vae.pt"))
            if not any(p.is_file() for p in vae_paths):
                raise ValueError("Count not find vae file: please provide a link to an appropriate pretrained vae with --vae_path, or put it in the same directory as your base checkpoint.")
            vae_path = next(p for p in vae_paths if p.is_file())
            """
            print(f"Beginning training on epoch {new_epoch_num}" + ('' if args.n_epochs == 1 else f", {epoch+1}/{args.n_epochs}"))

            steps = args.steps
            if args.learning_rate_steps:
                steps = int(args.learning_rate_steps / args.learn_rate)

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
                f"--max_train_steps={steps}"
            ]

            if args.vae_name:
                train_args.extend([
                    f"--pretrained_vae_name_or_path={args.vae_name}",
                    "--use_vae"
                ])

            if None not in [args.class_prompt, args.class_path]:
                train_args.extend([
                    f"--class_prompt={args.class_prompt}",
                    "--with_prior_preservation", 
                    "--prior_loss_weight=1.0",
                    f"--class_data_dir={args.class_path}",
                    f"--num_class_images=50"
                ])
            
            print("training with these args:")
            print_arglist(train_args)

            try:
                execute(train_args, env={
                    "LD_LIBRARY_PATH": "/usr/lib/wsl/lib"
                })
            except ValueError:
                print("Detected a malformed model folder from a previously crashed or cancelled training session.")
                if new_epoch_path.exists():
                    rmtree(new_epoch_path)
                if args.delete_malformed_models:
                    print(f"Deleting model folder {old_epoch_path}")
                    rmtree(old_epoch_path)
                    continue
                else:
                    print("If you would like the script to automatically delete this model folder and continue training, pass --delete_malformed_models")
                    return
            except CalledProcessError:
                print("Error during training, skipping model export")
                break

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

            break
    
    print("Finished training!")
    
if __name__ == "__main__":
    main(parser.parse_args())