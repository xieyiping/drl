from utils import Parameters
from trainer import Trainer
import time
import tensorflow as tf

def main(parms: Parameters):
    # Log code
    time_str = time.strftime("%m-%d_%H-%M", time.localtime())
    info = ""
    log_dir = "logs/" + "_".join([time_str, parms.game, info])
    fw = tf.summary.create_file_writer(log_dir)

    trainer = Trainer(parms)
    
    for i in range(parms.train_step):
        avg_ret = trainer.train_step()
        print(f"train_step:{i} ret: {avg_ret:.1f}")
        with fw.as_default():
            tf.summary.scalar("return", avg_ret, i)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hyper prams")
    
    parser.add_argument("--replay_size",           type=str)
    parser.add_argument("--start_size",          type=float)
    parser.add_argument("--gamma",          type=float)
    parser.add_argument("--lr_a",           type=float)
    parser.add_argument("--batch_size",     type=int)
    parser.add_argument("--lr_c",           type=float)
    parser.add_argument("--parms_path",     type=str, 
                        default="parms/Pendulum-v0.json")
    
    args = parser.parse_args()
    terminal_parms = vars(args)

    # json parms
    import json
    json_parms = json.load(open(terminal_parms['parms_path']))

    json_parms.update({k: terminal_parms[k] for k in terminal_parms \
                        if terminal_parms[k] is not None})
    parms_ = Parameters(json_parms)
    main(parms_)
