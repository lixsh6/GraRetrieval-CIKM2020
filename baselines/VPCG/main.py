from model import *
from config import *

def main(args,config):
    vpcg = VPCG(config)
    vpcg.build_graph()

    vpcg.train()
    vpcg.train_word_embed()

    vpcg.test(label_type='PSCM',save_result=True)
    vpcg.test(label_type='DBN', save_result=True)
    vpcg.test(label_type='HUMAN', save_result=True)


if __name__ == '__main__':
    args = load_arguments()
    config_state = eval(args.prototype)()
    main(args, config_state)