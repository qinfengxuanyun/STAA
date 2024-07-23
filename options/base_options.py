import argparse
import os
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self, parser):
        # base define
        parser.add_argument('--attention_type', type=str, default='original_full',
                            help='attention_type[original_full|block_sparse]')
        parser.add_argument('--num_hidden_layers', type=int, default=3)
        parser.add_argument('--chunk_size_feed_forward', type=int, default=0)
        parser.add_argument('--max_position_embeddings', type=int, default=2048) #1024
        parser.add_argument('--hidden_size', type=int, default=512)#512
        parser.add_argument('--intermediate_size', type=int, default=1024)#1024
        parser.add_argument('--num_attention_heads', type=int, default=8)
        parser.add_argument('--num_random_blocks', type=int, default=3)
        parser.add_argument('--block_size', type=int, default=6)
        parser.add_argument('--use_bias', type=bool, default=True)
        parser.add_argument('--layer_norm_eps', type=float, default=1e-12)
        parser.add_argument('--is_decoder', type=bool, default=False)
        parser.add_argument('--add_cross_attention', type=bool, default=False)
        parser.add_argument('--pad_token_id', type=int, default=3)
        parser.add_argument('--vocab_size', type=int, default=5)
        parser.add_argument('--type_vocab_size', type=int, default=2)
        parser.add_argument('--rescale_embeddings', type=bool, default=False)
        parser.add_argument('--output_attentions', type=bool, default=False)
        parser.add_argument('--output_hidden_states', type=bool, default=False)
        parser.add_argument('--use_return_dict', type=bool, default=True)
        parser.add_argument('--use_cache', type=bool, default=False)
        parser.add_argument('--hidden_dropout_prob', type=float, default=0)
        parser.add_argument('--classifier_dropout', type=float, default=0.3)#0.3
        parser.add_argument('--attention_probs_dropout_prob', type=float, default=0)#0.2
        parser.add_argument('--mask_dropout', type=float, default=0.5)#0.2
        parser.add_argument('--mask_dropout2', type=float, default=0.5)
        parser.add_argument('--num_labels', type=int, default=2)
        parser.add_argument('--use_attn1', type=bool, default=True)
        parser.add_argument('--use_attn2', type=bool, default=True)
        parser.add_argument('--use_gender', type=bool, default=False)
        parser.add_argument('--mri_th', type=int, default=120)
        parser.add_argument('--snp_th', type=int, default=20)
        parser.add_argument('--clamp_th', type=float, default=0.01)
        parser.add_argument('--acc_th', type=float, default=1.0)
        parser.add_argument('--alpha_sparse', type=float, default=-3)
        parser.add_argument('--alpha_sparse2', type=float, default=-2)
        parser.add_argument('--data_path', type=str, default="nonlinear_3.mat")
        parser.add_argument('--use_sparse', type=int, default=1)
        parser.add_argument('--use_sparse_supervise', type=int, default=1)
        parser.add_argument('--use_sparse2', type=int, default=1)
        parser.add_argument('--dataset_type', type=str, default='adni', help='[sim|adni]')
        parser.add_argument('--age_range_min', type=float, default=50.0)
        parser.add_argument('--age_range_max', type=float, default=100.0)
        parser.add_argument('--alpha_guide', type=float, default=1.0)
        parser.add_argument('--alpha_mse', type=float, default=1.0)
        parser.add_argument('--alpha_guide2', type=float, default=1.0)
        parser.add_argument('--alpha_mse2', type=float, default=1.0)
        parser.add_argument('--snp1', type=int, default=0)
        parser.add_argument('--snp2', type=int, default=2)
        return parser

    def gather_options(self):
        """Add additional model-specific options"""

        if not self.initialized:
            parser = self.initialize(self.parser)

        # get basic options
        #parser, _ = parser.parse_known_args()

        opt = parser.parse_args()

        return opt

    def parse(self):
        """Parse the options"""

        opt = self.gather_options()
        opt.isTrain = self.isTrain

        self.print_options(opt)

        self.opt = opt

        return self.opt

    @staticmethod
    def print_options(opt):
        """print and save options"""

        print('--------------Options--------------')
        for k, v in sorted(vars(opt).items()):
            print('%s: %s' % (str(k), str(v)))
        print('----------------End----------------')
