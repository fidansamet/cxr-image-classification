import argparse


class Options():
    def initialize(self, parser):
        parser.add_argument('--dataroot', required=False, default="./datasets/covid19", help='path to dataset')
        parser.add_argument('--name', type=str, default='knn', help='name of the experiment')
        parser.add_argument('--neighbor_num', type=int, default=3, help='number of closest neighbors to consider')
        parser.add_argument('--weighted_knn', action='store_true', help='if specified, use weighted k-nn algorithm')
        parser.add_argument('--kfold', action='store_true', help='if specified, use weighted k-fold algorithm')
        parser.add_argument('--fold_num', type=int, default=3, help='number of folds for k-fold algorithm')
        parser.add_argument('--tiny_img', action='store_true', help='if specified, use tiny image feature')
        parser.add_argument('--canny', action='store_true', help='if specified, use shape feature')
        parser.add_argument('--gabor', action='store_true', help='if specified, use texture feature')
        parser.add_argument('--vgg19', action='store_true', help='if specified, use deep image features')

        self.parser = parser
        return parser

    def print_options(self, opt):
        options = ''
        options += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            options += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        options += '----------------- End -------------------'
        print(options)

    def parse(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.opt = self.initialize(parser).parse_args()
        self.print_options(self.opt)
        return self.opt
