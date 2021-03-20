import argparse


class Options():
    def initialize(self, parser):
        parser.add_argument('--dataroot', type=str, default='./datasets/covid19', help='path to dataset')
        parser.add_argument('--phase', type=str, default='train', help='train or test phase')
        parser.add_argument('--name', type=str, default='knn', help='name of the experiment')
        parser.add_argument('--neighbor_num', type=int, default=6, help='number of closest neighbors to consider')
        parser.add_argument('--dist_measure', type=str, default='manhattan', help='distance measure for calculation')
        parser.add_argument('--weighted_knn', action='store_true', help='if specified, use weighted k-nn algorithm')
        parser.add_argument('--fold_num', type=int, default=5, help='number of folds for k-fold algorithm')
        parser.add_argument('--canny', action='store_true', help='if specified, use shape feature')
        parser.add_argument('--gabor', action='store_true', help='if specified, use texture feature')
        parser.add_argument('--hog', action='store_true', help='if specified, use HoG feature')
        parser.add_argument('--vgg19', action='store_true', help='if specified, use deep image features')
        parser.add_argument('--tiny_img', action='store_true', help='if specified, use tiny image feature')
        parser.add_argument('--small_img', action='store_true', help='if specified, use small image feature')
        parser.add_argument('--features_path', type=str, default='features', help='path to saved features')
        parser.add_argument('--normalize', action='store_true', help='if specified, normalize the features')

        self.parser = parser
        return parser

    def print_options(self, opt):
        options = ''
        options += '----------------- Options ---------------\n'
        for k, v in vars(opt).items():
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
