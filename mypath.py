class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'rgbFire':
            return '/media/miguel/New Volume/Linux/Thesis/DeepLabV3Plus_Xception_PyTorch/dataset/rgbFire/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
