class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/media/miguel/New Volume/Linux/Thesis/Sample Gits/pytorch-deeplab-xception/dataset/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/media/miguel/New Volume/Linux/Thesis/Sample Gits/pytorch-deeplab-xception/dataset/SBD/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        elif dataset == 'rgbFire':
            return '/media/miguel/New Volume/Linux/Thesis/DeepLabV3Plus_Xception_PyTorch/dataset/rgbFire/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
