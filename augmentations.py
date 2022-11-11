import torchvision.transforms as transforms

from lanemarkingfilter import color_filter,color_filter_ori, color_filter_ori_color, color_filter_ori_random_color

class LaneMarking(object):
    def __call__(self, img):
        return color_filter(img)

class LaneMarkingOri(object):
    def __call__(self, img):
        return color_filter_ori(img)

class LaneMarkingOriColor(object):
    def __call__(self, img):
        return color_filter_ori_color(img)

class LaneMarkingOriRandomColor(object):
    def __call__(self, img):
        return color_filter_ori_random_color(img)


class TrainTransform(object):
    def __init__(self):
        self.transform_prime = transforms.Compose(
            [   
                transforms.Resize(256),
                transforms.CenterCrop(224),
                LaneMarking(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                ]
            )

        self.transform= transforms.Compose(
            [   transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                ]
            )
        pass

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform_prime(sample)
        return x1, x2

class TrainTransformOriGrayScale(object):
    def __init__(self):
        self.transform_prime = transforms.Compose(
            [   
                transforms.Resize(256),
                transforms.CenterCrop(224),
                LaneMarkingOri(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                ]
            )

        self.transform= transforms.Compose(
            [   transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                ]
            )
        pass
    
    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform_prime(sample)
        return x1, x2

class TrainTransformOriColor(object):
    def __init__(self):
        self.transform_prime = transforms.Compose(
            [   
                transforms.Resize(256),
                transforms.CenterCrop(224),
                LaneMarkingOriColor(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                ]
            )

        self.transform= transforms.Compose(
            [   transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                ]
            )
        pass

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform_prime(sample)
        return x1, x2

class TrainTransformOriRandomColor(object):
    def __init__(self):
        self.transform_prime = transforms.Compose(
            [   
                transforms.Resize(256),
                transforms.CenterCrop(224),
                LaneMarkingOriRandomColor(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                ]
            )

        self.transform= transforms.Compose(
            [   transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                ]
            )
        pass

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform_prime(sample)
        return x1, x2

class TrainTransformOriColorJitter(object):
    def __init__(self):
        self.transform_prime = transforms.Compose(
            [   
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                LaneMarkingOriColor(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                ]
            )

        self.transform= transforms.Compose(
            [   transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                ]
            )
        pass

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform_prime(sample)
        return x1, x2


