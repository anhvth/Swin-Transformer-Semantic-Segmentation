from .custom import *

@DATASETS.register_module(force=True)
class WaymoDepthDataset(CustomDataset):
    def __init__(self, *args,  depth_dir=None, depth_map_suffix='.npy',**kwargs):
        self.depth_dir = depth_dir
        self.depth_map_suffix = depth_map_suffix
        super(WaymoDepthDataset, self).__init__(*args,**kwargs)
    
    
    def pre_pipeline(self, results):
        super().pre_pipeline(results)
        results['depth_prefix'] = self.depth_dir
        results['flip'] = False
        results['flip_direction'] = None
        
    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """
        depth_dir = self.depth_dir
        depth_map_suffix = self.depth_map_suffix
        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:
            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                if depth_dir is not None:
                    depth_map = img.replace(img_suffix, depth_map_suffix)
                    img_info['ann'] = dict(depth_map=depth_map)
                img_infos.append(img_info)
        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos