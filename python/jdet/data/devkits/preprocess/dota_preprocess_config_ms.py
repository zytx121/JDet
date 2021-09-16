type='DOTA'
source_dataset_path='/mnt/disk/czh/dota_images_single/dota_images'
target_dataset_path='/home/czh/JDet/DOTA_single_jdet'

# available labels: train, val, test, trainval
tasks=[
    dict(
        label='trainval',
        config=dict(
            subimage_size=1024,
            overlap_size=200,
            multi_scale=[0.5,1., 1.5],
            horizontal_flip=False,
            vertical_flip=False,
            rotation_angles=[0.] 
        )
    ),
    dict(
        label='test',
        config=dict(
            subimage_size=1024,
            overlap_size=200,
            multi_scale=[0.5,1.,1.5],
            horizontal_flip=False,
            vertical_flip=False,
            rotation_angles=[0.] 
        )
    )
]