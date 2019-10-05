# import timeit
from east import east, data
from dataset import msra

# model = east.EAST()
# model.summary_model()

msra_data = msra.load_msra_td_500(
    '/home/beekill/projects/bookual/datasets/MSRA-TD500/train/', 4)

# time = timeit.timeit(lambda: next(msra_data), number=10)

abc = data.flow_from_generator(msra_data,
                               crop_target_size=(512, 512),
                               crop_at_least_one_box_ratio=0.8,
                               random_scales=[0.5, 1.0, 1.5, 2.0],
                               random_angles=(-45, 45))

images, gts = next(abc)

print('done')
# print(f'done after {time}')
