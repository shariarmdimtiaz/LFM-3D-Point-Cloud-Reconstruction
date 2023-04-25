import numpy as np
import pathlib
from PIL import Image
from Scripts.PFM_rw import read_pfm
import time
import glob


img_dir = 'IC1'
full_data_root = pathlib.Path(f'./full_data/lfm/{img_dir}')
patch_data_root = pathlib.Path('./patch_data')
img_w = 76
img_h = 76

def create_patch_data(data_path):
    # stack EPI func
    def stack_epi(start, end, step):
        target_arr = np.zeros((9, img_w+1, img_h+1, 3), dtype=np.uint8)
        for i, image_path in enumerate([data_path / f'input_Cam{i:03}.png' for i in range(start, end, step)]):
            img = np.asarray(Image.open(image_path))
            target_arr[i, :img_w, :img_h] = img
            target_arr[i, img_w] = target_arr[i, img_w-1]
            target_arr[i, :, img_h] = target_arr[i, :, img_h-1]
        return target_arr

    # stack EPI
    stack_x = stack_epi(4, 81, 9)
    stack_y = stack_epi(36, 45, 1)
    stack_xd = stack_epi(0, 81, 10)
    stack_yd = stack_epi(8, 73, 8)

    # save binary
    save_dir = patch_data_root / data_path.name
    save_dir.mkdir(parents=True, exist_ok=True)
    file_num = 0
    for y in range(0, 38):
        for x in range(0, 38):
            np.save(save_dir / f'{file_num:04}_x.npy', stack_x[:, y * 13:y * 13 + 32, x * 13:x * 13 + 32])
            np.save(save_dir / f'{file_num:04}_y.npy', stack_y[:, y * 13:y * 13 + 32, x * 13:x * 13 + 32])
            np.save(save_dir / f'{file_num:04}_xd.npy', stack_xd[:, y * 13:y * 13 + 32, x * 13:x * 13 + 32])
            np.save(save_dir / f'{file_num:04}_yd.npy', stack_yd[:, y * 13:y * 13 + 32, x * 13:x * 13 + 32])
            file_num += 1

    np.save(save_dir / 'full_x.npy', stack_x[:, :512, :512])
    np.save(save_dir / 'full_y.npy', stack_y[:, :512, :512])
    np.save(save_dir / 'full_xd.npy', stack_xd[:, :512, :512])
    np.save(save_dir / 'full_yd.npy', stack_yd[:, :512, :512])
    return


def main():
    start_time = time.process_time()
    disp_path = full_data_root

    # create input patch
    create_patch_data(full_data_root)
    print(f'Done: {disp_path.name}')

    end_time = time.process_time() - start_time
    print("Time elapsed: ", end_time)


if __name__ == "__main__":
    main()