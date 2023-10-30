import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MATRIX_PATH = 'src/utils/perspective_matrix.txt'


def transform_points(pts1):
    M = np.loadtxt(MATRIX_PATH, delimiter=',')
    pts2 = cv2.perspectiveTransform(pts1, M)
    return pts2


def transform_df_points(df):
    M = np.loadtxt(MATRIX_PATH, delimiter=',')

    df_pts = df[['xc', 'y2']]
    pts1 = df_pts.to_numpy(dtype=np.float32)
    pts1 = np.expand_dims(pts1, axis=0)

    pts2 = cv2.perspectiveTransform(pts1, M)
    df['x_ipm'] = pts2[0, :, 0]
    df['y_ipm'] = pts2[0, :, 1]
    return df


def plot_measuring_points():
    pts1 = np.array([[[155, 287],
                      [120, 223],
                      [145, 160],
                      [75, 140],
                      [258, 167],
                      [287, 163],
                      [315, 159],
                      [326, 178],
                      [259, 132],
                      [100, 216],
                      [68, 144],
                      [258, 132],
                      [320, 154],
                      [80, 169],
                      ]], dtype=np.float32)
    pts2 = transform_points(pts1)
    plot_points(pts1, pts2, 'ipm_measuring_pts.jpg')


def plot_annotations():
    from io import StringIO

    DATA = StringIO('''object_id name x1 y1 x2 y2 occlusion
00052909 human 162 100 170 119 0
00052912 human 190 98 196 115 0
00052913 human 215 134 227 170 0
00052914 human 92 143 102 167 0
00052915 human 227 139 235 170 0
00052916 human 99 153 108 195 0
00052917 human 99 154 111 192 0
00052919 human 132 86 137 98 0
00052922 human 242 125 254 156 1
00052923 human 127 98 132 114 0
00052924 human 132 98 138 114 0
00052925 human 212 93 218 101 0
00052926 human 223 85 229 102 1
00052928 human 156 79 161 91 1
00052942 human 152 144 163 159 0
00052943 vehicle 203 237 373 288 0
00052944 vehicle 201 197 311 251 0
00052945 vehicle 147 179 260 221 0
00052946 vehicle 146 169 236 200 0
00052947 vehicle 133 154 206 189 0''')

    df = pd.read_csv(DATA, sep=" ")
    df['xc'] = (df.x2 + df.x1) // 2
    pts1 = df[['xc', 'y2']].to_numpy(dtype=np.float32)
    pts1 = np.expand_dims(pts1, axis=0)

    df = transform_df_points(df)
    pts2 = df[['x_ipm', 'y_ipm']].to_numpy(dtype=np.float32)
    pts2 = np.expand_dims(pts2, axis=0)

    plot_points(pts1, pts2, 'ipm_annotations.jpg')


def plot_points(pts1, pts2, save_name: str = 'ipm.jpg'):
    image = cv2.imread('src/data/20200816_clip_1_0028_image_0043.jpg')
    M = np.loadtxt(MATRIX_PATH, delimiter=',')
    img_warp = cv2.warpPerspective(image, M, (target_w, target_h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)

    # Draw results
    for pt in pts1[0]:
        cv2.circle(image, np.rint(pt).astype(int), 2, (255, 0, 0), -1)
    for pt in pts2[0]:
        cv2.circle(img_warp, np.rint(pt).astype(int), 15, (255, 0, 0), -1)

    # images for paper
    # TODO: should save individual images with axis
    save_path = 'src/data/tmp/ipm_original.jpg'
    cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    save_path = 'src/data/tmp/ipm_warped.jpg'
    cv2.imwrite(save_path, cv2.cvtColor(img_warp, cv2.COLOR_RGB2BGR))

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image, origin='upper')
    ax[0].set_title('Camera view')
    ax[1].imshow(img_warp, origin='lower')
    ax[1].set_title('IPM')
    save_path = f'src/data/tmp/{save_name}'
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=100)
    plt.show()


if __name__ == '__main__':
    # 4 point correspondence between image and target mapping
    img_pts = np.array([[100, 216],
                        [68, 144],
                        [258, 132],
                        [320, 154],
                        ], dtype=np.float32)
    offset = 7.5 + 7.5
    target_pts = np.array([
                           [0, offset],
                           [0, offset + 19.6 + 20.0],
                           [12.8, offset + 19.6 - 1.5 + 17.0],
                           [12.4, offset + 19.6 - 1.5],
                           ], dtype=np.float32)
    target_pts *= 100  # convert from meter to cm
    target_pts[:, 0] += 50  # offset x-axis
    target_w, target_h = 1500, 6000

    # compute projection matrix
    M = cv2.getPerspectiveTransform(img_pts, target_pts)
    np.savetxt(MATRIX_PATH, M, fmt='%f', delimiter=',')

    plot_measuring_points()
    # plot_annotations()
