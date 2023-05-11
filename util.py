# -*- coding:utf-8 -*-
import json
import win32gui


def load_cfg():
    with open('./config.json', 'r', encoding="utf-8") as f:
        content = f.read()
        cfg = json.loads(content)
        return cfg

def get_handle_by_name(handle, name="明日方舟"):
    # if old handle is working
    if handle is not None and handle > 0 and win32gui.IsWindowVisible(handle):
        return handle
    # find handle by name
    handle = None
    def winEnumHandler(hwnd, ctx):
        nonlocal handle
        if win32gui.IsWindowVisible(hwnd):
            win_name = win32gui.GetWindowText(hwnd)
            if name in win_name:
                handle = hwnd
                print('Use name: ', win_name)
    win32gui.EnumWindows(winEnumHandler, None)
    if handle is None:
        print("Window not found", end='\r')
    return handle
    
def get_size_by_pts(img, pts):
    h, w = img.shape[:2]
    h1, h2 = int(pts[1] * h), int(pts[3] * h)
    w1, w2 = int(pts[0] * w), int(pts[2] * w)
    return w1, h1, w2, h2
    
def crop_image_by_pts(img, pts):
    w1, h1, w2, h2 = get_size_by_pts(img, pts)
    return img[h1:h2, w1:w2]

def get_window_roi(handle, pos, padding):
    if handle is None:
        return {'top': -1, 'left': -1, 'width': 100, 'height': 100}

    window_rect = win32gui.GetWindowRect(handle)
    
    x1, y1, x2, y2 = pos
    ptop, pdown, pleft, pright = padding
    
    w = window_rect[2] - window_rect[0] - pleft - pright
    h = window_rect[3] - window_rect[1] - ptop - pdown
    
    window_dict = {
        'left': window_rect[0] + int(x1 * w) + pleft,
        'top': window_rect[1] + int(y1 * h) + ptop,
        'width': int((x2 - x1) * w),
        'height': int((y2 - y1) * h)
    }
    return window_dict

    
if __name__ == '__main__':
    print("Resize images")
    import glob 
    import cv2 as cv
    imgs = glob.glob('img/*.png')
    
    for i in imgs:
        img = cv.imread(i)
        if img.shape[1] > 1024:
            print(f"Process {i}")
            img = cv.resize(img, (1024, 576))
            cv.imwrite(i, img)
    
    