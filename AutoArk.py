# -*- coding:utf-8 -*-
import mss
from util import *
from tkinter import *
import tkinter.font as tkFont
from PIL import Image, ImageDraw, ImageFont, ImageTk
from adb_shell.adb_device import AdbDeviceTcp
from itertools import combinations
import numpy as np
import cv2
from datetime import datetime
import threading
import random
import time


class TagDetector:
    def __init__(self, thre):
        self.dbs = {}
        self.tags = []
        self.cache = {}
        self.font = ImageFont.truetype("./res/NotoSansSC-Medium.otf", 16, encoding="unic")  # 设置字体
        self.load_db()
        self.thre = thre

    def load_db(self):
        # load DB
        with open('./res/opt.txt', 'r', encoding="utf-8") as f:
            db = f.read()
        opts = db.split('\n')
        dbs = {}
        
        def add_to_dbs(ts):
            name = ts[0]
            level = int(ts[1])
            tgs = ts[2:]
            if level == 5:
                tgs.append('资深干员')
            if level == 6:
                tgs.append('高级资深干员')
            
            for tg in tgs:
                # 1 tags
                if tg in dbs:
                    dbs[tg].append((name, level))
                else:
                    dbs[tg] = [(name, level)]


        for opt in opts:
            ts = opt.split()
            if len(ts) > 2:
                add_to_dbs(ts)
                
        self.dbs = dbs
        self.tags = list(dbs.keys())
        print("Keys: ", self.tags)
        print("公招数据库加载完毕.")
        # print("Dbs: ", self.dbs)
        


    def query_tag(self, tag, src):
        if tag in self.cache:
            tpl = self.cache[tag]
        else:
            img_w = 100
            img_h = 30
            image= Image.new('L', (img_w, img_h), 49)
            draw = ImageDraw.Draw(image)
            # draw.text()
            w, h = draw.textsize(tag, font=self.font)
            draw.text(((img_w - w) // 2, 2), tag, 'white', self.font)
            tpl = np.array(image)
            self.cache[tag] = tpl
            """
            # check generated tag
            if tag == '先锋干员':
                cv2.imshow('target', tpl)
                cv2.waitKey(0)
            """
        
        res = cv2.matchTemplate(src, tpl, cv2.TM_CCOEFF_NORMED)
        """
        cv2.imshow('src', src)
        cv2.imshow('tpl', tpl)
        cv2.waitKey(0)
        """
        h, w = res.shape[:2]
        p_h = h // 2
        p_w = w // 3
        patch_val = []
        for patch in [res[:p_h, :p_w], res[:p_h, p_w: 2 * p_w], res[:p_h, 2* p_w:], res[p_h:, :p_w], res[p_h:, p_w: 2 * p_w]]:
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(patch)
            patch_val.append(max_val)
        return patch_val
        


    def check_tags(self, cut):
        # pre processing
        # print(tags)
        # print(cut.shape)
        height = 112
        ratio = cut.shape[0] / height
        new_size = (int(cut.shape[1] / ratio), height)
        cut = cv2.resize(cut, new_size)
        p_order = np.array([self.query_tag(tag, cut) for tag in self.tags])
        # print(p_order, p_order.shape)
        text_index = np.argmax(p_order, axis=0)
        # 提出文字
        t_order = [self.tags[i] for j, i in enumerate(text_index) if p_order[i, j] > self.thre]
        
        text = ''
        # print(t_order)
        n = len(t_order)
        if n == 5:
            text = ' '.join(t_order[:3]) + '\n' + ' '.join(t_order[3:])
            # print(np.max(p_order, axis=0))
        elif n == 0:
            text = '未检测到Tag'
            return '', '', '', 0
        else:
            text = '检测数量异常' + ' '.join(t_order)
            return '', '', '', 0
        

        rules = {} # 匹配规则
        # t_order.append('高级资深干员')
        for tag in t_order:
            if '高级资深干员' in t_order:
                rules[tag] = set(self.dbs[tag]) # 如果有高资则启动高资
            else:
                rules[tag] = set([i for i in self.dbs[tag] if i[1] != 6]) # 否则滤除高资
        
        for i in range(2, 4):
            for c in combinations(t_order, i): # 从2-4个标签组合
                # 取出2 or 3个标签的交集
                res = rules[c[0]].copy()
                for t in c[1:]:
                    res &= rules[t] # intersection
                # 如果没有选择高资就不显示6星
                if '高级资深干员' not in c:
                    d6 = {opt for opt in res if opt[1] == 6}
                    res -= d6
                    # print(d6)
                if len(res):
                    # 如果有可用组合则使用
                    rules[tuple(c)] = res
        
        def get_level_len(opts):
            min_level = 6
            for _, level in opts:
                if level == 1:
                    level = 4
                min_level = min(min_level, level)
            return -min_level, len(opts)
                
        
        l_rules = [(i, list(rules[i]), get_level_len(rules[i])) for i in rules]
        l_rules.sort(key=lambda x: x[2])
        
        r_rules = [] # 稀有的公招数量
        t_rules = [] # 一般的公招的数量
        for rule in l_rules:
            if isinstance(rule[0], tuple):
                tgs = '+'.join(rule[0])
            else:
                tgs = rule[0]
            rule[1].sort(key=lambda x: x[1])
            opts = ','.join([f'{i[0]}({i[1]})' for i in rule[1]])
            
            if '(2)' in opts or '(3)' in opts:
                t_rules.append(tgs + ': ' + opts)
            else:
                # 稀有
                r_rules.append(tgs + ': ' + opts)
            
        
        # print(l_rules)
        if len(r_rules): 
            return text, '\n'.join(r_rules), '', len(r_rules)
        else:
            return text, '', '\n'.join(t_rules), len(t_rules)
            

class GameManager:
    def __init__(self, config):
        # 链接shell
        self.shell_pipe = AdbDeviceTcp(config['adb_ip'], config['adb_port'], default_transport_timeout_s=9.)
        self.shell_pipe.connect()
        
        # roc.set_boot_state()
        self.check_interval = 1 / config['check_fps']
        self.cfg = config
        self.img_dict = {}
        self.load_img()
        self.run = True
        self.src_img = None
        self.text = "加载中……"
        self.bbox = {"pt": []}
        self.repeat = 0
        if "repeat" in config:
            self.repeat = config["repeat"]
        self.pause_game = False
        self.thread = threading.Thread(target=GameManager.check_loop, args=(self,))
        self.thread.start()
    
    def get_repeat(self):
        if self.repeat < 0:
            return "循环：无限次"
        if self.repeat == 0:
            return "循环结束了"
        else:
            return f"循环：还有{self.repeat:d}次"
    
    def _adb_send_touch(self, px, py):
        cmd = "input tap %d %d" % (px, py)
        # d = random.randint(500, 1500)
        # cmd = "input swipe %d %d %d %d %d" % (px, py, px + random.randint(-10, 10), py + random.randint(-10, 10), d)
        self._adb_send_cmd([cmd])
        # time.sleep(d / 1000 + 0.1)
        return 0
        
    def _adb_send_cmd(self, cmd):
        self.shell_pipe.shell(' '.join(cmd))
    
    # 点击一个区域
    def screen_mouse_touch_area(self, rect):
        x = random.uniform(rect[0], rect[2])
        y = random.uniform(rect[1], rect[3])
        return self.screen_mouse_press_point((x, y))
    
    
    def screen_mouse_press_point(self, pt):
        px = round(self.src_img.shape[1] * pt[0])
        py = round(self.src_img.shape[0] * pt[1])
        self.bbox["pt"] = [(px, py)]
        px = round(self.cfg['adb_shape'][0] * pt[0])
        py = round(self.cfg['adb_shape'][1] * pt[1])
        
        return self._adb_send_touch(px, py)
    
    
    def load_img(self):
        self.img_dict = {}
        for item in self.cfg['data']:
            name = item['file']
            img = cv2.imread('./img/' + name, 0)
            w = self.cfg['match_width']
            h = int(w * img.shape[0] / img.shape[1])
            img = cv2.resize(img, (w, h))
            # print(img.shape)
            self.img_dict[name] = img
    
    
    def check_loop(self):
        time.sleep(1)
        while self.run:
            if self.pause_game:
                # self.text = "已暂停"
                continue
            if self.src_img is not None:
                src = self.src_img.copy()
                text_list = []
                
                do_action = False
                for item in self.cfg['data']:
                    name = item['file']
                    tgt = self.img_dict[name]
                    thre = item['thre']
                    action = item['action']
                    area = item['area']
                    if 'comment' in item:
                        name = item['comment']
                    if item['enable']:
                    
                        def get_patch(img, area):
                            h, w = img.shape[:2]
                            return img[int(h * area[1]): int(h * area[3]), int(w * area[0]): int(w * area[2])]
                        
                        tgt = get_patch(tgt, area)
                        src_c = cv2.resize(get_patch(src, area), (tgt.shape[1], tgt.shape[0]))
                        
                        # print(tgt.shape, src_c.shape)
                        # cv2.imshow(name, np.hstack([src_c, tgt]))
                        # cv2.waitKey(1)
                        res = cv2.matchTemplate(src_c, tgt, cv2.TM_CCOEFF_NORMED)
                        val = np.max(res)
                        
                        
                        # print("Check " + name, val, thre)
                        tmp_text = f"{name}: {val:.2f}({thre:.2f})"
                        
                        if 'count' in item:
                            tmp_text += f"[{item['count']}]"
                            if self.repeat == 0: # skip this action
                                text_list.append(tmp_text + ' X')
                                continue
                            repeat_dif = item['count']
                        else:
                            repeat_dif = 0

                        if val > thre and not do_action:
                            # cv2.imshow(name, src_c)
                            # cv2.waitKey(10)
                            # print(name, "checked")
                            tmp_text += " √"
                            self.screen_mouse_touch_area(item['action'])
                            do_action = True
                            if self.repeat >= 0:
                                self.repeat += repeat_dif
                            
                        text_list.append(tmp_text)
                    """
                    else:
                        text_list.append(f"{name}: disabled ({thre:.2f})")
                    """
                self.text = '\n'.join(text_list)
                
                if not do_action:
                    self.bbox["pt"] = []
            else:
                self.text = "Image Not Found"
                self.bbox["pt"] = []
                # print('-' * 9)
            time.sleep(self.check_interval)
            
    
    def check_img(self, src):
        self.src_img = src
        
    def set_config(self, cfg):
        self.cfg = cfg
        self.load_img()
        print("Config updated")
        
    def close(self):
        self.run = False
        self.thread.join()
        if self.shell_pipe:
            self._adb_send_cmd(['exit'])
            self.shell_pipe = None
        # self.con.close()
        

def main(cfg):
    # Windows
    root = Tk()
    # Create a frame
    app = Frame(root)
    app.pack()

    # Create a label in the frame
    lmain = Label(app)
    lmain.pack()
    
    ldtag1 = Label(app, font=tkFont.Font(size=15, weight=tkFont.BOLD))
    ldtag1.pack()
    lentry = Entry(app) 
    lentry.pack()
    ldtag = Label(app, font=tkFont.Font(size=15, weight=tkFont.BOLD))
    ldtag.pack()
    ldres = Message(app, width=cfg['recruitment']['width'], font=tkFont.Font(size=15, weight=tkFont.NORMAL))
    ldres.pack()
    
    root.title('AutoArk')
    # root.geometry('1300x760')
    target_name = cfg['name']
    td = TagDetector(cfg['recruitment']['thre'])
    gm = GameManager(cfg)
    scale = cfg['scale']

    save_img = False
    
    def fresh_repeat():
        try:
            gm.repeat = int(lentry.get())
        except:
            pass

    def onKeyPress(event):
        nonlocal save_img
        # print(event)
        if event.keysym in 'rR':
            cfg = load_cfg()
            gm.set_config(cfg)
        elif event.keysym in 'qQ':
            root.quit()
        elif event.keysym in 'sS':
            save_img = True
        elif event.keysym in 'pP' or event.keycode == 32:
            gm.pause_game = ~gm.pause_game
            if gm.pause_game:
                gm.text = "已暂停"
            else:
                gm.text = "恢复中"
        if event.widget == lentry:
            # print(event.char, event.keysym)
            if event.char == '\r':
                fresh_repeat()

    def get_stick(des, win):
        words = des.split(',')
        value = 0
        for w in words:
            if w in ['top', 'left', 'width', 'height']:
                value += win[w]
            else:
                value += int(w)
        return value

    root.bind('<KeyPress>', onKeyPress)
    
    display_interval = int(1000 / cfg['display_fps'])
    
    img_cache = None
    img_diff_count = 0
    
    def check_img_diff(img):
        nonlocal img_diff_count
        if img_cache is None:
            return True
        # print(img.shape, img_cache.shape)
        if img.shape != img_cache.shape:
            # print(img.shape, img_cache.shape)
            return True
            
        p_diff = np.sum((img.astype(np.float32) - img_cache.astype(np.float32)) != 0)
        if p_diff > 10000:
            if img_diff_count < cfg['display_fps'] / 5:
                img_diff_count += 1
                return False
            else:
                img_diff_count = 0
                return True
        else:
            img_diff_count = 0
            return False
    
    with mss.mss() as m:
        def capture_stream():
            nonlocal save_img
            nonlocal img_cache
            win_info = get_window_roi(target_name, [0, 0, 1, 1], cfg['padding'])
            if win_info['left'] < 0 and win_info['top'] < 0:
                ldtag1.configure(text='未检测到窗口')
                ldtag.configure(text='')
                ldres.configure(text='')
                img_cache = None
            else:
                full_win = get_window_roi(target_name,[0, 0, 1, 1], [0, 0, 0, 0])
                if len(cfg['stick']) == 2:
                    root.geometry(f"+{get_stick(cfg['stick'][0], full_win)}+{get_stick(cfg['stick'][1], full_win)}")
                img = np.array(m.grab(win_info))
                
                # ArkQuery
                tag_query = crop_image_by_pts(img[:, :, 2], cfg['recruitment']['area'])
                pil_img = Image.fromarray(tag_query)
                results = td.check_tags(tag_query.copy())
                if results[3] > 0: # 存在tag组合
                    imgtk = ImageTk.PhotoImage(image=pil_img)
                    lmain.imgtk = imgtk
                    lmain.configure(image=imgtk)
                    ldtag1.configure(text=results[0])
                    ldtag.configure(text=results[1])
                    ldres.configure(text=results[2])
                    
                else:
                    # AutoArk
                    results = gm.check_img(img[:, :, 2])
                    """
                    draw vis
                    """
                    for pt in gm.bbox["pt"]:
                        img = cv2.circle(img, pt, 5, (0, 0, 255), -1)

                    ldtag1.configure(text=gm.get_repeat())
                    ldtag.configure(text='')
                    ldres.configure(text=gm.text)
                    
                    pil_img = Image.fromarray(img[:, :, :3][:, :, ::-1])
                    if scale > 0:
                        pil_img = pil_img.resize((int(pil_img.size[0] * scale), int(pil_img.size[1] * scale)))
                        imgtk = ImageTk.PhotoImage(image=pil_img)
                        lmain.imgtk = imgtk
                        lmain.configure(image=imgtk)
                        
                if save_img:
                    now = datetime.now()
                    date_time = now.strftime("./%H-%M-%S")
                    Image.fromarray(img[:, :, 2]).save(date_time + ".png")
                    save_img = False
                
                    
                # update the display
                imgtk = ImageTk.PhotoImage(image=pil_img)
                lmain.imgtk = imgtk
                lmain.configure(image=imgtk)
            lmain.after(display_interval, capture_stream) 

        capture_stream()
        root.mainloop()
        
    gm.close()


def usage():
    print("AutoArk操作说明:\n空格/P:暂停自动\nS:保存当前截图\nR:重新加载配置\nQ:退出\n检测到公招Tag后会自动输出方案\n" + '-'*8)


if __name__ == '__main__':
    usage()
    cfg = load_cfg()
    main(cfg)
