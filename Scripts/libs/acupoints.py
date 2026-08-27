# -*- coding: utf-8 -*-
"""穴位关键点定义：37 个背部穴位名称、中文、连接关系、配色。"""

ACUPOINT_NAMES_EN = [
    "dazhui",
    "jianjing_L", "jianjing_R",
    "naoshu_L", "naoshu_R",
    "jianzhen_L", "jianzhen_R",
    "dazhu_L", "dazhu_R",
    "fengmen_L", "fengmen_R",
    "feishu_L", "feishu_R",
    "jueyinshu_L", "jueyinshu_R",
    "xinshu_L", "xinshu_R",
    "gaohuang_L", "gaohuang_R",
    "tianzong_L", "tianzong_R",
    "geshu_L", "geshu_R",
    "ganshu_L", "ganshu_R",
    "danshu_L", "danshu_R",
    "pishu_L", "pishu_R",
    "weishu_L", "weishu_R",
    "sanjiaoshu_L", "sanjiaoshu_R",
    "shenshu_L", "shenshu_R",
    "dachangshu_L", "dachangshu_R",
]

ACUPOINT_NAMES_CN = [
    "大椎",
    "左肩井", "右肩井",
    "左臑俞", "右臑俞",
    "左肩贞", "右肩贞",
    "左大杼", "右大杼",
    "左风门", "右风门",
    "左肺俞", "右肺俞",
    "左厥阴俞", "右厥阴俞",
    "左心俞", "右心俞",
    "左膏肓", "右膏肓",
    "左天宗", "右天宗",
    "左膈俞", "右膈俞",
    "左肝俞", "右肝俞",
    "左胆俞", "右胆俞",
    "左脾俞", "右脾俞",
    "左胃俞", "右胃俞",
    "左三焦俞", "右三焦俞",
    "左肾俞", "右肾俞",
    "左大肠俞", "右大肠俞",
]

assert len(ACUPOINT_NAMES_EN) == 37
assert len(ACUPOINT_NAMES_CN) == 37

ACUPOINT_CN_MAP = dict(zip(ACUPOINT_NAMES_EN, ACUPOINT_NAMES_CN))

ACUPOINT_SKELETON = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (3, 5), (4, 6),
    (1, 7), (2, 8),
    (7, 9), (8, 10),
    (9, 11), (10, 12),
    (11, 13), (12, 14),
    (13, 15), (14, 16),
    (15, 17), (16, 18),
    (11, 19), (12, 20),
    (19, 21), (20, 22),
    (21, 23), (22, 24),
    (23, 25), (24, 26),
    (25, 27), (26, 28),
    (27, 29), (28, 30),
    (29, 31), (30, 32),
    (31, 33), (32, 34),
    (33, 35), (34, 36),
    (7, 0), (8, 0),
]

LEFT_INDICES = [i for i in range(37) if ACUPOINT_NAMES_EN[i].endswith("_L")]
RIGHT_INDICES = [i for i in range(37) if ACUPOINT_NAMES_EN[i].endswith("_R")]
CENTER_INDICES = [i for i in range(37) if not ACUPOINT_NAMES_EN[i].endswith("_L")
                  and not ACUPOINT_NAMES_EN[i].endswith("_R")]

COLOR_LEFT = (255, 0, 0)
COLOR_RIGHT = (0, 0, 255)
COLOR_CENTER = (0, 255, 0)
COLOR_BOX = (0, 255, 255)
COLOR_SKELETON = (200, 200, 200)


def color_for(idx):
    if idx in LEFT_INDICES:
        return COLOR_LEFT
    if idx in RIGHT_INDICES:
        return COLOR_RIGHT
    return COLOR_CENTER