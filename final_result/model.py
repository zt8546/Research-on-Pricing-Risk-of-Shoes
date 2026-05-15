import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import re

current_dir = os.path.dirname(os.path.abspath(__file__))

# 1. 模型加载
brand_net    = load_model(os.path.join(current_dir, "best_brand_model.h5"), compile=False)
classify_net = load_model(os.path.join(current_dir, "best_shoes_classifier.h5"), compile=False)
aigc_net     = load_model(os.path.join(current_dir, 'deepfake_detector_model_best.keras'), compile=False)
health_net   = load_model(os.path.join(current_dir, "best_transfer_model.h5"), compile=False)

BRAND_MAP = {
    0: "安踏 (Anta)", 1: "李宁 (LiNing)", 2: "阿迪达斯 (adidas)",
    3: "匡威 (converse)", 4: "耐克 (nike)"
}
CLASS_MAP = {
    0: "运动鞋", 1: "平底鞋", 2: "套脚鞋", 3: "凉鞋",
    4: "高跟鞋", 5: "靴子", 6: "橡胶拖鞋", 7: "棉拖鞋"
}

# 2. 视觉预测
def predict_integrated(path):
    img_b = image.load_img(path, target_size=(224, 224))
    x_b = image.img_to_array(img_b)
    res_brand  = brand_net.predict(np.expand_dims(x_b, axis=0), verbose=0)
    brand_name = BRAND_MAP[np.argmax(res_brand)]
    brand_conf = float(np.max(res_brand))

    img_c = image.load_img(path, target_size=(224, 224))
    x_c   = image.img_to_array(img_c) / 255.0
    res_class    = classify_net.predict(np.expand_dims(x_c, axis=0), verbose=0)
    shoe_type    = CLASS_MAP[np.argmax(res_class)]
    classify_conf = float(np.max(res_class))

    img_a  = image.load_img(path, target_size=(128, 128))
    x_a    = image.img_to_array(img_a)         
    res_aigc    = aigc_net.predict(np.expand_dims(x_a, axis=0), verbose=0)
    prediction  = float(res_aigc[0][0])
    aigc_text   = '真实照片' if prediction >= 0.95 else 'AI生成图片' if prediction <= 0.05 else '可能生成'

    img_h  = image.load_img(path, target_size=(224, 224))
    x_h    = image.img_to_array(img_h)
    res_health = health_net.predict(np.expand_dims(x_h, axis=0), verbose=0)
    h_score    = float(res_health[0][0]) * 100

    if h_score >= 90:
        h_status, h_color = "优质", "success"
    elif h_score >= 70:
        h_status, h_color = "良好", "success"
    elif h_score >= 60:
        h_status, h_color = "尚可", "warning"
    elif h_score >= 50:
        h_status, h_color = "一般", "warning"
    else:
        h_status, h_color = "破损", "danger"

    return shoe_type, classify_conf, aigc_text, prediction, h_score, h_status, h_color, brand_name, brand_conf



_NEG = ("不", "非", "没")
_SEP = "，,。.；;、！!？?\n "
# 含"不"的褒义词
_POS = ("不错", "不赖", "不差", "不坏")

def _kw_ok(text, pattern, flags=0):
    """搜 pattern：命中项所在分句前段含否定词则跳过，有有效命中返回 True"""
    for m in re.finditer(pattern, text, flags):
        seg = text[:m.start()]
        cut = max((seg.rfind(c) for c in _SEP), default=-1)  
        clause = seg[cut + 1:]
        for p in _POS:                      
            clause = clause.replace(p, "")
        if not any(n in clause for n in _NEG):
            return True
    return False


def analyze_integrity(text, m_type, m_score, m_aigc_text, m_brand, price_range):
    if not text or len(text.strip()) < 2:
        return 0, "<span style='color:red;'>描述缺失 (0分)</span>"

    score = 100
    details = []
    p_low, p_high = price_range

    real_kws = r"真|实拍|真实|真图|原图|无修|非AI|非ai|不是AI|不是ai|无滤镜|实物|现货|未经修改|未经处理|纯实拍|纯原图|手机实拍|相机实拍|现场实拍|实物拍摄|现货拍摄|真实拍摄|原图拍摄|实拍原图|实拍真图|实拍无修|原图无修|真实无修|实物无修"
    ai_kws   = r"(?<!不是)(?<!非)(?<!不是 )(AI生成|ai生成|合成图|效果图|非实物图|虚拟渲染|AI作图|ai作图|AI绘图|ai绘图|AI合成|ai合成|算法生成|虚拟生成|智能作图|数字生成|人工合成|模型生成|机器生成|AI渲染|算法渲染|非实拍|非原图|非真实|非实物|虚拟图像|数字图像|算法图像|AI图像|ai图像|生成图像|伪造图像|AI制作|ai制作|算法制作|虚拟制作|AI创作|ai创作|假)"
    has_real = re.search(real_kws, text)
    has_ai   = re.search(ai_kws, text)

    if m_aigc_text == '真实照片' and has_real:
        details.append("<span style='color:green;'>真伪诚实 (不扣分)</span>")
    elif m_aigc_text != '真实照片' and has_ai:
        details.append("<span style='color:green;'>真伪诚实 (不扣分)</span>")
    else:
        reason = "真伪欺瞒" if (has_real or has_ai) else "真伪未提"
        score = max(0, score - 40)
        details.append(f"<span style='color:red;'>{reason} (-40分)</span>")

    brand_reg = {
        "耐克 (nike)":      r"耐克|nike|Nike|aj|jordan",
        "阿迪达斯 (adidas)": r"阿迪|adidas|Adidas|yeezy|椰子",
        "李宁 (LiNing)":    r"李宁|lining|Lining",
        "安踏 (Anta)":      r"安踏|anta|Anta",
        "匡威 (converse)":  r"匡威|converse|Convrse"
    }
    if m_brand in brand_reg and _kw_ok(text, brand_reg[m_brand], re.I):
        details.append("<span style='color:green;'>品牌诚实 (不扣分)</span>")
    else:
        reason = "品牌错误" if any(_kw_ok(text, r, re.I) for r in brand_reg.values()) else "品牌未提"
        score = max(0, score - 40)
        details.append(f"<span style='color:red;'>{reason} (-40分)</span>")

    type_reg = {
        "运动鞋":   r"运动鞋|跑步鞋|球鞋|篮球鞋|足球鞋|网球鞋|训练鞋|健身鞋|滑板鞋|潮鞋|老爹鞋",
        "平底鞋":   r"帆布鞋|平底鞋|平跟鞋|板鞋|休闲鞋|德比鞋|牛津鞋|硫化鞋|玛丽珍鞋|芭蕾舞鞋",
        "套脚鞋":   r"套脚鞋|一脚蹬|懒人鞋|穆勒鞋|豆豆鞋|船鞋|乐福鞋",
        "凉鞋":     r"凉鞋",
        "高跟鞋":   r"高跟鞋|细跟鞋|粗跟鞋|坡跟鞋|松糕鞋|厚底鞋",
        "靴子":     r"靴子|马丁靴|切尔西靴|工装靴|机车靴|雪地靴|长靴|中筒靴|短靴",
        "橡胶拖鞋": r"拖鞋|凉拖|塑胶拖鞋|人字拖|一字拖|洞洞鞋|橡胶拖",
        "棉拖鞋":   r"棉拖|毛绒拖鞋|居家拖鞋|保暖拖鞋"
    }
    if m_type in type_reg and _kw_ok(text, type_reg[m_type]):
        details.append("<span style='color:green;'>种类诚实 (不扣分)</span>")
    else:
        reason = "种类错误" if any(_kw_ok(text, r) for r in type_reg.values()) else "种类未提"
        score = max(0, score - 40)
        details.append(f"<span style='color:red;'>{reason} (-40分)</span>")

    actual_lv = 5 if m_score >= 90 else 4 if m_score >= 70 else 3 if m_score >= 60 else 2 if m_score >= 50 else 1

    lv5_kws = r"全新|优质|优|100%新|崭新|崭新如初|完美|完美无瑕|无瑕|极品|零磨损|零瑕疵|零划痕|零破损|零使用痕迹|无任何瑕疵|未上脚|未穿着|未使用|全新未开封|全新无磨损|全新无瑕疵|绝对全新|绝对完美|真正全新|纯全新|全新到极致|完美到极致|几乎全新|九成九五新|九五新|九成八新|接近全新|近乎全新|准全新|基本全新|几乎未穿|几乎未用|极轻微磨损|极轻微痕迹|几乎无磨损|几乎无瑕疵|外观如新|内里如新|鞋底如新|无明显磨损|无明显瑕疵|品相极佳|成色极佳|状态极佳|保养极佳|仅试穿未出门|试穿一次|几乎未上脚|近乎全新状态|接近全新状态"
    lv4_kws = r"良好|九成新|八成新|品相良好|成色良好|状态良好|正常穿着痕迹|轻微磨损|轻微瑕疵|小瑕疵|轻微划痕|轻微擦痕|轻微蹭痕|轻微污渍|少许磨损|少许瑕疵|些许磨损|些许瑕疵|轻微脱色|轻微变形|轻微开胶|轻微脱线|无严重磨损|无严重瑕疵|外观整洁|轻度磨损|轻度瑕疵|保养尚可|基本完好|大致完好|无大面积磨损|较好|还行"
    lv3_kws = r"尚可|还行|七成新|六成新|成色中等|中等|轻微开胶|轻微脱线|轻微变形|轻微脱色|略有破损中度磨损|边角磨损|边缘磨损|鞋内磨损|鞋跟磨损|小面积污渍|小面积划痕|小面积破损|成色偏旧|品相偏旧|状态偏旧|整体偏旧|外观偏旧"
    lv2_kws = r"五成新|成色一般|品相一般|明显使用痕迹|重度使用痕迹|局部破损|局部污渍|局部划痕|鞋面瑕疵|鞋边磨损|鞋底磨痕|鞋头磨损|穿着痕迹明显|中度污渍|中度划痕|中度破损|明显旧化|旧化明显|中度旧化|局部开胶|局部脱胶|局部脱线|轻微裂纹|表面磨损|表面瑕疵|成色不佳|品相不佳|状态不佳|整体不佳|外观不佳|明显磨损|明显瑕疵|中度瑕疵|明显划痕|明显擦痕|明显蹭痕|明显污渍|较多磨损|较多瑕疵|明显脱色|中度脱色|明显变形|中度变形|明显开胶|中度开胶|明显脱线|局部磨损|局部瑕疵|鞋面磨损|鞋底磨损|磨损明显|瑕疵明显"
    lv1_kws = r"差|破损|较差|四成新|三成新|二成新|一成新|品相较差|成色较差|状态较差|严重磨损|重度磨损|深度磨损|大面积磨损|严重瑕疵|大面积划痕|大面积污渍|顽固污渍|严重开胶|完全开胶|严重开裂|完全开裂|脱线|断底|断面|断帮|脱胶|掉底|掉面|掉帮|破洞|大面积破洞|扭曲变形|严重老化|脱漆|掉漆|陈旧|老旧|破旧|老化严重|磨损严重|瑕疵严重|破损严重"

    user_lv = -1
    if _kw_ok(text, lv5_kws):   user_lv = 5
    elif _kw_ok(text, lv4_kws): user_lv = 4
    elif _kw_ok(text, lv3_kws): user_lv = 3
    elif _kw_ok(text, lv2_kws): user_lv = 2
    elif _kw_ok(text, lv1_kws): user_lv = 1

    if user_lv == -1:
        score = max(0, score - 60)
        details.append("<span style='color:red;'>成色未提 (-60分)</span>")
    else:
        diff = user_lv - actual_lv
        if diff <= 0:
            label = "成色诚实" if diff == 0 else "成色描述谦虚"
            details.append(f"<span style='color:green;'>{label} (不扣分)</span>")
        elif diff == 1:
            score = max(0, score - 20)
            details.append("<span style='color:orange;'>成色偏差1级 (-20分)</span>")
        elif diff == 2:
            score = max(0, score - 40)
            details.append("<span style='color:orange;'>成色偏差2级 (-40分)</span>")
        else:
            score = max(0, score - 60)
            details.append("<span style='color:red;'>成色偏差过大 (-60分)</span>")

    price_match = re.search(r"(\d+)", text)
    if price_match:
        u_p = float(price_match.group(1))
        lower_bound = p_low * 0.9
        upper_bound = p_high * 1.1
        if lower_bound <= u_p <= upper_bound:
            details.append("<span style='color:green;'>价格诚实 (不扣分)</span>")
        else:
            if u_p < lower_bound:
                diff_ratio = (lower_bound - u_p) / max(1.0, lower_bound)
                reason = "价格偏低"
            else:
                diff_ratio = (u_p - upper_bound) / max(1.0, upper_bound)
                reason = "价格偏高"
            over_pct = diff_ratio * 100
            deduction = min(int(over_pct), 40)
            score = max(0, score - deduction)
            color = "red" if deduction >= 40 else "orange"
            details.append(f"<span style='color:{color};'>{reason}(偏离{over_pct:.1f}%, -{deduction}分)</span>")
    else:
        score = max(0, score - 40)
        details.append("<span style='color:red;'>价格未提 (-40分)</span>")

    risk_text  = "低风险" if score >= 75 else "中风险" if score >= 50 else "高风险"
    risk_color = "green"  if score >= 75 else "orange"  if score >= 50 else "red"
    summary = f"<b style='color:{risk_color}; font-size:28px;'>{risk_text}</b> <span style='font-size:18px;'>({score}分)</span><br>"
    return score, summary + "<div style='font-size:16px; line-height:2; margin-top:10px;'>" + "<br>".join(details) + "</div>"


# 4. 建议参考价
def get_price_info(brand, s_type, h_score, ai_prob):
    price_table = {
        "耐克 (nike)":      {"运动鞋": (650, 850), "平底鞋": (450, 600), "套脚鞋": (350, 480), "凉鞋": (300, 420), "高跟鞋": (750, 950),  "靴子": (950, 1300),  "橡胶拖鞋": (240, 350), "棉拖鞋": (160, 260)},
        "阿迪达斯 (adidas)": {"运动鞋": (600, 780), "平底鞋": (400, 550), "套脚鞋": (320, 450), "凉鞋": (280, 380), "高跟鞋": (650, 850),  "靴子": (850, 1100),  "橡胶拖鞋": (200, 300), "棉拖鞋": (140, 240)},
        "李宁 (LiNing)":    {"运动鞋": (420, 550), "平底鞋": (280, 380), "套脚鞋": (220, 320), "凉鞋": (180, 280), "高跟鞋": (400, 550),  "靴子": (550, 750),   "橡胶拖鞋": (140, 220), "棉拖鞋": (100, 160)},
        "安踏 (Anta)":      {"运动鞋": (380, 500), "平底鞋": (250, 350), "套脚鞋": (200, 280), "凉鞋": (160, 250), "高跟鞋": (350, 480),  "靴子": (500, 700),   "橡胶拖鞋": (120, 180), "棉拖鞋": (80,  140)},
        "匡威 (converse)":  {"运动鞋": (400, 550), "平底鞋": (350, 500), "套脚鞋": (250, 380), "凉鞋": (220, 350), "高跟鞋": (300, 450),  "靴子": (450, 650),   "橡胶拖鞋": (180, 280), "棉拖鞋": (120, 200)},
    }
    b_range  = price_table.get(brand, {}).get(s_type, (200, 400))
    k_health = h_score / 100.0
    if ai_prob >= 0.95:
       m_ai = 1.0
    elif ai_prob <= 0.05:
       m_ai = 0.01
    else:
       s = (0.95 - ai_prob) / 0.9          
       m_ai = round(1 - 0.99 * (s ** 3), 4)  
    res_low  = int(b_range[0] * k_health * m_ai)
    res_high = int(b_range[1] * k_health * m_ai)
    formula  = (
        f"计算过程: 区间: [ <span style='color:#1890ff;'>{b_range[0]}</span>, "
        f"<span style='color:#1890ff;'>{b_range[1]}</span> ] (基准区间) × "
        f"<span style='color:#52c41a;'>{k_health:.2f}</span> (成色系数) × "
        f"<span style='color:#faad14;'>{m_ai:.2f}</span> (真实度系数)"
    )
    return (res_low, res_high), formula