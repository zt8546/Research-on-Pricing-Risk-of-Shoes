from flask import Flask, render_template, request, url_for, send_from_directory
import os, traceback
from model import predict_integrated, analyze_integrity, get_price_info

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# 启动时自动创建 uploads 目录，防止首次运行报错
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    user_desc, img_name = "", ""
    error_msg = None

    if request.method == 'POST':
        file        = request.files.get('file')
        user_desc   = request.form.get('user_desc', '')
        old_img_name = request.form.get('old_img_name', '')
        target_path = None

        if file and file.filename != '':
            img_name    = file.filename
            target_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
            file.save(target_path)
        elif old_img_name:
            img_name    = old_img_name
            target_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)

        if target_path and os.path.exists(target_path):
            try:
                s_type, conf, a_res, a_p, h_s, h_t, h_color, b_name, b_conf = predict_integrated(target_path)
                p_range, s_formula = get_price_info(b_name, s_type, h_s, a_p)
                i_score, i_msg     = analyze_integrity(user_desc, s_type, h_s, a_res, b_name, p_range)

                return render_template('index.html',
                    result=a_res,
                    res_color=('success' if a_res == '真实照片' else 'danger'),
                    prediction_percentage=f"{a_p * 100:.2f}",
                    shoe_type=s_type,      shoe_reliability=f"{conf * 100:.2f}",
                    brand_name=b_name,     brand_reliability=f"{b_conf * 100:.2f}",
                    health_score=f"{h_s:.2f}", health_status=h_t, health_color=h_color,
                    integrity_score=i_score,   integrity_msg=i_msg,
                    suggested_price=f"¥ {p_range[0]} - {p_range[1]}",
                    price_formula=s_formula,
                    img_name=img_name,     user_text=user_desc
                )
            except Exception:
                # 打印完整报错到终端，同时在页面上给用户提示
                traceback.print_exc()
                error_msg = "模型推理出错，请检查终端日志。"
        else:
            error_msg = "请先上传一张图片。"

    return render_template('index.html',
        user_text=user_desc, img_name=img_name, error_msg=error_msg)

if __name__ == '__main__':
    app.run(port=9008, debug=False)