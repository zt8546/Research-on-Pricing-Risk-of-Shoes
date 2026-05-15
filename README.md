# Research on Pricing Risk of Shoes

This repository contains the code and assets for an undergraduate graduation project on second-hand shoe pricing risk identification.

## Project Goal

The system takes one shoe image and a seller description as input, then outputs:

- Authenticity estimation
- Brand identification
- Shoe type classification
- Condition score
- Suggested reference price
- Text honesty score
- Risk level

## Project Structure

- `final_result/`: main Flask web system for inference and pricing
- `AI_Detector/`: authenticity image detection training project
- `dirty_bad_shoes/`: shoe condition training project
- `shoes_brands/`: brand classification training project
- `Types-of-Shoes-main/`: shoe type classification training project
- `price/`: pricing-related resources

## Dataset

Due to the large file size (~2.1GB), the training dataset is hosted externally:

- **Baidu Netdisk**: [通过网盘分享的文件：Research-on-Pricing-Risk-of-Shoes
链接: https://pan.baidu.com/s/19m3OMBkwnUkPLntqvFW_IQ?pwd=Wisa 提取码: Wisa 
--来自百度网盘超级会员v5的分享] (Extraction Code: `Wisa`)

Please download and extract the content into the respective training folders before running the scripts.

## Main System

Run the web application locally:

```powershell
cd final_result
python app.py
```

Open [http://127.0.0.1:9008](http://127.0.0.1:9008) in your browser.

## Model Notes

- The inference system in `final_result/` depends on the trained model files already stored in the project.
- Training code is kept in the four training subprojects and was preserved as experiment code for the thesis.
- Some very large data archives are excluded from Git tracking to keep the GitHub repository publishable.
- `.keras` model files are tracked with Git LFS because GitHub does not accept regular Git blobs larger than 100 MB.

## Environment

- Python 3.x
- TensorFlow 2.x
- Flask
- NumPy
- Pillow
- Matplotlib
- Seaborn
- scikit-learn

## Thesis Context

The repository includes both the final adopted methods and baseline comparison code used for the experimental section of the thesis:

- Brand recognition: EfficientNet-B0 transfer learning
- Shoe type classification: MobileNetV2 transfer learning
- Condition recognition: EfficientNet-B0 transfer learning
- Authenticity detection: CNN-based binary classifier

## License

This repository is released under the MIT License.
