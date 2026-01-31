**Training details:**

- 5,000+ hours of speech data
- 2,000+ hours of noise data
- Trained for 2.2 million steps
- Used evolutionary architecture search guided by predicted human ratings

**Results with hearing-impaired users:**

| Noise Type | Improvement (dB) |
|------------|------------------|
| Speech-shaped noise | -3.5 dB SRT |
| Restaurant noise | -3.5 dB SRT |
| Traffic noise | -2.8 dB SRT |

(Lower SRT = better. These improvements brought hearing-impaired users to normal-hearing levels.)

**The key insight:**

Traditional hearing aid noise reduction doesn't improve speech intelligibility - it just makes things more comfortable. This deep learning approach actually helps people understand speech better.

**What I'd like to implement:**

- Real-time noise suppression for the audio stream
- Speaker separation when multiple people are talking
- Automatic volume normalization

---

## Datasets Referenced

### AffectNet

**Link:** http://mohammadmahoor.com/affectnet/

- 450,000+ images
- 7 emotion categories + valence/arousal annotations
- Collected from the internet (real-world conditions)
- Highly imbalanced (lots of happy faces, few disgusted)

### RAF-DB (Real-world Affective Faces)

**Link:** http://www.whdeng.cn/raf/model1.html

- 29,672 images
- 7 emotion categories
- Includes 5 accurate + 37 automatic facial landmarks
- More controlled than AffectNet

### FER2013

**Link:** Available on Kaggle

- 35,887 images
- 48 × 48 pixel grayscale
- Good for quick prototyping
- Lower quality than AffectNet/RAF-DB

### Extended Cohn-Kanade (CK+)

**Link:** Available through CMU

- 593 sequences from 123 subjects
- Lab-controlled conditions
- Gold standard for benchmarking
- Limited diversity

---

## How This Research Shaped the Project

| Research Finding | Implementation |
|------------------|----------------|
| SE-ResNet focuses on mouth/nose | We crop face region tightly and send to Gemini at high resolution |
| AU12 predicts positive emotion | Gemini prompt specifically asks for AU12 detection |
| AU4 predicts negative emotion | Gemini prompt asks for AU4 (brow furrow) |
| Transfer learning helps | Using Gemini's pre-trained vision (massive training data) |
| BlazePose tracks 33 landmarks | BodyLanguageAnalyzer extracts shoulder/hip positions |
| Shoulder width indicates openness | We calculate shoulder_width / hip_width ratio |
| Forward lean shows engagement | We calculate torso angle from shoulder-hip midpoints |
| U-Net improves speech clarity | Future work: add audio denoising pipeline |

---

## Citation 

Huang, Q., Huang, C., Wang, X., & Jiang, F. (2023). Facial expression recognition 
based on fusion feature of CNN and SE-ResNet. Scientific Reports, 13, 7086.

Haines, N., Southward, M. W., Cheavens, J. S., Beauchaine, T. P., & Ahn, W. Y. (2019). 
Using computer-vision and machine learning to automate facial coding of positive and 
negative affect intensity. PLOS ONE, 14(2), e0211735.

Bazarevsky, V., et al. (2020). BlazePose: On-device Real-time Body Pose tracking. 
CVPR Workshop on Computer Vision for Augmented and Virtual Reality.

Diehl, P. U., et al. (2023). Restoring speech intelligibility for hearing aid users 
with deep learning. Scientific Reports, 13, 2719.
