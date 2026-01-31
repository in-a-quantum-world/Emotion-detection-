# Emotion-detection-
An idea of mine which is still in progress! The code I have provided is an implementation in Java and Python, using the Gemini 3 API. I'm working on auditory processing at the moment, and in the future, combining into a single application for those with sensory disabilities

## Goal 
To build an accessibility application for sensory disabiltiies using voice recognition, emotion detectiona and body language analysis with Google's Gemini3 API - as part of the Gemini 3 Hackathon.


## How It Works

The app combines a few different technologies:

1. **MediaPipe** handles the computer vision stuff - detecting faces and tracking body pose in real-time
2. **Google's Gemini API** analyses the detected face and generates detailed emotion assessments
3. **OpenCV** manages the webcam feed and displays results

The emotion detection is based on the Facial Action Coding System (FACS), which breaks down expressions into specific muscle movements. For example, a genuine smile involves both the lip corners pulling up (AU12) AND the cheeks raising (AU6) - that's the difference between a real smile and a fake one.

---

## Research Background

I spent some time reading through academic papers to understand the best approaches for emotion recognition. Here's what informed this project:

### Facial Expression Recognition

The SE-ResNet architecture from Huang et al. is lightweight enough to run on mobile devices (only 11M parameters) while still achieving solid accuracy. Their key insight was using "squeeze-and-excitation" blocks that help the model focus on the most important facial features.

**Key paper:** [Facial expression recognition based on SE-ResNet](https://www.nature.com/articles/s41598-023-35446-4) - Scientific Reports, 2023

Some interesting findings from the research:
- Transfer learning makes a huge difference - pretraining on a large dataset (AffectNet) then fine-tuning on a smaller one improved accuracy by almost 27%
- The mouth and nose regions are the most important for detecting emotions (which makes sense)
- Models trained on diverse, "in-the-wild" images generalize much better than those trained on lab photos

### Action Unit Detection

The Facial Action Coding System maps specific muscle movements to emotions. This paper helped me understand which action units matter most:

**Key paper:** [Computer vision methods for affect detection](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0211735) - PLOS ONE, 2019

Their random forest model found that:
- AU12 (lip corner pull) is the strongest predictor of positive emotion
- AU4 (brow lowerer) is most associated with negative emotion
- Combining multiple AUs gives much better predictions than looking at any single one

### Body Language Analysis

For pose estimation, I'm using MediaPipe's BlazePose model which tracks 33 body landmarks. The original paper is worth a read if you're interested in how it works:

**Reference:** [BlazePose: On-device Real-time Body Pose Tracking](https://research.google/blog/on-device-real-time-body-pose-tracking-with-mediapipe-blazepose/) - Google Research

### Audio/Speech Processing (Future Work)

I also looked into speech enhancement for hearing-impaired users. This paper shows how deep learning can dramatically improve speech intelligibility in noisy environments:

**Key paper:** [Restoring speech intelligibility for hearing aid users with deep learning](https://www.nature.com/articles/s41598-023-29871-8) - Scientific Reports, 2023

Their U-Net based denoising system improved speech reception thresholds by 3-4 dB, which is a big deal for hearing aid users. I haven't implemented this yet but it's on the roadmap.
For pose estimation, I'm using MediaPipe's BlazePose model which tracks 33 body landmarks. The original paper is worth a read if you're interested in how it works:

**Reference:** [BlazePose: On-device Real-time Body Pose Tracking](https://research.google/blog/on-device-real-time-body-pose-tracking-with-mediapipe-blazepose/) - Google Research
