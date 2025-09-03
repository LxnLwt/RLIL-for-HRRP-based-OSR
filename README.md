# RLIL-for-HRRP-based-OSR
Code for "Radar Open Set Recognition Performance Analysis from the Perspective of Cross-Entropy Sensitivity".

Wentao Li, Shuai Li, Junyan Chen, and Biao Tian

Sun Yat-sen University

## Dataset Preparation
In train.py, we evaluate four HRRP-based OSR scenarios: Scenarios 1 and 2 utilize measured HRRPs, while Scenarios 3 and 4 employ simulated HRRPs. The partitioning of known and unknown classes for all four scenarios is summarized in Table I. The simulated HRRPs used in this study are available at https://ieee-dataport.org/documents/sysu-hrrp-aircraft-10-sha1-2. To run this code, users need to construct the OSR dataset (Scenarios 3 and 4) according to Table~I. Note that to directly use our dataloader, both the training and testing data must be saved in CSV format, with columns 1–512 containing the data and column 513 containing the labels.

<img width="501" height="558" alt="Table" src="https://github.com/user-attachments/assets/96898876-4560-4fa8-9440-8e28c6ffb3c0" />
