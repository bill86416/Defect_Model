# Defect_Model

# Install Environmnet

```
conda env create -f env.yml
```

# Training
Train Regular Model
```
python3 normal_model.py
```

Train Normal Noise Model (Add Noise to Input Images)
```
python3 noise_model.py
```

Train LogNormal Noise Model (Add Noise to Input Images)
Without High Curvature Thrshold
```
python3 lognormal_noise_train.py
```

With High Curvature Thrshold
```
python3 lognormal_noise_train_thres.py
```

# Testing
Test Model Without Noise
```
python3 test.py
```

Test Model with Additive Normal Noise (All layers)
```
python3 test_normal_add.py
```

Test Model with Additive Normal Noise (All Conv layers)
```
python3 test_normal_add_after_conv.py
```

Test Model with "Scaling" Additive Normal Noise (All Conv layers)
```
python3 test_normal_add_after_conv_with_zeta.py
```

Test Model with Multiplicative Normal Noise (All layers)
```
python3 test_normal_mul.py
```

Test Model with LogNormal Noise (All weightss)
```
python3 test_lognormal.py
```
