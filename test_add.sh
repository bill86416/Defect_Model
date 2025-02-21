#!/bin/bash
# python3 test_normal_add.py --noise 0.0
# python3 test_normal_add_after_conv.py --noise 0.1
# python3 test_normal_add_after_conv.py --noise 0.2
# python3 test_normal_add_after_conv.py --noise 0.3

python3 test_normal_add_after_conv_with_zeta.py --noise 0.3
python3 test_normal_add_after_conv_with_zeta.py --noise 0.5
python3 test_normal_add_after_conv_with_zeta.py --noise 1.0
